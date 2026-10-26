"""Разбор ошибок матчинга: типы FP/FN и экспорт в GeoJSON для QGIS.

Таксономия по результатам match() (жадное сопоставление по убыванию score):
FP:
  duplicate       — FP дублирует TP-предсказание своего класса (IoU >= iou_thr);
  localization    — FP пересекается с GT своего класса (0 < IoU < iou_thr,
                    точка/круг — в r_tol от GT), но качества на TP не хватает;
  class_confusion — то же перекрытие, но с GT другого класса;
  background      — всё остальное (ложное срабатывание на фоне).
FN:
  covered         — рядом с FN есть предсказание (IoU > 0 / центр в r_tol);
  missed          — рядом нет ничего.

Приоритет классификации FP: duplicate > localization > class_confusion >
background. errors_to_geojson пишет FP и FN одним FeatureCollection
(kind=FP/FN, error=тип, score) — открывается в QGIS поверх слоёв участка.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS

from expds.eval.metric import (
    WEB_MERCATOR_R,
    EvalConfig,
    EvalObject,
    MatchResult,
    polygon_iou,
)
from expds.io.geojson import GEOM_POINT, GEOM_POLYGON, polygon_rings_to_coords

FP_TYPES = ("duplicate", "localization", "class_confusion", "background")
FN_TYPES = ("covered", "missed")


@dataclass(frozen=True)
class ErrorReport:
    """Индексы FP/FN по типам (индексы — в порядке входных списков preds/gts)."""

    fp_types: dict[str, tuple[int, ...]] = field(default_factory=dict)
    fn_types: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def counts_fp(self) -> dict[str, int]:
        """Число FP по типам."""
        return {k: len(v) for k, v in self.fp_types.items()}

    def counts_fn(self) -> dict[str, int]:
        """Число FN по типам."""
        return {k: len(v) for k, v in self.fn_types.items()}


def format_error_report(report: ErrorReport) -> str:
    """Компактный текст отчёта для stdout/лога."""
    lines = [
        "FP:",
        *(f"  {k:>15}: {n}" for k, n in report.counts_fp().items()),
        "FN:",
        *(f"  {k:>15}: {n}" for k, n in report.counts_fn().items()),
    ]
    return "\n".join(lines)


def _r_tol_for(gts: Sequence[EvalObject], cfg: EvalConfig) -> np.ndarray:
    """Допуск r_tol в единицах CRS для каждого GT (как metric._r_tol_crs)."""
    r = np.full(len(gts), cfg.r_tol, dtype=np.float64)
    if cfg.r_tol_units == "meters" and gts:
        y = np.array([g.centroid[1] for g in gts])
        r *= np.cosh(y / WEB_MERCATOR_R)
    return r


def _overlap(p: EvalObject, g: EvalObject, r_tol: float) -> float:
    """Мера перекрытия pred с GT: IoU для полигонов, близость центроидов для точек.

    Точка против полигона сравнивается центроидами — так же, как в метрике.
    """
    if p.geom_type == GEOM_POLYGON and g.geom_type == GEOM_POLYGON:
        return polygon_iou(p.rings, g.rings)
    d = float(np.hypot(*(p.centroid - g.centroid)))
    return max(0.0, 1.0 - d / r_tol) if d <= r_tol else 0.0


def analyze_errors(
    preds: Sequence[EvalObject],
    gts: Sequence[EvalObject],
    result: MatchResult,
    cfg: EvalConfig | None = None,
) -> ErrorReport:
    """Классифицирует FP и FN из результата match(preds, gts, cfg)."""
    cfg = cfg or EvalConfig()
    matched_p = {pi for pi, _, _ in result.pairs}
    matched_g = {gi for _, gi, _ in result.pairs}
    r_tol = _r_tol_for(gts, cfg)

    tp_preds = [preds[pi] for pi, _, _ in result.pairs]
    fp_types: dict[str, list[int]] = {k: [] for k in FP_TYPES}
    for i, p in enumerate(preds):
        if i in matched_p:
            continue
        kind = ""
        for tp in tp_preds:  # дубль: повторяет TP-предсказание своего класса
            if tp.cls == p.cls and polygon_iou(p.rings, tp.rings) >= cfg.iou_thr:
                kind = "duplicate"
                break
        if not kind:  # перекрытие с GT: свой класс важнее чужого
            for j, g in enumerate(gts):
                if _overlap(p, g, r_tol[j]) <= 0.0:
                    continue
                if g.cls == p.cls:
                    kind = "localization"
                    break
                kind = kind or "class_confusion"
        fp_types[kind or "background"].append(i)

    fn_types: dict[str, list[int]] = {k: [] for k in FN_TYPES}
    for i, g in enumerate(gts):
        if i in matched_g:
            continue
        covered = any(_overlap(p, g, r_tol[i]) > 0.0 for p in preds)
        fn_types["covered" if covered else "missed"].append(i)

    return ErrorReport(
        fp_types={k: tuple(v) for k, v in fp_types.items()},
        fn_types={k: tuple(v) for k, v in fn_types.items()},
    )


def _feature(kind: str, error: str, obj: EvalObject, class_key: str) -> dict[str, Any]:
    """EvalObject -> Feature GeoJSON (координаты как во входных объектах)."""
    if obj.geom_type == GEOM_POINT:
        geometry: dict[str, Any] = {"type": "Point", "coordinates": obj.rings[0][0].tolist()}
    else:
        geometry = {"type": "Polygon", "coordinates": polygon_rings_to_coords(obj.rings)}
    return {
        "type": "Feature",
        "properties": {"kind": kind, "error": error, class_key: obj.cls, "score": obj.score},
        "geometry": geometry,
    }


def errors_to_geojson(
    path: str | Path,
    preds: Sequence[EvalObject],
    gts: Sequence[EvalObject],
    report: ErrorReport,
    crs: str = "EPSG:3857",
    class_key: str = "class",
) -> int:
    """FP и FN в один GeoJSON для QGIS. Возвращает число записанных объектов.

    Координаты пишутся как есть: объекты из read_objects уже в crs файла,
    объекты из пайплайна — в crs.internal (тогда укажите его в crs).
    """
    features: list[dict[str, Any]] = []
    for kind, idxs in report.fp_types.items():
        features += [_feature("FP", kind, preds[i], class_key) for i in idxs]
    for kind, idxs in report.fn_types.items():
        features += [_feature("FN", kind, gts[i], class_key) for i in idxs]

    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    epsg = CRS.from_user_input(crs).to_epsg()
    if epsg is not None:
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return len(features)
