"""Метрика детекции: сопоставление предсказаний с разметкой, TP/FP/FN, F1.

Всё считается в одной CRS (по регламенту EPSG:3857), координаты как есть.
- GT Polygon <-> pred Polygon: IoU >= iou_thr.
- GT Point <-> pred (Polygon или Point): расстояние от центроида pred до точки <= r_tol.
Жадно по убыванию score: pred берёт лучший свободный GT своего класса. Один GT — один TP.
Качество кандидата: IoU для полигонов, 1 - 0.5 * d / r для точек (в [0.5, 1]).
r_tol_units: "crs" — r_tol в единицах CRS как есть; "meters" — метры на местности,
в EPSG:3857 переводятся множителем cosh(y / R) = 1 / cos(широта).
IoU: площади точно (шнурование), пересечение — выборкой even-odd на сетке iou_samples.
0/0 в precision / recall / F1 считается 1.0 (нечего искать и нечего лишнего).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from expds.features.grid import Grid
from expds.io.geojson import (
    GEOM_POINT,
    GEOM_POLYGON,
    Detection,
    LabelFeature,
    read_labels,
    ring_centroid,
)

WEB_MERCATOR_R = 6378137.0
R_TOL_UNITS = ("crs", "meters")
ALL = "__all__"


@dataclass(frozen=True)
class EvalConfig:
    """Параметры метрики (секция `eval` YAML)."""

    iou_thr: float = 0.5
    r_tol: float = 5.0
    r_tol_units: str = "crs"
    iou_samples: int = 128
    class_agnostic: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> EvalConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"eval: unknown keys {sorted(unknown)}")
        cfg = cls(**d)
        if not 0.0 < cfg.iou_thr <= 1.0:
            raise ValueError("eval.iou_thr must be in (0, 1]")
        if cfg.r_tol <= 0:
            raise ValueError("eval.r_tol must be > 0")
        if cfg.r_tol_units not in R_TOL_UNITS:
            raise ValueError(f"eval.r_tol_units must be one of {R_TOL_UNITS}")
        if cfg.iou_samples < 8:
            raise ValueError("eval.iou_samples must be >= 8")
        return cfg


@dataclass(frozen=True, eq=False)
class EvalObject:
    """Объект для метрики: Point — одно кольцо (1, 2); Polygon — [0] внешнее, остальные дыры."""

    cls: str
    geom_type: str
    rings: tuple[np.ndarray, ...]
    score: float = 1.0
    centroid: np.ndarray = field(init=False)
    bbox: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.geom_type not in (GEOM_POINT, GEOM_POLYGON):
            raise ValueError(f"EvalObject: неизвестный тип {self.geom_type}")
        rings = tuple(np.asarray(r, dtype=np.float64) for r in self.rings)
        object.__setattr__(self, "rings", rings)
        c = rings[0][0] if self.geom_type == GEOM_POINT else ring_centroid(rings[0])
        object.__setattr__(self, "centroid", np.asarray(c, dtype=np.float64))
        outer = rings[0]
        object.__setattr__(self, "bbox", np.concatenate([outer.min(0), outer.max(0)]))


def objects_from_detections(dets: Iterable[Detection]) -> list[EvalObject]:
    return [EvalObject(d.cls, d.geom_type, (d.geometry,), float(d.score)) for d in dets]


def objects_from_labels(
    feats: Iterable[LabelFeature], score_key: str = "score"
) -> list[EvalObject]:
    return [
        EvalObject(f.cls, f.geom_type, f.world, float(f.properties.get(score_key, 1.0)))
        for f in feats
    ]


def read_objects(
    path: str | Path, crs: str = "EPSG:3857", class_key: str = "class"
) -> list[EvalObject]:
    """GeoJSON -> EvalObject в координатах файла (без перепроекции)."""
    identity = Grid(crs=crs, pixel_size=1.0, x_min=0.0, y_min=0.0, width=1, height=1)
    return objects_from_labels(read_labels(Path(path), identity, crs, class_key=class_key))


# --- геометрия -----------------------------------------------------------------


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0] - ring[0, 0], ring[:, 1] - ring[0, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _poly_area(rings: Sequence[np.ndarray]) -> float:
    return max(_ring_area(rings[0]) - sum(_ring_area(r) for r in rings[1:]), 0.0)


def _inside(px: np.ndarray, py: np.ndarray, rings: Sequence[np.ndarray]) -> np.ndarray:
    """Even-odd по всем кольцам (дыры учитываются автоматически)."""
    inside = np.zeros(px.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for ring in rings:
            x0, y0 = ring[:, 0], ring[:, 1]
            x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
            for a, b, c, d in zip(x0, y0, x1, y1, strict=True):
                if b == d:
                    continue
                cross = (b > py) != (d > py)
                xint = a + (py - b) * (c - a) / (d - b)
                inside ^= cross & (px < xint)
    return inside


def polygon_iou(a: Sequence[np.ndarray], b: Sequence[np.ndarray], samples: int = 128) -> float:
    """IoU двух полигонов (кольца: [0] внешнее, дальше дыры), без замыкающих точек."""
    lo = np.maximum(a[0].min(0), b[0].min(0))
    hi = np.minimum(a[0].max(0), b[0].max(0))
    if np.any(hi <= lo):
        return 0.0
    area_a, area_b = _poly_area(a), _poly_area(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    origin = lo
    a = [r - origin for r in a]
    b = [r - origin for r in b]
    size = hi - lo
    step = float(size.max()) / samples
    nx, ny = max(int(np.ceil(size[0] / step)), 1), max(int(np.ceil(size[1] / step)), 1)
    sx, sy = size[0] / nx, size[1] / ny
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * sx, (np.arange(ny) + 0.5) * sy)
    px, py = gx.ravel(), gy.ravel()
    inter = float(np.count_nonzero(_inside(px, py, a) & _inside(px, py, b))) * sx * sy
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- сопоставление -------------------------------------------------------------


def _safe_ratio(num: int, den: int) -> float:
    return num / den if den > 0 else 1.0


@dataclass
class MatchResult:
    """Счётчики и пары (pred_idx, gt_idx, quality)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    pairs: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return _safe_ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe_ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _safe_ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def to_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _r_tol_crs(gts: Sequence[EvalObject], cfg: EvalConfig) -> np.ndarray:
    """Радиус допуска для каждого GT в единицах CRS."""
    r = np.full(len(gts), cfg.r_tol, dtype=np.float64)
    if cfg.r_tol_units == "meters" and gts:
        y = np.array([g.centroid[1] for g in gts])
        r *= np.cosh(y / WEB_MERCATOR_R)
    return r


def match(
    preds: Sequence[EvalObject], gts: Sequence[EvalObject], cfg: EvalConfig | None = None
) -> MatchResult:
    """Жадное сопоставление по убыванию score. См. docstring модуля."""
    cfg = cfg or EvalConfig()
    res = MatchResult()
    if not gts:
        res.fp = len(preds)
        return res
    r_tol = _r_tol_crs(gts, cfg)
    is_point = np.array([g.geom_type == GEOM_POINT for g in gts])
    gbox = np.array([g.bbox for g in gts])
    gbox[is_point, :2] -= r_tol[is_point, None]
    gbox[is_point, 2:] += r_tol[is_point, None]
    gcls = np.array([g.cls for g in gts])
    used = np.zeros(len(gts), dtype=bool)

    for pi in sorted(range(len(preds)), key=lambda i: -preds[i].score):
        p = preds[pi]
        pb = p.bbox
        cand = (
            ~used
            & (gbox[:, 0] <= pb[2])
            & (gbox[:, 2] >= pb[0])
            & (gbox[:, 1] <= pb[3])
            & (gbox[:, 3] >= pb[1])
        )
        if not cfg.class_agnostic:
            cand &= gcls == p.cls
        best_q, best_g = 0.0, -1
        for gi in np.flatnonzero(cand):
            g = gts[gi]
            if is_point[gi]:
                d = float(np.hypot(*(p.centroid - g.centroid)))
                q = 1.0 - 0.5 * d / r_tol[gi] if d <= r_tol[gi] else 0.0
            elif p.geom_type == GEOM_POLYGON:
                iou = polygon_iou(p.rings, g.rings, cfg.iou_samples)
                q = iou if iou >= cfg.iou_thr else 0.0
            else:
                q = 0.0
            if q > best_q:
                best_q, best_g = q, int(gi)
        if best_g >= 0:
            used[best_g] = True
            res.tp += 1
            res.pairs.append((pi, best_g, best_q))
        else:
            res.fp += 1
    res.fn = int((~used).sum())
    return res


def evaluate(
    preds: Sequence[EvalObject], gts: Sequence[EvalObject], cfg: EvalConfig | None = None
) -> dict[str, MatchResult]:
    """{"__all__": итог, <класс>: по классу}. Классы — объединение классов pred и GT."""
    cfg = cfg or EvalConfig()
    total = match(preds, gts, cfg)
    matched_p = {pi for pi, _, _ in total.pairs}
    matched_g = {gi for _, gi, _ in total.pairs}
    out: dict[str, MatchResult] = {ALL: total}
    for cls in sorted({o.cls for o in [*preds, *gts]}):
        r = MatchResult()
        r.tp = sum(1 for _, gi, _ in total.pairs if gts[gi].cls == cls)
        r.fp = sum(1 for i, p in enumerate(preds) if p.cls == cls and i not in matched_p)
        r.fn = sum(1 for i, g in enumerate(gts) if g.cls == cls and i not in matched_g)
        out[cls] = r
    return out


def evaluate_files(
    pred_path: str | Path,
    gt_path: str | Path,
    cfg: EvalConfig | None = None,
    crs: str = "EPSG:3857",
    class_key: str = "class",
) -> dict[str, MatchResult]:
    """Оба GeoJSON в одной CRS (по регламенту 3857)."""
    return evaluate(
        read_objects(pred_path, crs, class_key), read_objects(gt_path, crs, class_key), cfg
    )
