"""Подбор параметров постобработки по val: кэш probs на диск, перебор без инференса.

Инференс дорогой, перебор — нет: вероятности участков один раз пишутся в .npz
(save_probs_cache), дальше sweep_postparams гоняет только postprocess + метрику.
Кэш связан с чекпойнтом тегом weights_tag (sha256): при смене весов скрипт
пересчитывает его автоматически.

Перебираются seg_thr, hm_thr, min_area_px, nms_radius_px; остальные параметры
PostConfig берутся из базового конфига. Счётчики суммируются по участкам —
как в train._score, поэтому итог совместим с val f1 из лога обучения.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from expds.eval.metric import (
    ALL,
    EvalConfig,
    EvalObject,
    MatchResult,
    evaluate,
    objects_from_detections,
)
from expds.features.grid import Grid
from expds.post.vectorize import PostConfig, postprocess

#: Ключи перебора — ровно те, что задают PostGrid.
SWEEP_KEYS = ("seg_thr", "hm_thr", "min_area_px", "nms_radius_px")


@dataclass(frozen=True)
class ProbsCache:
    """Вероятности участка: probs (2K, H, W), его Grid и маска валидности."""

    site: str
    probs: np.ndarray
    grid: Grid
    valid: np.ndarray | None = None
    weights_tag: str = ""


def save_probs_cache(
    path: str | Path,
    site: str,
    probs: np.ndarray,
    grid: Grid,
    valid: np.ndarray | None = None,
    weights_tag: str = "",
) -> None:
    """Пишет кэш участка в .npz (сжатие; float32)."""
    payload: dict[str, Any] = {
        "site": site,
        "weights_tag": weights_tag,
        "crs": grid.crs,
        "pixel_size": grid.pixel_size,
        "x_min": grid.x_min,
        "y_min": grid.y_min,
        "probs": probs.astype(np.float32, copy=False),
        "valid": valid.astype(np.uint8, copy=False) if valid is not None else np.zeros(0, np.uint8),
    }
    np.savez_compressed(path, **payload)


def load_probs_cache(path: str | Path) -> ProbsCache:
    """Читает кэш участка (Grid восстанавливается из формы probs)."""
    with np.load(path, allow_pickle=False) as z:
        valid = z["valid"]
        grid = Grid(
            crs=str(z["crs"]),
            pixel_size=float(z["pixel_size"]),
            x_min=float(z["x_min"]),
            y_min=float(z["y_min"]),
            width=int(z["probs"].shape[2]),
            height=int(z["probs"].shape[1]),
        )
        return ProbsCache(
            site=str(z["site"]),
            probs=z["probs"].astype(np.float32, copy=False),
            grid=grid,
            valid=valid.astype(bool, copy=False) if valid.size else None,
            weights_tag=str(z["weights_tag"]),
        )


@dataclass(frozen=True)
class PostGrid:
    """Сетка перебора PostConfig (по значению на параметр)."""

    seg_thr: tuple[float, ...] = (0.35, 0.5, 0.65)
    hm_thr: tuple[float, ...] = (0.25, 0.4)
    min_area_px: tuple[float, ...] = (16.0, 48.0)
    nms_radius_px: tuple[float, ...] = (6.0, 10.0)

    def combos(self) -> list[dict[str, float]]:
        """Декартово произведение сетки -> список словарей для replace(PostConfig)."""
        values = (self.seg_thr, self.hm_thr, self.min_area_px, self.nms_radius_px)
        return [dict(zip(SWEEP_KEYS, v, strict=True)) for v in itertools.product(*values)]


@dataclass(frozen=True)
class SweepRow:
    """Итог одной комбинации (счётчики суммированы по всем участкам)."""

    params: dict[str, float]
    tp: int
    fp: int
    fn: int
    f1: float
    precision: float
    recall: float
    per_class_f1: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls, params: Mapping[str, float], per_class: Mapping[str, MatchResult]
    ) -> SweepRow:
        """Собирает строку из суммарных счётчиков (ALL + по классам)."""
        total = per_class[ALL]
        return cls(
            params=dict(params),
            tp=total.tp,
            fp=total.fp,
            fn=total.fn,
            f1=total.f1,
            precision=total.precision,
            recall=total.recall,
            per_class_f1={k: r.f1 for k, r in sorted(per_class.items()) if k != ALL},
        )


def sweep_postparams(
    caches: Mapping[str, ProbsCache],
    gts: Mapping[str, Sequence[EvalObject]],
    classes: Sequence[str],
    base_post: PostConfig,
    eval_cfg: EvalConfig,
    grid: PostGrid | None = None,
) -> list[SweepRow]:
    """Перебор сетки: postprocess по кэшу -> метрика. Сортировка по F1 (убывание).

    Args:
        caches: кэш вероятностей по имени участка.
        gts: GT-объекты того же участка (в crs.internal участка).
        classes: классы модели (порядок каналов probs).
        base_post: базовый PostConfig (нескользируемые поля берутся из него).
        eval_cfg: параметры метрики.
        grid: сетка перебора (по умолчанию PostGrid()).

    Returns:
        Строки по всем комбинациям, отсортированные по убыванию F1
        (при равенстве — с большим tp, затем лексикографически по параметрам).
    """
    grid = grid or PostGrid()
    if set(caches) != set(gts):
        raise ValueError(f"участки кэша и GT не совпадают: {sorted(caches)} vs {sorted(gts)}")
    k = len(classes)
    rows: list[SweepRow] = []
    for combo in grid.combos():
        post = replace(base_post, **combo)
        per_class: dict[str, MatchResult] = {}
        for name, cache in caches.items():
            if cache.probs.shape[0] != 2 * k:
                raise ValueError(f"{name}: probs {cache.probs.shape[0]} каналов, ожидалось {2 * k}")
            dets = postprocess(cache.probs, cache.grid, classes, post, valid=cache.valid)
            res = evaluate(objects_from_detections(dets), list(gts[name]), eval_cfg)
            for key, r in res.items():
                bucket = per_class.setdefault(key, MatchResult())
                bucket.tp += r.tp
                bucket.fp += r.fp
                bucket.fn += r.fn
        rows.append(SweepRow.from_results(combo, per_class))
    rows.sort(key=lambda r: (-r.f1, -r.tp, tuple(sorted(r.params.items()))))
    return rows


def format_sweep_table(rows: Sequence[SweepRow], top: int = 10) -> str:
    """Таблица топ-комбинаций для stdout."""
    head = (
        f"{'seg_thr':>7} {'hm_thr':>7} {'min_area':>9} {'nms':>5} | "
        f"{'tp':>5} {'fp':>5} {'fn':>5} {'prec':>6} {'rec':>6} {'f1':>6}"
    )
    lines = [head, "-" * len(head)]
    for r in rows[:top]:
        p = r.params
        lines.append(
            f"{p['seg_thr']:>7.3f} {p['hm_thr']:>7.3f} {p['min_area_px']:>9.1f} "
            f"{p['nms_radius_px']:>5.1f} | {r.tp:>5} {r.fp:>5} {r.fn:>5} "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f}"
        )
    return "\n".join(lines)


def best_post_yaml(rows: Sequence[SweepRow], base_post: PostConfig) -> str:
    """YAML-сниппет секции post для лучшей комбинации (копируется в конфиг датасета)."""
    if not rows:
        raise ValueError("нет строк свипа — нечего рекомендовать")
    best = rows[0]
    fixed = (
        "simplify_px",
        "peak_window_px",
        "point_radius_px",
        "point_vertices",
        "max_detections",
    )
    post: dict[str, Any] = {name: getattr(base_post, name) for name in fixed}
    post.update(best.params)
    body = yaml.safe_dump({"post": post}, sort_keys=True, allow_unicode=True).strip()
    return f"# лучшая из {len(rows)} комбинаций: F1 {best.f1:.3f}\n{body}"


def sweep_report(rows: Sequence[SweepRow], eval_cfg: EvalConfig, grid: PostGrid) -> dict[str, Any]:
    """Полный отчёт свипа в JSON-совместимом словаре."""
    return {
        "eval_config": asdict(eval_cfg),
        "grid": {key: list(getattr(grid, key)) for key in SWEEP_KEYS},
        "n_combos": len(rows),
        "rows": [
            {
                "params": r.params,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "per_class_f1": r.per_class_f1,
            }
            for r in rows
        ],
    }
