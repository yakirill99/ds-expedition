"""Геометрический NMS по расстоянию между центроидами детекций."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

from expds.io.geojson import Detection


def nms_centroids(
    dets: Sequence[Detection], radius: float, per_class: bool = True
) -> list[Detection]:
    """Жадный NMS: по убыванию score оставляем детекцию и гасим всё в радиусе от её центроида.

    Args:
        dets: детекции в crs.internal.
        radius: радиус подавления в единицах crs.internal (метры).
        per_class: подавлять только внутри класса.

    Returns:
        Оставшиеся детекции, отсортированные по убыванию score.
    """
    if radius <= 0 or len(dets) < 2:
        return sorted(dets, key=lambda d: -d.score)
    groups: dict[str, list[Detection]] = {}
    for d in dets:
        groups.setdefault(d.cls if per_class else "", []).append(d)
    kept: list[Detection] = []
    for group in groups.values():
        order = sorted(group, key=lambda d: -d.score)
        pts = np.array([np.asarray(d.centroid, dtype=np.float64) for d in order])
        tree = cKDTree(pts)
        suppressed = np.zeros(len(order), dtype=bool)
        for i, d in enumerate(order):
            if suppressed[i]:
                continue
            kept.append(d)
            suppressed[tree.query_ball_point(pts[i], r=radius)] = True
    return sorted(kept, key=lambda d: -d.score)
