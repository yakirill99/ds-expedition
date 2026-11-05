"""Геометрический NMS по расстоянию между центроидами детекций."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

from expds.io.geojson import Detection


def _centroids_batched(dets):
    """Та же формула центроида, пакетами одинаковой длины колец."""
    from expds.io.geojson import _AREA_EPS

    result = np.empty((len(dets), 2), dtype=np.float64)
    buckets = {}

    for i, d in enumerate(dets):
        vertices = len(d.geometry)
        if d.geom_type != "Polygon" or not 3 <= vertices <= 128:
            result[i] = d.centroid
        else:
            buckets.setdefault(vertices, []).append(i)

    for indices in buckets.values():
        for start in range(0, len(indices), 2048):
            ids = np.asarray(indices[start : start + 2048], dtype=np.intp)
            rings = np.stack([np.asarray(dets[int(i)].geometry, dtype=np.float64) for i in ids])

            origin = rings[:, 0, :]
            x = rings[:, :, 0] - origin[:, 0, None]
            y = rings[:, :, 1] - origin[:, 1, None]
            xn = np.roll(x, -1, axis=1)
            yn = np.roll(y, -1, axis=1)
            cross = x * yn - xn * y
            area = cross.sum(axis=1) / 2.0

            centers = np.empty((len(ids), 2), dtype=np.float64)
            regular = ~(np.abs(area) < _AREA_EPS)
            centers[regular, 0] = ((x[regular] + xn[regular]) * cross[regular]).sum(axis=1) / (
                6.0 * area[regular]
            ) + origin[regular, 0]
            centers[regular, 1] = ((y[regular] + yn[regular]) * cross[regular]).sum(axis=1) / (
                6.0 * area[regular]
            ) + origin[regular, 1]
            if (~regular).any():
                centers[~regular] = rings[~regular].mean(axis=1)
            result[ids] = centers

    return result


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
        pts = _centroids_batched(order)
        tree = cKDTree(pts)
        suppressed = np.zeros(len(order), dtype=bool)
        for i, d in enumerate(order):
            if suppressed[i]:
                continue
            kept.append(d)
            suppressed[tree.query_ball_point(pts[i], r=radius)] = True
    return sorted(kept, key=lambda d: -d.score)
