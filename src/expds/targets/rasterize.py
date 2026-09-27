"""Разметка -> растровые таргеты на Grid.

polygons_to_mask:  Polygon -> (K, H, W) float32 {0, 1}. Пиксель внутри, если внутри
                   его ЦЕНТР (правило even-odd; дыры учитываются автоматически).
points_to_heatmap: Point -> (K, H, W) float32, гаусс с пиком 1 в точке; перекрытия — max.

Канал k = classes[k] в обоих тензорах. Классы не из списка — warning и пропуск.
Всё считается в пикселях Grid (row 0 = юг): без Affine и без флипов.
Центр пикселя (col, row) в пиксельных координатах Grid — это ровно (col, row).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np

from expds.features.grid import Grid
from expds.io.geojson import GEOM_POINT, GEOM_POLYGON, LabelFeature

logger = logging.getLogger(__name__)

_DEFAULT_TRUNCATE = 4.0


def points_in_rings(px: np.ndarray, py: np.ndarray, rings: Sequence[np.ndarray]) -> np.ndarray:
    """Even-odd тест «точка внутри» по всем кольцам (внешнее + дыры).

    Args:
        px, py: координаты точек (одинаковой формы).
        rings: кольца (N, 2) без замыкающей точки, в тех же координатах.

    Returns:
        Булев массив формы px.shape.
    """
    inside = np.zeros(px.shape, dtype=bool)
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        for j in range(n):
            x1, y1 = ring[j]
            x2, y2 = ring[(j + 1) % n]
            if y1 == y2:
                continue  # горизонтальное ребро луч не пересекает
            crosses = (y1 > py) != (y2 > py)
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (px < x_cross)
    return inside


def _class_index(classes: Sequence[str]) -> dict[str, int]:
    """Имя класса -> канал; проверка уникальности."""
    if len(set(classes)) != len(classes):
        raise ValueError(f"Дублирующиеся классы: {classes}")
    return {c: i for i, c in enumerate(classes)}


def _window(
    lo_col: float, hi_col: float, lo_row: float, hi_row: float, grid: Grid
) -> tuple[int, int, int, int] | None:
    """Окно пикселей [c0..c1] x [r0..r1], обрезанное по сетке; None — вне сетки."""
    c0 = max(int(np.floor(lo_col)), 0)
    c1 = min(int(np.ceil(hi_col)), grid.width - 1)
    r0 = max(int(np.floor(lo_row)), 0)
    r1 = min(int(np.ceil(hi_row)), grid.height - 1)
    if c0 > c1 or r0 > r1:
        return None
    return c0, c1, r0, r1


def polygons_to_mask(
    features: Iterable[LabelFeature],
    grid: Grid,
    classes: Sequence[str],
) -> np.ndarray:
    """Растеризует полигоны в маски классов.

    Args:
        features: объекты разметки (Point игнорируются).
        grid: сетка.
        classes: порядок каналов.

    Returns:
        (K, H, W) float32 со значениями {0, 1}.
    """
    index = _class_index(classes)
    mask = np.zeros((len(classes), grid.height, grid.width), dtype=np.float32)
    skipped: set[str] = set()

    for feat in features:
        if feat.geom_type != GEOM_POLYGON:
            continue
        k = index.get(feat.cls)
        if k is None:
            skipped.add(feat.cls)
            continue
        ext = feat.pixel[0]
        win = _window(ext[:, 0].min(), ext[:, 0].max(), ext[:, 1].min(), ext[:, 1].max(), grid)
        if win is None:
            continue
        c0, c1, r0, r1 = win
        cols, rows = np.meshgrid(
            np.arange(c0, c1 + 1, dtype=np.float64), np.arange(r0, r1 + 1, dtype=np.float64)
        )
        inside = points_in_rings(cols, rows, feat.pixel)
        sub = mask[k, r0 : r1 + 1, c0 : c1 + 1]
        sub[inside] = 1.0

    if skipped:
        logger.warning("Классы не из списка пропущены (маски): %s", sorted(skipped))
    return mask


def points_to_heatmap(
    features: Iterable[LabelFeature],
    grid: Grid,
    classes: Sequence[str],
    sigma_px: float,
    truncate: float = _DEFAULT_TRUNCATE,
) -> np.ndarray:
    """Строит heatmap точек: гаусс с пиком 1, перекрытия — максимум.

    Args:
        features: объекты разметки (Polygon игнорируются).
        grid: сетка.
        classes: порядок каналов.
        sigma_px: сигма гаусса, пиксели.
        truncate: радиус окна в сигмах.

    Returns:
        (K, H, W) float32 в [0, 1].
    """
    if sigma_px <= 0:
        raise ValueError(f"sigma_px должна быть > 0, получено {sigma_px}")
    index = _class_index(classes)
    heat = np.zeros((len(classes), grid.height, grid.width), dtype=np.float32)
    radius = float(np.ceil(truncate * sigma_px))
    two_s2 = 2.0 * sigma_px**2
    skipped: set[str] = set()

    for feat in features:
        if feat.geom_type != GEOM_POINT:
            continue
        k = index.get(feat.cls)
        if k is None:
            skipped.add(feat.cls)
            continue
        col, row = feat.pixel[0][0]
        win = _window(col - radius, col + radius, row - radius, row + radius, grid)
        if win is None:
            continue
        c0, c1, r0, r1 = win
        cols, rows = np.meshgrid(
            np.arange(c0, c1 + 1, dtype=np.float64), np.arange(r0, r1 + 1, dtype=np.float64)
        )
        gauss = np.exp(-((cols - col) ** 2 + (rows - row) ** 2) / two_s2).astype(np.float32)
        sub = heat[k, r0 : r1 + 1, c0 : c1 + 1]
        np.maximum(sub, gauss, out=sub)

    if skipped:
        logger.warning("Классы не из списка пропущены (heatmap): %s", sorted(skipped))
    return heat


def labels_to_targets(
    features: Sequence[LabelFeature],
    grid: Grid,
    classes: Sequence[str],
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Оба таргета сразу: (y_seg, y_hm), каждый (K, H, W) float32."""
    return (
        polygons_to_mask(features, grid, classes),
        points_to_heatmap(features, grid, classes, sigma_px),
    )
