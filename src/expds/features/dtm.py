"""Построение DTM из облака точек земли.

Шаги:
1. Биннинг точек в Grid: для каждой ячейки — статистика (mean/min/median)
   и количество точек.
2. Заполнение дыр: там, где точек нет, интерполяция (IDW или nearest).
   Дыры больше max_hole_pixels не заполняются — остаются nodata.
3. Детектор артефактов: полосы от траекторий, резкие скачки.

Возвращает DtmResult с тремя слоями:
- dtm: собственно высоты;
- hole_mask: 1 там, где данных не было (для модели);
- count: сколько точек попало в ячейку.

Все параметры — из DtmConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from expds.features.config import DtmConfig
from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)

# Константы: имена статистик и методов заполнения.
_STAT_MEAN = "mean"
_STAT_MIN = "min"
_STAT_MEDIAN = "median"
_SUPPORTED_STATS = (_STAT_MEAN, _STAT_MIN, _STAT_MEDIAN)

_FILL_IDW = "idw"
_FILL_NEAREST = "nearest"
_SUPPORTED_FILLS = (_FILL_IDW, _FILL_NEAREST)

# Константа: nodata для DTM (в метрах, ниже любого реального рельефа).
_DTM_NODATA = -9999.0

# Константа: минимальное число точек для статистики.
_MIN_POINTS_FOR_MEDIAN = 1


@dataclass(frozen=True)
class DtmResult:
    """Результат построения DTM.

    dtm — высоты (float32), hole_mask — 1.0 там, где не было точек,
    count — количество точек на ячейку.
    """

    dtm: Layer
    hole_mask: Layer
    count: Layer


def _bin_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: Grid,
    statistic: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Биннит точки в сетку, считает статистику и count.

    Args:
        x, y, z: координаты точек (метры).
        grid: целевая сетка.
        statistic: "mean", "min" или "median".

    Returns:
        Кортеж (values, count). values — float32, count — int32.
        Пустые ячейки в values помечены nan.

    Raises:
        ValueError: если statistic не поддерживается.
    """
    if statistic not in _SUPPORTED_STATS:
        raise ValueError(
            f"Статистика '{statistic}' не поддерживается. Доступны: {_SUPPORTED_STATS}"
        )

    col, row = grid.transform_to_pixel(x=x, y=y)
    col_idx = np.floor(col).astype(np.int64)
    row_idx = np.floor(row).astype(np.int64)

    valid = (col_idx >= 0) & (col_idx < grid.width) & (row_idx >= 0) & (row_idx < grid.height)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        logger.warning("Отброшено точек вне сетки: %d", n_dropped)

    col_idx = col_idx[valid]
    row_idx = row_idx[valid]
    z_valid = z[valid]

    count = np.zeros(grid.shape, dtype=np.int32)
    np.add.at(count, (row_idx, col_idx), 1)

    values = np.full(grid.shape, np.nan, dtype=np.float64)
    if statistic == _STAT_MEAN:
        sums = np.zeros(grid.shape, dtype=np.float64)
        np.add.at(sums, (row_idx, col_idx), z_valid)
        non_empty = count > 0
        values[non_empty] = sums[non_empty] / count[non_empty]
    elif statistic == _STAT_MIN:
        values = np.full(grid.shape, np.inf, dtype=np.float64)
        np.minimum.at(values, (row_idx, col_idx), z_valid)
        values[~np.isfinite(values)] = np.nan
    elif statistic == _STAT_MEDIAN:
        # Медиана через сортировку по ячейкам. Для больших облаков медленно,
        # но даёт устойчивость. Используем группировку по индексам.
        flat_idx = row_idx * grid.width + col_idx
        order = np.argsort(flat_idx, kind="stable")
        sorted_idx = flat_idx[order]
        sorted_z = z_valid[order]
        unique, starts = np.unique(sorted_idx, return_index=True)
        ends = np.append(starts[1:], len(sorted_idx))
        flat_values = np.full(grid.width * grid.height, np.nan, dtype=np.float64)
        for u, s, e in zip(unique, starts, ends, strict=True):
            flat_values[u] = np.median(sorted_z[s:e])
        values = flat_values.reshape(grid.shape)

    return values.astype(np.float32), count


def _fill_nearest(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Заполняет дыры значением ближайшей валидной ячейки.

    Args:
        values: массив с nan в дырах.
        mask: булева маска дыр (True = дыра).

    Returns:
        Заполненный массив float32.
    """
    if not mask.any():
        return values.astype(np.float32)
    indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = values[tuple(indices)]
    return filled.astype(np.float32)


def _fill_idw(
    values: np.ndarray,
    mask: np.ndarray,
    grid: Grid,
    k: int,
    power: float,
) -> np.ndarray:
    """Заполняет дыры методом IDW.

    Для каждой дыры берём k ближайших валидных ячеек, считаем
    взвешенное среднее с весом 1 / d^power.

    Args:
        values: массив с nan в дырах.
        mask: булева маска дыр (True = дыра).
        grid: сетка (для перевода индексов в мировые координаты).
        k: число соседей.
        power: степень расстояния.

    Returns:
        Заполненный массив float32.
    """
    valid = ~mask & np.isfinite(values)
    if not valid.any():
        logger.warning("IDW: нет валидных ячеек для заполнения")
        return values.astype(np.float32)

    valid_rows, valid_cols = np.where(valid)
    hole_rows, hole_cols = np.where(mask)

    valid_x, valid_y = grid.transform_to_world(col=valid_cols, row=valid_rows)
    hole_x, hole_y = grid.transform_to_world(col=hole_cols, row=hole_rows)

    valid_points = np.column_stack([valid_x, valid_y])
    valid_values = values[valid_rows, valid_cols]

    tree = cKDTree(valid_points)
    k_actual = min(k, len(valid_points))

    distances, indices = tree.query(np.column_stack([hole_x, hole_y]), k=k_actual)
    # Если k=1, distances и indices — 1D, приводим к 2D.
    if k_actual == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Защита от деления на ноль.
    distances = np.maximum(distances, 1e-6)
    weights = 1.0 / (distances**power)
    weighted_values = valid_values[indices] * weights
    filled_values = weighted_values.sum(axis=1) / weights.sum(axis=1)

    result = values.copy()
    result[hole_rows, hole_cols] = filled_values
    return result.astype(np.float32)


def _filter_large_holes(
    mask: np.ndarray,
    max_hole_pixels: int,
) -> np.ndarray:
    """Оставляет в маске только большие дыры (для пропуска заполнения).

    Args:
        mask: булева маска дыр.
        max_hole_pixels: порог размера дыры в пикселях.

    Returns:
        Булева маска больших дыр (True = слишком большая, не заполняем).
    """
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = ndimage.sum(mask, labels, index=np.arange(1, n_labels + 1))
    big_labels = np.where(sizes > max_hole_pixels)[0] + 1
    big_mask = np.isin(labels, big_labels)
    logger.info(
        "Больших дыр: %d из %d (порог %d пикселей)",
        len(big_labels),
        n_labels,
        max_hole_pixels,
    )
    return big_mask


def detect_artifacts(
    dtm: np.ndarray,
    window: int,
    threshold: float,
) -> np.ndarray:
    """Детектор артефактов: резкие скачки относительно медианы окна.

    Args:
        dtm: 2D массив высот.
        window: размер окна медианного фильтра (пиксели).
        threshold: порог разницы в метрах.

    Returns:
        Булева маска артефактов.
    """
    median = ndimage.median_filter(dtm, size=window, mode="nearest")
    diff = np.abs(dtm - median)
    artifacts = diff > threshold
    logger.info(
        "Артефактов найдено: %d (%.3f%% площади)",
        int(artifacts.sum()),
        100.0 * artifacts.mean(),
    )
    return artifacts


def build_dtm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: Grid,
    config: DtmConfig,
    source: str = "unknown",
) -> DtmResult:
    """Строит DTM из точек земли.

    Args:
        x, y, z: координаты точек земли (метры, в CRS сетки).
        grid: целевая сетка.
        config: параметры DTM.
        source: описание источника (для метаданных Layer).

    Returns:
        DtmResult с тремя слоями.

    Raises:
        ValueError: если массивы разной длины, пусты, или метод
            заполнения / статистика не поддерживаются.
    """
    if not (len(x) == len(y) == len(z)):
        raise ValueError(f"Длины x, y, z должны совпадать: {len(x)}, {len(y)}, {len(z)}")
    if len(x) == 0:
        raise ValueError("Пустой набор точек для DTM")
    if config.statistic not in _SUPPORTED_STATS:
        raise ValueError(
            f"Статистика '{config.statistic}' не поддерживается. Доступны: {_SUPPORTED_STATS}"
        )
    if config.fill_method not in _SUPPORTED_FILLS:
        raise ValueError(
            f"Метод заполнения '{config.fill_method}' не поддерживается. "
            f"Доступны: {_SUPPORTED_FILLS}"
        )

    logger.info(
        "Построение DTM: точек=%d, grid=%dx%d, statistic=%s, fill=%s",
        len(x),
        grid.width,
        grid.height,
        config.statistic,
        config.fill_method,
    )

    # 1. Биннинг.
    values, count = _bin_points(x=x, y=y, z=z, grid=grid, statistic=config.statistic)
    hole_mask = ~np.isfinite(values)
    n_holes = int(hole_mask.sum())
    logger.info(
        "Биннинг завершён: пустых ячеек %d (%.2f%%)",
        n_holes,
        100.0 * n_holes / values.size,
    )

    # 2. Большие дыры не заполняем.
    big_holes = _filter_large_holes(mask=hole_mask, max_hole_pixels=config.max_hole_pixels)
    fillable = hole_mask & ~big_holes
    logger.info(
        "К заполнению: %d ячеек, пропущено (большие дыры): %d",
        int(fillable.sum()),
        int(big_holes.sum()),
    )

    # 3. Заполнение.
    if fillable.any():
        if config.fill_method == _FILL_IDW:
            filled = _fill_idw(
                values=values,
                mask=fillable,
                grid=grid,
                k=config.idw_k,
                power=config.idw_power,
            )
        else:  # nearest
            filled = _fill_nearest(values=values, mask=fillable)
    else:
        filled = values.copy()

    # 4. Финальный nodata: большие дыры + то, что не заполнилось.
    final_nodata_mask = ~np.isfinite(filled)
    filled[final_nodata_mask] = _DTM_NODATA

    # 5. Детектор артефактов (только для отчёта, в Layer не кладём).
    valid_for_artifacts = filled != _DTM_NODATA
    if valid_for_artifacts.any():
        artifacts = detect_artifacts(
            dtm=filled,
            window=config.artifact_window,
            threshold=config.artifact_threshold,
        )
        logger.info("Артефактов: %d ячеек", int(artifacts.sum()))

    # 6. Собираем Layer'ы.
    dtm_layer = Layer(
        name="dtm",
        data=filled.astype(np.float32),
        grid=grid,
        nodata=_DTM_NODATA,
        source=source,
        license="unknown",
        unit="meters",
    )
    hole_layer = Layer(
        name="hole_mask",
        data=hole_mask.astype(np.float32),
        grid=grid,
        nodata=-1.0,
        source=source,
        license="unknown",
        unit="bool",
    )
    count_layer = Layer(
        name="count",
        data=count.astype(np.float32),
        grid=grid,
        nodata=-1.0,
        source=source,
        license="unknown",
        unit="points",
    )

    logger.info(
        "DTM готов: Z в диапазоне [%.2f, %.2f], nodata-ячеек %d",
        float(filled[~final_nodata_mask].min()) if (~final_nodata_mask).any() else 0.0,
        float(filled[~final_nodata_mask].max()) if (~final_nodata_mask).any() else 0.0,
        int(final_nodata_mask.sum()),
    )

    return DtmResult(dtm=dtm_layer, hole_mask=hole_layer, count=count_layer)
