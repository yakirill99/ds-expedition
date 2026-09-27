"""Растровые признаки из облака точек.

DTM — это одно. Но в исходном облаке есть атрибуты, которые не
попадают в DTM: интенсивность, номер отражения, класс точки.
Эти признаки иногда дают модели больше, чем сам рельеф.

Все признаки считаются по ВСЕМ точкам, не только по земле.
Это важно: если фильтр земли что-то отбросил, признаки всё равно
увидят.

Модуль строит растры:
    point_density             — сколько точек на м²
    mean_intensity            — средняя интенсивность
    std_intensity             — разброс интенсивности
    non_first_return_ratio    — доля точек не-первых отражений
    mean_return_number        — средний номер отражения
    mean_z                    — средняя высота точек
    z_std                     — разброс высот

Все функции чистые: принимают массивы точек и Grid, возвращают Layer.
Кэширование — снаружи, через LayerCache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)

# Константы: имена признаков.
FEATURE_POINT_DENSITY = "point_density"
FEATURE_MEAN_INTENSITY = "mean_intensity"
FEATURE_STD_INTENSITY = "std_intensity"
FEATURE_NON_FIRST_RETURN_RATIO = "non_first_return_ratio"
FEATURE_MEAN_RETURN_NUMBER = "mean_return_number"
FEATURE_MEAN_Z = "mean_z"
FEATURE_Z_STD = "z_std"

SUPPORTED_FEATURES = (
    FEATURE_POINT_DENSITY,
    FEATURE_MEAN_INTENSITY,
    FEATURE_STD_INTENSITY,
    FEATURE_NON_FIRST_RETURN_RATIO,
    FEATURE_MEAN_RETURN_NUMBER,
    FEATURE_MEAN_Z,
    FEATURE_Z_STD,
)

# Константы: единицы и лицензии по умолчанию.
_DEFAULT_NODATA = -9999.0


@dataclass(frozen=True)
class CloudFeaturesResult:
    """Результат построения признаков из облака.

    layers — словарь имя -> Layer, только для запрошенных признаков.
    """

    layers: dict[str, Layer]


def _bin_indices(
    x: np.ndarray,
    y: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Считает индексы ячеек для точек и маску валидных.

    Args:
        x, y: координаты точек (метры, в CRS сетки).
        grid: целевая сетка.

    Returns:
        Кортеж (row_idx, col_idx, valid). row_idx, col_idx — int64,
        valid — булев массив той же длины, что x и y.
    """
    col, row = grid.transform_to_pixel(x=x, y=y)
    col_idx = np.floor(col).astype(np.int64)
    row_idx = np.floor(row).astype(np.int64)

    valid = (col_idx >= 0) & (col_idx < grid.width) & (row_idx >= 0) & (row_idx < grid.height)
    return row_idx, col_idx, valid


def compute_point_density(
    x: np.ndarray,
    y: np.ndarray,
    grid: Grid,
    nodata: float = _DEFAULT_NODATA,
) -> Layer:
    """Считает плотность точек на м².

    Args:
        x, y: координаты точек.
        grid: целевая сетка.
        nodata: значение для пустых ячеек.

    Returns:
        Layer с плотностью (точек на м²).
    """
    row_idx, col_idx, valid = _bin_indices(x=x, y=y, grid=grid)

    count = np.zeros(grid.shape, dtype=np.float64)
    np.add.at(count, (row_idx[valid], col_idx[valid]), 1)

    cell_area = grid.pixel_size * grid.pixel_size
    density = count / cell_area

    # Пустые ячейки -> nodata.
    empty = count == 0
    density[empty] = nodata

    logger.debug(
        "point_density: max=%.2f, mean по непустым=%.2f",
        float(density[~empty].max()) if (~empty).any() else 0.0,
        float(density[~empty].mean()) if (~empty).any() else 0.0,
    )

    return Layer(
        name=FEATURE_POINT_DENSITY,
        data=density.astype(np.float32),
        grid=grid,
        nodata=nodata,
        source="cloud",
        license="unknown",
        unit="points/m^2",
    )


def compute_mean_intensity(
    x: np.ndarray,
    y: np.ndarray,
    intensity: np.ndarray,
    grid: Grid,
    nodata: float = _DEFAULT_NODATA,
) -> Layer:
    """Считает среднюю интенсивность по ячейке.

    Args:
        x, y: координаты точек.
        intensity: интенсивность отражения.
        grid: целевая сетка.
        nodata: значение для пустых ячеек.

    Returns:
        Layer со средней интенсивностью.
    """
    row_idx, col_idx, valid = _bin_indices(x=x, y=y, grid=grid)

    count = np.zeros(grid.shape, dtype=np.int64)
    sums = np.zeros(grid.shape, dtype=np.float64)
    np.add.at(count, (row_idx[valid], col_idx[valid]), 1)
    np.add.at(sums, (row_idx[valid], col_idx[valid]), intensity[valid])

    non_empty = count > 0
    mean_intensity = np.full(grid.shape, nodata, dtype=np.float64)
    mean_intensity[non_empty] = sums[non_empty] / count[non_empty]

    return Layer(
        name=FEATURE_MEAN_INTENSITY,
        data=mean_intensity.astype(np.float32),
        grid=grid,
        nodata=nodata,
        source="cloud",
        license="unknown",
        unit="intensity",
    )


def compute_std_intensity(
    x: np.ndarray,
    y: np.ndarray,
    intensity: np.ndarray,
    grid: Grid,
    nodata: float = _DEFAULT_NODATA,
) -> Layer:
    """Считает стандартное отклонение интенсивности по ячейке.

    Ячейки с одной точкой дают std=0 (не nan).

    Args:
        x, y: координаты точек.
        intensity: интенсивность отражения.
        grid: целевая сетка.
        nodata: значение для пустых ячеек.

    Returns:
        Layer со std интенсивности.
    """
    row_idx, col_idx, valid = _bin_indices(x=x, y=y, grid=grid)

    count = np.zeros(grid.shape, dtype=np.int64)
    sums = np.zeros(grid.shape, dtype=np.float64)
    sums_sq = np.zeros(grid.shape, dtype=np.float64)

    r = row_idx[valid]
    c = col_idx[valid]
    v = intensity[valid]

    np.add.at(count, (r, c), 1)
    np.add.at(sums, (r, c), v)
    np.add.at(sums_sq, (r, c), v * v)

    non_empty = count > 0
    mean = np.zeros(grid.shape, dtype=np.float64)
    mean[non_empty] = sums[non_empty] / count[non_empty]

    variance = np.zeros(grid.shape, dtype=np.float64)
    variance[non_empty] = sums_sq[non_empty] / count[non_empty] - mean[non_empty] ** 2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    result = np.full(grid.shape, nodata, dtype=np.float64)
    result[non_empty] = std[non_empty]

    return Layer(
        name=FEATURE_STD_INTENSITY,
        data=result.astype(np.float32),
        grid=grid,
        nodata=nodata,
        source="cloud",
        license="unknown",
        unit="intensity",
    )


def compute_non_first_return_ratio(
    x: np.ndarray,
    y: np.ndarray,
    return_number: np.ndarray,
    grid: Grid,
    nodata: float = _DEFAULT_NODATA,
) -> Layer:
    """Считает долю точек, которые НЕ первые отражения.

    Первое отражение — верхушка (дерево, крыша).
    Не первое — ниже (земля, ствол).
    Высокая доля непервых = густая растительность или сложный рельеф.

    Args:
        x, y: координаты точек.
        return_number: номер отражения (1 = первое).
        grid: целевая сетка.
        nodata: значение для пустых ячеек.

    Returns:
        Layer с долей непервых отражений в [0, 1].
    """
    row_idx, col_idx, valid = _bin_indices(x=x, y=y, grid=grid)

    count = np.zeros(grid.shape, dtype=np.int64)
    non_first = np.zeros(grid.shape, dtype=np.int64)

    r = row_idx[valid]
    c = col_idx[valid]
    rn = return_number[valid]

    np.add.at(count, (r, c), 1)
    is_non_first = (rn > 1).astype(np.int64)
    np.add.at(non_first, (r, c), is_non_first)

    non_empty = count > 0
    ratio = np.full(grid.shape, nodata, dtype=np.float64)
    ratio[non_empty] = non_first[non_empty] / count[non_empty]

    return Layer(
        name=FEATURE_NON_FIRST_RETURN_RATIO,
        data=ratio.astype(np.float32),
        grid=grid,
        nodata=nodata,
        source="cloud",
        license="unknown",
        unit="ratio",
    )


def compute_mean_return_number(
    x: np.ndarray,
    y: np.ndarray,
    return_number: np.ndarray,
    grid: Grid,
    nodata: float = _DEFAULT_NODATA,
) -> Layer:
    """Считает средний номер отражения по ячейке.

    Args:
        x, y: координаты точек.
        return_number: номер отражения.
        grid: целевая сетка.
        nodata: значение для пустых ячеек.

    Returns:
        Layer со средним номером отражения.
    """
    row_idx, col_idx, valid = _bin_indices(x=x, y=y, grid=grid)

    count = np.zeros(grid.shape, dtype=np.int64)
    sums = np.zeros(grid.shape, dtype=np.float64)
    r = row_idx[valid]
    c = col_idx[valid]
    rn = return_number[valid].astype(np.float64)

    np.add.at(count, (r, c), 1)
    np.add.at(sums, (r, c), rn)

    non_empty = count > 0
    mean_rn = np.full(grid.shape, nodata, dtype=np.float64)
    mean_rn[non_empty] = sums[non_empty] / count[non_empty]

    return Layer(
        name=FEATURE_MEAN_RETURN_NUMBER,
        data=mean_rn.astype(np.float32),
        grid=grid,
        nodata=nodata,
        source="cloud",
        license="unknown",
        unit="return_number",
    )


def compute_mean_z(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: Grid,
    nodata: float = _DEFAULT_NODATA,
) -> Layer:
    """Считает среднюю высоту точек по ячейке.

    Отличается от DTM: там минимум или mean по земле. Здесь — mean
    по ВСЕМ точкам. Если mean_z сильно выше DTM — в ячейке есть
    растительность или объекты.

    Args:
        x, y, z: координаты точек.
        grid: целевая сетка.
        nodata: значение для пустых ячеек.

    Returns:
        Layer со средней высотой.
    """
    row_idx, col_idx, valid = _bin_indices(x=x, y=y, grid=grid)

    count = np.zeros(grid.shape, dtype=np.int64)
    sums = np.zeros(grid.shape, dtype=np.float64)
    r = row_idx[valid]
    c = col_idx[valid]
    zv = z[valid]

    np.add.at(count, (r, c), 1)
    np.add.at(sums, (r, c), zv)

    non_empty = count > 0
    mean_z = np.full(grid.shape, nodata, dtype=np.float64)
    mean_z[non_empty] = sums[non_empty] / count[non_empty]

    return Layer(
        name=FEATURE_MEAN_Z,
        data=mean_z.astype(np.float32),
        grid=grid,
        nodata=nodata,
        source="cloud",
        license="unknown",
        unit="meters",
    )


def compute_z_std(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: Grid,
    nodata: float = _DEFAULT_NODATA,
) -> Layer:
    """Считает разброс высот по ячейке.

    Высокий std = неоднородная вертикальная структура (растительность,
    здания, сложный рельеф).

    Args:
        x, y, z: координаты точек.
        grid: целевая сетка.
        nodata: значение для пустых ячеек.

    Returns:
        Layer со std высот.
    """
    row_idx, col_idx, valid = _bin_indices(x=x, y=y, grid=grid)

    count = np.zeros(grid.shape, dtype=np.int64)
    sums = np.zeros(grid.shape, dtype=np.float64)
    sums_sq = np.zeros(grid.shape, dtype=np.float64)

    r = row_idx[valid]
    c = col_idx[valid]
    zv = z[valid]

    np.add.at(count, (r, c), 1)
    np.add.at(sums, (r, c), zv)
    np.add.at(sums_sq, (r, c), zv * zv)

    non_empty = count > 0
    mean = np.zeros(grid.shape, dtype=np.float64)
    mean[non_empty] = sums[non_empty] / count[non_empty]

    variance = np.zeros(grid.shape, dtype=np.float64)
    variance[non_empty] = sums_sq[non_empty] / count[non_empty] - mean[non_empty] ** 2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    result = np.full(grid.shape, nodata, dtype=np.float64)
    result[non_empty] = std[non_empty]

    return Layer(
        name=FEATURE_Z_STD,
        data=result.astype(np.float32),
        grid=grid,
        nodata=nodata,
        source="cloud",
        license="unknown",
        unit="meters",
    )


def build_cloud_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    intensity: np.ndarray,
    return_number: np.ndarray,
    grid: Grid,
    enabled: tuple[str, ...],
    nodata: float = _DEFAULT_NODATA,
) -> CloudFeaturesResult:
    """Строит все запрошенные растровые признаки из облака.

    Args:
        x, y, z: координаты точек.
        intensity: интенсивность отражения.
        return_number: номер отражения (1 = первое).
        grid: целевая сетка.
        enabled: список имён признаков (см. SUPPORTED_FEATURES).
        nodata: значение для пустых ячеек.

    Returns:
        CloudFeaturesResult со словарём Layer'ов.

    Raises:
        ValueError: если длины массивов не совпадают, или запрошен
            неизвестный признак.
    """
    n = len(x)
    if not (len(y) == len(z) == len(intensity) == len(return_number) == n):
        raise ValueError(
            f"Длины массивов должны совпадать: "
            f"x={len(x)}, y={len(y)}, z={len(z)}, "
            f"intensity={len(intensity)}, return_number={len(return_number)}"
        )
    if n == 0:
        raise ValueError("Пустое облако точек")

    unknown = set(enabled) - set(SUPPORTED_FEATURES)
    if unknown:
        raise ValueError(f"Неизвестные признаки: {sorted(unknown)}. Доступны: {SUPPORTED_FEATURES}")

    layers: dict[str, Layer] = {}

    for name in enabled:
        if name == FEATURE_POINT_DENSITY:
            layers[name] = compute_point_density(x=x, y=y, grid=grid, nodata=nodata)
        elif name == FEATURE_MEAN_INTENSITY:
            layers[name] = compute_mean_intensity(
                x=x, y=y, intensity=intensity, grid=grid, nodata=nodata
            )
        elif name == FEATURE_STD_INTENSITY:
            layers[name] = compute_std_intensity(
                x=x, y=y, intensity=intensity, grid=grid, nodata=nodata
            )
        elif name == FEATURE_NON_FIRST_RETURN_RATIO:
            layers[name] = compute_non_first_return_ratio(
                x=x, y=y, return_number=return_number, grid=grid, nodata=nodata
            )
        elif name == FEATURE_MEAN_RETURN_NUMBER:
            layers[name] = compute_mean_return_number(
                x=x, y=y, return_number=return_number, grid=grid, nodata=nodata
            )
        elif name == FEATURE_MEAN_Z:
            layers[name] = compute_mean_z(x=x, y=y, z=z, grid=grid, nodata=nodata)
        elif name == FEATURE_Z_STD:
            layers[name] = compute_z_std(x=x, y=y, z=z, grid=grid, nodata=nodata)
        else:
            raise ValueError(f"Признак '{name}' не реализован")
        logger.info("Построен признак: %s", name)

    return CloudFeaturesResult(layers=layers)
