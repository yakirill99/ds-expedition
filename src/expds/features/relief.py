"""Производные рельефа из DTM.

Итерация 1: slope, aspect, hillshade, SLRM.
Итерация 2: positive_openness, negative_openness, sky_view_factor, curvature.
Итерация 3: TPI, TRI.

Все функции принимают DTM (2D float32) и nodata. Перед вычислением
производных nodata заполняется интерполированными значениями
(_fill_nodata_for_filter), чтобы фильтры (gaussian, gradient,
uniform_filter, laplace) не размазывали -9999 в соседние пиксели.
После вычисления nodata восстанавливается там, где входной DTM был
nodata.

Функции чистые: не читают файлы, не пишут. Кэширование — снаружи,
через LayerCache.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# Константы: параметры по умолчанию.
_DEFAULT_HILLSHADE_ALTITUDE = 45.0
_DEFAULT_HILLSHADE_Z_FACTOR = 1.0
_DEFAULT_NODATA = -9999.0

# Константы: преобразование градусов в радианы.
_DEG_TO_RAD = np.pi / 180.0

# Константы: минимальное значение для atan2, чтобы избежать деления на ноль.
_EPSILON = 1e-9


# ============================================================
# Вспомогательные функции
# ============================================================


def _fill_nodata_for_filter(
    dtm: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """Заполняет nodata интерполированными значениями перед фильтрацией.

    Использует nearest-neighbour заполнение через distance_transform_edt.
    Это позволяет gaussian_filter / gradient / uniform_filter работать
    без размазывания -9999 в соседние пиксели.

    Args:
        dtm: 2D массив высот с nodata.
        nodata: значение nodata.

    Returns:
        2D массив без nodata (заполненный).
    """
    mask = dtm == nodata
    if not mask.any():
        return dtm.copy()

    indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return dtm[tuple(indices)].copy()


def _apply_nodata_mask(
    result: np.ndarray,
    dtm: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """Восстанавливает nodata в результате там, где входной DTM был nodata.

    Args:
        result: 2D массив производной.
        dtm: исходный DTM (с nodata).
        nodata: значение nodata.

    Returns:
        Копия result с nodata в соответствующих пикселях.
    """
    out = result.copy()
    out[dtm == nodata] = nodata
    return out


# ============================================================
# Итерация 1: slope, aspect, hillshade, SLRM
# ============================================================


def compute_slope(
    dtm: np.ndarray,
    pixel_size: float,
    nodata: float = _DEFAULT_NODATA,
) -> np.ndarray:
    """Считает наклон поверхности.

    Наклон — модуль градиента, в градусах (arctan от модуля).
    На плоской поверхности — 0, на вертикальной — 90.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.
        nodata: значение nodata.

    Returns:
        2D массив float32 с наклоном в градусах.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    gy, gx = np.gradient(dtm_filled, pixel_size)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope_deg = slope_rad / _DEG_TO_RAD
    return _apply_nodata_mask(slope_deg, dtm, nodata).astype(np.float32)


def compute_aspect(
    dtm: np.ndarray,
    pixel_size: float,
    nodata: float = _DEFAULT_NODATA,
) -> np.ndarray:
    """Считает экспозицию склона (направление, куда стекает вода).

    Аспект — угол от севера по часовой стрелке, в градусах [0, 360).
    Плоские участки получают значение 0.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.
        nodata: значение nodata.

    Returns:
        2D массив float32 с аспектом в градусах.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    gy, gx = np.gradient(dtm_filled, pixel_size)
    # Grid: row 0 = юг, поэтому -gy.
    aspect_rad = np.arctan2(-gx, -gy)
    aspect_deg = (aspect_rad / _DEG_TO_RAD) % 360.0
    return _apply_nodata_mask(aspect_deg, dtm, nodata).astype(np.float32)


def compute_hillshade(
    dtm: np.ndarray,
    pixel_size: float,
    azimuth_deg: float,
    altitude_deg: float = _DEFAULT_HILLSHADE_ALTITUDE,
    z_factor: float = _DEFAULT_HILLSHADE_Z_FACTOR,
    nodata: float = _DEFAULT_NODATA,
) -> np.ndarray:
    """Считает hillshade (освещённость рельефа) методом Horn.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.
        azimuth_deg: азимут солнца (0 = север, 90 = восток), градусы.
        altitude_deg: высота солнца над горизонтом, градусы.
        z_factor: вертикальное преувеличение рельефа.
        nodata: значение nodata.

    Returns:
        2D массив float32 [0, 255] с освещённостью.
    """
    azimuth_rad = azimuth_deg * _DEG_TO_RAD
    altitude_rad = altitude_deg * _DEG_TO_RAD

    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    gy, gx = np.gradient(dtm_filled * z_factor, pixel_size)

    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect_rad = np.arctan2(-gx, -gy)

    zenith_rad = np.pi / 2.0 - altitude_rad
    shaded = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(zenith_rad) * np.sin(
        slope_rad
    ) * np.cos(azimuth_rad - aspect_rad)
    shaded = np.clip(shaded, 0.0, 1.0)
    result = shaded * 255.0
    return _apply_nodata_mask(result, dtm, nodata).astype(np.float32)


def compute_slrm(
    dtm: np.ndarray,
    sigma: float,
    nodata: float = _DEFAULT_NODATA,
) -> np.ndarray:
    """Считает Simple Local Relief Model.

    SLRM = DTM − gaussian_blur(DTM, sigma).

    Убирает крупные формы рельефа, оставляет локальные.

    Args:
        dtm: 2D массив высот (метры).
        sigma: сигма гауссова размытия в пикселях.
        nodata: значение nodata.

    Returns:
        2D массив float32 с локальным рельефом.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    smoothed = ndimage.gaussian_filter(dtm_filled, sigma=sigma, mode="nearest")
    slrm = dtm_filled - smoothed
    return _apply_nodata_mask(slrm, dtm, nodata).astype(np.float32)


def compute_slope_aspect(
    dtm: np.ndarray,
    pixel_size: float,
    nodata: float = _DEFAULT_NODATA,
) -> tuple[np.ndarray, np.ndarray]:
    """Считает slope и aspect за один проход градиента.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.
        nodata: значение nodata.

    Returns:
        Кортеж (slope_deg, aspect_deg) — float32.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    gy, gx = np.gradient(dtm_filled, pixel_size)

    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope_deg = slope_rad / _DEG_TO_RAD

    aspect_rad = np.arctan2(-gx, -gy)
    aspect_deg = (aspect_rad / _DEG_TO_RAD) % 360.0

    slope_out = _apply_nodata_mask(slope_deg, dtm, nodata).astype(np.float32)
    aspect_out = _apply_nodata_mask(aspect_deg, dtm, nodata).astype(np.float32)
    return slope_out, aspect_out


# ============================================================
# Итерация 2: openness, sky-view factor, curvature
# ============================================================


def _compute_openness_direction(
    dtm: np.ndarray,
    pixel_size: float,
    azimuth_rad: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Считает positive и negative openness в одном направлении.

    Принимает уже заполненный DTM (без nodata).

    Args:
        dtm: 2D массив высот без nodata.
        pixel_size: размер пикселя в метрах.
        azimuth_rad: направление в радианах.
        radius: радиус поиска в пикселях.

    Returns:
        Кортеж (positive, negative) — 2D float32 массивы с углами в градусах.
    """
    height, width = dtm.shape
    max_elevation = np.full((height, width), -90.0, dtype=np.float32)
    min_elevation = np.full((height, width), 90.0, dtype=np.float32)

    cos_a = np.cos(azimuth_rad)
    sin_a = np.sin(azimuth_rad)

    for step in range(1, radius + 1):
        dx = int(round(step * cos_a))
        dy = int(round(step * sin_a))
        if dx == 0 and dy == 0:
            continue

        shifted = np.roll(np.roll(dtm, -dy, axis=0), -dx, axis=1)

        distance = step * pixel_size
        dz = shifted - dtm
        angle = np.degrees(np.arctan2(dz, distance)).astype(np.float32)

        max_elevation = np.maximum(max_elevation, angle)
        min_elevation = np.minimum(min_elevation, angle)

    positive = (90.0 - max_elevation).astype(np.float32)
    negative = (90.0 + min_elevation).astype(np.float32)
    return positive, negative


def compute_openness(
    dtm: np.ndarray,
    pixel_size: float,
    radius: int,
    n_directions: int,
    nodata: float = _DEFAULT_NODATA,
) -> tuple[np.ndarray, np.ndarray]:
    """Считает positive и negative openness.

    Args:
        dtm: 2D массив высот.
        pixel_size: размер пикселя в метрах.
        radius: радиус поиска в пикселях.
        n_directions: число направлений (например, 16).
        nodata: значение nodata.

    Returns:
        Кортеж (positive, negative) — 2D float32 массивы.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)

    height, width = dtm_filled.shape
    positive_sum = np.zeros((height, width), dtype=np.float64)
    negative_sum = np.zeros((height, width), dtype=np.float64)

    for i in range(n_directions):
        azimuth_rad = 2.0 * np.pi * i / n_directions
        pos, neg = _compute_openness_direction(
            dtm=dtm_filled,
            pixel_size=pixel_size,
            azimuth_rad=azimuth_rad,
            radius=radius,
        )
        positive_sum += pos
        negative_sum += neg

    positive = positive_sum / n_directions
    negative = negative_sum / n_directions

    positive_out = _apply_nodata_mask(positive, dtm, nodata).astype(np.float32)
    negative_out = _apply_nodata_mask(negative, dtm, nodata).astype(np.float32)
    return positive_out, negative_out


def compute_sky_view_factor(
    positive_openness: np.ndarray,
    negative_openness: np.ndarray,
) -> np.ndarray:
    """Считает sky-view factor из openness.

    Упрощение: sky-view factor = positive_openness / 180.
    Значения в [0, 1].

    Args:
        positive_openness: positive openness в градусах.
        negative_openness: negative openness в градусах (не используется).

    Returns:
        2D float32 массив sky-view factor [0, 1].
    """
    svf = positive_openness / 180.0
    svf = np.clip(svf, 0.0, 1.0)
    return svf.astype(np.float32)


def compute_curvature(
    dtm: np.ndarray,
    pixel_size: float,
    sigma: float,
    nodata: float = _DEFAULT_NODATA,
) -> np.ndarray:
    """Считает локальную кривизну (лапласиан).

    Args:
        dtm: 2D массив высот.
        pixel_size: размер пикселя в метрах.
        sigma: сигма гауссова размытия.
        nodata: значение nodata.

    Returns:
        2D float32 массив кривизны.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    smoothed = ndimage.gaussian_filter(dtm_filled, sigma=sigma, mode="nearest")
    laplacian = ndimage.laplace(smoothed) / (pixel_size * pixel_size)
    return _apply_nodata_mask(laplacian, dtm, nodata).astype(np.float32)


# ============================================================
# Итерация 3: TPI, TRI
# ============================================================


def compute_tpi(
    dtm: np.ndarray,
    radius: int,
    nodata: float = _DEFAULT_NODATA,
) -> np.ndarray:
    """Считает Topographic Position Index.

    TPI = DTM − mean(DTM в окне радиуса).

    Args:
        dtm: 2D массив высот (метры).
        radius: радиус окна в пикселях.
        nodata: значение nodata.

    Returns:
        2D float32 массив TPI.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    kernel_size = 2 * radius + 1
    mean = ndimage.uniform_filter(dtm_filled, size=kernel_size, mode="nearest")
    tpi = dtm_filled - mean
    return _apply_nodata_mask(tpi, dtm, nodata).astype(np.float32)


def compute_tri(
    dtm: np.ndarray,
    radius: int,
    nodata: float = _DEFAULT_NODATA,
) -> np.ndarray:
    """Считает Terrain Ruggedness Index.

    TRI = sqrt(mean((DTM − mean(DTM в окне))²)).

    Args:
        dtm: 2D массив высот (метры).
        radius: радиус окна в пикселях.
        nodata: значение nodata.

    Returns:
        2D float32 массив TRI.
    """
    dtm_filled = _fill_nodata_for_filter(dtm, nodata)
    kernel_size = 2 * radius + 1
    mean = ndimage.uniform_filter(dtm_filled, size=kernel_size, mode="nearest")
    mean_sq = ndimage.uniform_filter(dtm_filled * dtm_filled, size=kernel_size, mode="nearest")
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    tri = np.sqrt(variance)
    return _apply_nodata_mask(tri, dtm, nodata).astype(np.float32)


# ============================================================
# Сборка всех слоёв
# ============================================================


def build_relief_layers(
    dtm: np.ndarray,
    pixel_size: float,
    hillshade_azimuths: tuple[float, ...],
    hillshade_altitude: float,
    hillshade_z_factor: float,
    slrm_sigmas: tuple[float, ...],
    openness_radius: int,
    openness_n_directions: int,
    curvature_sigmas: tuple[float, ...],
    tpi_radii: tuple[int, ...],
    tri_radii: tuple[int, ...],
    nodata: float = _DEFAULT_NODATA,
) -> dict[str, np.ndarray]:
    """Строит все производные рельефа.

    Args:
        dtm: 2D массив высот.
        pixel_size: размер пикселя в метрах.
        hillshade_azimuths: список азимутов.
        hillshade_altitude: высота солнца.
        hillshade_z_factor: вертикальное преувеличение.
        slrm_sigmas: список сигм для SLRM.
        openness_radius: радиус для openness.
        openness_n_directions: число направлений для openness.
        curvature_sigmas: список сигм для кривизны.
        tpi_radii: список радиусов для TPI.
        tri_radii: список радиусов для TRI.
        nodata: значение nodata.

    Returns:
        Словарь слоёв.
    """
    layers: dict[str, np.ndarray] = {}

    slope, aspect = compute_slope_aspect(dtm=dtm, pixel_size=pixel_size, nodata=nodata)
    layers["slope"] = slope
    layers["aspect"] = aspect
    logger.info("Построены slope и aspect")

    for azimuth in hillshade_azimuths:
        name = f"hillshade_az_{int(azimuth)}"
        layers[name] = compute_hillshade(
            dtm=dtm,
            pixel_size=pixel_size,
            azimuth_deg=azimuth,
            altitude_deg=hillshade_altitude,
            z_factor=hillshade_z_factor,
            nodata=nodata,
        )
        logger.info("Построен %s", name)

    for sigma in slrm_sigmas:
        name = f"slrm_sigma_{sigma}"
        layers[name] = compute_slrm(dtm=dtm, sigma=sigma, nodata=nodata)
        logger.info("Построен %s", name)

    positive, negative = compute_openness(
        dtm=dtm,
        pixel_size=pixel_size,
        radius=openness_radius,
        n_directions=openness_n_directions,
        nodata=nodata,
    )
    layers["positive_openness"] = positive
    layers["negative_openness"] = negative
    layers["sky_view_factor"] = compute_sky_view_factor(
        positive_openness=positive,
        negative_openness=negative,
    )
    logger.info("Построены openness и sky_view_factor")

    for sigma in curvature_sigmas:
        name = f"curvature_sigma_{sigma}"
        layers[name] = compute_curvature(dtm=dtm, pixel_size=pixel_size, sigma=sigma, nodata=nodata)
        logger.info("Построен %s", name)

    for radius in tpi_radii:
        name = f"tpi_radius_{radius}"
        layers[name] = compute_tpi(dtm=dtm, radius=radius, nodata=nodata)
        logger.info("Построен %s", name)

    for radius in tri_radii:
        name = f"tri_radius_{radius}"
        layers[name] = compute_tri(dtm=dtm, radius=radius, nodata=nodata)
        logger.info("Построен %s", name)

    return layers
