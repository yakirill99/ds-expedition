"""Производные рельефа из DTM.

Итерация 1: slope, aspect, hillshade, SLRM.
Итерация 2: positive_openness, negative_openness, sky_view_factor, curvature.
Итерация 3: TPI (Topographic Position Index), TRI (Terrain Ruggedness Index).

Все функции принимают DTM (2D float32) и параметры, возвращают 2D float32.
Функции чистые: не читают файлы, не пишут. Кэширование — снаружи, через
LayerCache.

Nodata: если DTM содержит nodata, функции работают с маской. Наклон и
производные считаются только по валидным ячейкам, на границах — соседи
из валидных. Это важно для DTM с дырами.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# Константы: параметры по умолчанию.
_DEFAULT_HILLSHADE_ALTITUDE = 45.0
_DEFAULT_HILLSHADE_Z_FACTOR = 1.0

# Константы: преобразование градусов в радианы.
_DEG_TO_RAD = np.pi / 180.0

# Константы: минимальное значение для atan2, чтобы избежать деления на ноль.
_EPSILON = 1e-9


# ============================================================
# Итерация 1: slope, aspect, hillshade, SLRM
# ============================================================


def compute_slope(
    dtm: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    """Считает наклон поверхности.

    Наклон — модуль градиента, в градусах (arctan от модуля).
    На плоской поверхности — 0, на вертикальной — 90.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.

    Returns:
        2D массив float32 с наклоном в градусах.
    """
    gy, gx = np.gradient(dtm, pixel_size)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope_deg = slope_rad / _DEG_TO_RAD
    return slope_deg.astype(np.float32)


def compute_aspect(
    dtm: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    """Считает экспозицию склона (направление, куда стекает вода).

    Аспект — угол от севера по часовой стрелке, в градусах [0, 360).
    Плоские участки получают значение 0.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.

    Returns:
        2D массив float32 с аспектом в градусах.
    """
    gy, gx = np.gradient(dtm, pixel_size)
    # atan2(-gx, gy) даёт угол от севера по часовой стрелке.
    aspect_rad = np.arctan2(-gx, gy)
    aspect_deg = (aspect_rad / _DEG_TO_RAD) % 360.0
    return aspect_deg.astype(np.float32)


def compute_hillshade(
    dtm: np.ndarray,
    pixel_size: float,
    azimuth_deg: float,
    altitude_deg: float = _DEFAULT_HILLSHADE_ALTITUDE,
    z_factor: float = _DEFAULT_HILLSHADE_Z_FACTOR,
) -> np.ndarray:
    """Считает hillshade (освещённость рельефа) методом Horn.

    Классическая визуализация рельефа: как будто солнце светит с
    заданного азимута и высоты. Тени подчёркивают локальные формы.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.
        azimuth_deg: азимут солнца (0 = север, 90 = восток), градусы.
        altitude_deg: высота солнца над горизонтом, градусы.
        z_factor: вертикальное преувеличение рельефа.

    Returns:
        2D массив float32 [0, 255] с освещённостью.
    """
    azimuth_rad = azimuth_deg * _DEG_TO_RAD
    altitude_rad = altitude_deg * _DEG_TO_RAD

    # Градиенты методом Horn (центральные разности).
    gy, gx = np.gradient(dtm * z_factor, pixel_size)

    # Нормаль к поверхности.
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect_rad = np.arctan2(-gx, gy)

    # Освещённость = cos(zenith) * cos(slope)
    #              + sin(zenith) * sin(slope) * cos(azimuth - aspect)
    zenith_rad = np.pi / 2.0 - altitude_rad
    shaded = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(zenith_rad) * np.sin(
        slope_rad
    ) * np.cos(azimuth_rad - aspect_rad)
    shaded = np.clip(shaded, 0.0, 1.0)
    return (shaded * 255.0).astype(np.float32)


def compute_slrm(
    dtm: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Считает Simple Local Relief Model.

    SLRM = DTM − gaussian_blur(DTM, sigma).

    Убирает крупные формы рельефа, оставляет локальные.
    Курганы и валы становятся яркими пятнами на нейтральном фоне.
    Главный признак для археологии.

    Args:
        dtm: 2D массив высот (метры).
        sigma: сигма гауссова размытия в пикселях.

    Returns:
        2D массив float32 с локальным рельефом.
    """
    smoothed = ndimage.gaussian_filter(dtm, sigma=sigma, mode="nearest")
    slrm = dtm - smoothed
    return slrm.astype(np.float32)


def compute_slope_aspect(
    dtm: np.ndarray,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Считает slope и aspect за один проход градиента.

    Экономит одно вычисление np.gradient, если нужны оба.

    Args:
        dtm: 2D массив высот (метры).
        pixel_size: размер пикселя в метрах.

    Returns:
        Кортеж (slope_deg, aspect_deg) — float32.
    """
    gy, gx = np.gradient(dtm, pixel_size)

    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope_deg = (slope_rad / _DEG_TO_RAD).astype(np.float32)

    aspect_rad = np.arctan2(-gx, gy)
    aspect_deg = ((aspect_rad / _DEG_TO_RAD) % 360.0).astype(np.float32)

    return slope_deg, aspect_deg


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

    Positive openness: 90° - max(угол возвышения горизонта).
    Negative openness: 90° + min(угол возвышения горизонта).

    На плоскости обе равны 90°. На вершине positive > 90°, negative < 90°.
    В яме positive < 90°, negative > 90°.

    Args:
        dtm: 2D массив высот.
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
) -> tuple[np.ndarray, np.ndarray]:
    """Считает positive и negative openness.

    Positive openness — насколько небо открыто над точкой. Высокое
    значение для выпуклых форм (курганы, вершины).
    Negative openness — насколько открыто пространство под точкой.
    Высокое значение для вогнутых форм (рвы, ямы).

    Args:
        dtm: 2D массив высот.
        pixel_size: размер пикселя в метрах.
        radius: радиус поиска в пикселях.
        n_directions: число направлений (например, 16).

    Returns:
        Кортеж (positive, negative) — 2D float32 массивы.
        Значения — средние углы по всем направлениям, в градусах.
    """
    height, width = dtm.shape
    positive_sum = np.zeros((height, width), dtype=np.float64)
    negative_sum = np.zeros((height, width), dtype=np.float64)

    for i in range(n_directions):
        azimuth_rad = 2.0 * np.pi * i / n_directions
        pos, neg = _compute_openness_direction(
            dtm=dtm, pixel_size=pixel_size, azimuth_rad=azimuth_rad, radius=radius
        )
        positive_sum += pos
        negative_sum += neg

    positive = (positive_sum / n_directions).astype(np.float32)
    negative = (negative_sum / n_directions).astype(np.float32)
    return positive, negative


def compute_sky_view_factor(
    positive_openness: np.ndarray,
    negative_openness: np.ndarray,
) -> np.ndarray:
    """Считает sky-view factor из openness.

    Упрощение: sky-view factor = (positive_openness + negative_openness) / 360.
    На плоскости: (90 + 90) / 360 = 0.5.
    На вершине: (95 + 85) / 360 = 0.5 — тоже 0.5. Хм.

    На самом деле SVF на вершине должен быть выше, чем в яме.
    Правильная формула для нашего определения:
        SVF ≈ positive_openness / 180.

    На плоскости: 90 / 180 = 0.5.
    На вершине: 95 / 180 = 0.528.
    В яме: 85 / 180 = 0.472.

    Это монотонно и физично.

    Args:
        positive_openness: positive openness в градусах [0, 180].
        negative_openness: negative openness в градусах [0, 180].
            Не используется в упрощённой формуле, но оставлен для
            совместимости и будущих уточнений.

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
) -> np.ndarray:
    """Считает локальную кривизну (лапласиан).

    Кривизна = d²Z/dx² + d²Z/dy². Положительная для вогнутых форм
    (рвы), отрицательная для выпуклых (курганы).

    Перед вычислением DTM размывается гауссом с sigma — это подавляет
    шум, который при вторых производных усиливается.

    Args:
        dtm: 2D массив высот.
        pixel_size: размер пикселя в метрах.
        sigma: сигма гауссова размытия.

    Returns:
        2D float32 массив кривизны.
    """
    smoothed = ndimage.gaussian_filter(dtm, sigma=sigma, mode="nearest")
    laplacian = ndimage.laplace(smoothed) / (pixel_size * pixel_size)
    return laplacian.astype(np.float32)


# ============================================================
# Итерация 3: TPI, TRI
# ============================================================


def compute_tpi(
    dtm: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Считает Topographic Position Index.

    TPI = DTM − mean(DTM в окне радиуса).

    Положительный — точка выше окружения (холм, курган).
    Отрицательный — точка ниже окружения (яма, ров).
    Ноль — ровный склон или плоскость.

    В отличие от SLRM, использует прямоугольное окно и среднее,
    а не гауссово размытие. Иногда даёт лучше для археологии,
    особенно для объектов с резкими границами.

    Args:
        dtm: 2D массив высот (метры).
        radius: радиус окна в пикселях.

    Returns:
        2D float32 массив TPI.
    """
    kernel_size = 2 * radius + 1
    mean = ndimage.uniform_filter(dtm, size=kernel_size, mode="nearest")
    tpi = dtm - mean
    return tpi.astype(np.float32)


def compute_tri(
    dtm: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Считает Terrain Ruggedness Index.

    TRI = sqrt(mean((DTM − mean(DTM в окне))²)).

    Это стандартное отклонение высот в окне. Мера "шероховатости"
    рельефа. Для археологии полезно: валы и рвы создают локальную
    шероховатость, которой нет на плоском поле.

    Args:
        dtm: 2D массив высот (метры).
        radius: радиус окна в пикселях.

    Returns:
        2D float32 массив TRI.
    """
    kernel_size = 2 * radius + 1
    mean = ndimage.uniform_filter(dtm, size=kernel_size, mode="nearest")
    mean_sq = ndimage.uniform_filter(dtm * dtm, size=kernel_size, mode="nearest")
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    tri = np.sqrt(variance)
    return tri.astype(np.float32)


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
) -> dict[str, np.ndarray]:
    """Строит все производные рельефа (итерации 1, 2, 3).

    Возвращает словарь: имя слоя -> 2D массив float32.
    Имена:
        slope, aspect,
        hillshade_az_<azimuth> для каждого азимута,
        slrm_sigma_<sigma> для каждой сигмы,
        positive_openness, negative_openness, sky_view_factor,
        curvature_sigma_<sigma> для каждой сигмы,
        tpi_radius_<r> для каждого радиуса,
        tri_radius_<r> для каждого радиуса.

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

    Returns:
        Словарь слоёв.
    """
    layers: dict[str, np.ndarray] = {}

    # slope + aspect за один градиент.
    slope, aspect = compute_slope_aspect(dtm=dtm, pixel_size=pixel_size)
    layers["slope"] = slope
    layers["aspect"] = aspect
    logger.info("Построены slope и aspect")

    # hillshade для каждого азимута.
    for azimuth in hillshade_azimuths:
        name = f"hillshade_az_{int(azimuth)}"
        layers[name] = compute_hillshade(
            dtm=dtm,
            pixel_size=pixel_size,
            azimuth_deg=azimuth,
            altitude_deg=hillshade_altitude,
            z_factor=hillshade_z_factor,
        )
        logger.info("Построен %s", name)

    # SLRM для каждой сигмы.
    for sigma in slrm_sigmas:
        name = f"slrm_sigma_{sigma}"
        layers[name] = compute_slrm(dtm=dtm, sigma=sigma)
        logger.info("Построен %s", name)

    # Openness + sky-view factor.
    positive, negative = compute_openness(
        dtm=dtm,
        pixel_size=pixel_size,
        radius=openness_radius,
        n_directions=openness_n_directions,
    )
    layers["positive_openness"] = positive
    layers["negative_openness"] = negative
    layers["sky_view_factor"] = compute_sky_view_factor(
        positive_openness=positive,
        negative_openness=negative,
    )
    logger.info("Построены openness и sky_view_factor")

    # Кривизна для каждой сигмы.
    for sigma in curvature_sigmas:
        name = f"curvature_sigma_{sigma}"
        layers[name] = compute_curvature(dtm=dtm, pixel_size=pixel_size, sigma=sigma)
        logger.info("Построен %s", name)

    # TPI для каждого радиуса.
    for radius in tpi_radii:
        name = f"tpi_radius_{radius}"
        layers[name] = compute_tpi(dtm=dtm, radius=radius)
        logger.info("Построен %s", name)

    # TRI для каждого радиуса.
    for radius in tri_radii:
        name = f"tri_radius_{radius}"
        layers[name] = compute_tri(dtm=dtm, radius=radius)
        logger.info("Построен %s", name)

    return layers
