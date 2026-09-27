"""Фильтр земли для облака точек лидара.

Алгоритм — прогрессивный морфологический фильтр (Zhang et al., 2003):

1. Строим грубую сетку минимальных Z (min surface) по облаку точек.
2. Прогрессивно применяем морфологическое открытие с растущим окном.
   Это сглаживает выступы (растительность, здания), но сохраняет
   крупные положительные формы (насыпи, курганы).
3. Считаем наклон сглаженной поверхности.
4. Для каждой точки: земля, если
       Z - surface <= height_threshold
   и   наклон в её ячейке <= slope_threshold.
5. Итеративно добавляем отброшенные точки, которые близки к поверхности.

Два пресета:
- strict: агрессивно убирает всё, что выше поверхности. Хорош для равнин.
- soft:   сохраняет крупные положительные формы. Хорош для археологии.

Модуль работает на точках, но использует Grid и scipy.ndimage для
морфологии. Все параметры — из GroundConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from expds.features.config import GroundConfig
from expds.features.grid import Grid

logger = logging.getLogger(__name__)

# Константы: имена пресетов.
_PRESET_STRICT = "strict"
_PRESET_SOFT = "soft"

# Константы: параметры пресетов. Переопределяют часть GroundConfig.
# strict: меньший height_threshold, больше итераций.
# soft:   больший height_threshold, меньше итераций.
_PRESET_OVERRIDES: dict[str, dict[str, float | int | tuple[int, ...]]] = {
    _PRESET_STRICT: {
        "height_threshold": 0.3,
        "slope_threshold": 0.3,
        "max_iterations": 8,
        "window_sizes": (3, 5, 7, 9, 11, 13, 15),
    },
    _PRESET_SOFT: {
        "height_threshold": 1.0,
        "slope_threshold": 0.15,
        "max_iterations": 4,
        "window_sizes": (5, 9, 15),
    },
}

# Константы: для вычисления наклона.
_SLOPE_EPSILON = 1e-6


@dataclass(frozen=True)
class GroundResult:
    """Результат фильтра земли.

    ground_mask — булев массив на точки (True = земля).
    surface — сглаженная поверхность в Grid (для отладки).
    slope — наклон поверхности в Grid (для отладки).
    n_iterations — сколько итераций реально потребовалось.
    """

    ground_mask: np.ndarray
    surface: np.ndarray
    slope: np.ndarray
    n_iterations: int


def _resolve_config(config: GroundConfig) -> GroundConfig:
    """Применяет пресет к конфигу.

    Пресет переопределяет height_threshold, slope_threshold,
    max_iterations и window_sizes. Остальные поля (cell_size,
    min_change_ratio) берутся из конфига как есть.

    Args:
        config: исходный конфиг из YAML.

    Returns:
        Новый GroundConfig с применённым пресетом.

    Raises:
        ValueError: если preset неизвестен.
    """
    if config.preset not in _PRESET_OVERRIDES:
        raise ValueError(
            f"Неизвестный preset '{config.preset}'. Доступны: {sorted(_PRESET_OVERRIDES)}"
        )

    overrides = _PRESET_OVERRIDES[config.preset]
    return GroundConfig(
        cell_size=config.cell_size,
        preset=config.preset,
        window_sizes=overrides["window_sizes"],  # type: ignore[arg-type]
        height_threshold=float(overrides["height_threshold"]),
        slope_threshold=float(overrides["slope_threshold"]),
        max_iterations=int(overrides["max_iterations"]),
        min_change_ratio=config.min_change_ratio,
    )


def _build_grid_from_points(
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    crs: str,
) -> Grid:
    """Строит внутреннюю сетку по границам точек.

    Args:
        x: массив X-координат точек.
        y: массив Y-координат точек.
        cell_size: размер ячейки в единицах CRS.
        crs: CRS сетки.

    Returns:
        Grid, покрывающий все точки.
    """
    x_min = float(x.min())
    x_max = float(x.max())
    y_min = float(y.min())
    y_max = float(y.max())

    width = int(np.ceil((x_max - x_min) / cell_size)) + 1
    height = int(np.ceil((y_max - y_min) / cell_size)) + 1

    grid = Grid(
        crs=crs,
        pixel_size=cell_size,
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
    )
    logger.info(
        "Внутренняя сетка фильтра: %dx%d, pixel=%.2f, CRS=%s",
        grid.width,
        grid.height,
        cell_size,
        crs,
    )
    return grid


def _build_min_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    """Строит поверхность минимальных Z по сетке.

    Для каждой ячейки берётся минимальная Z среди попавших точек.
    Пустые ячейки заполняются nodata, потом интерполируются.

    Args:
        x: X-координаты точек.
        y: Y-координаты точек.
        z: Z-координаты точек.
        grid: внутренняя сетка.

    Returns:
        np.ndarray float32 формы grid.shape с минимальными Z.
    """
    col, row = grid.transform_to_pixel(x=x, y=y)
    col_idx = np.floor(col).astype(np.int64)
    row_idx = np.floor(row).astype(np.int64)

    # Отбрасываем точки вне сетки (численные погрешности).
    valid = (col_idx >= 0) & (col_idx < grid.width) & (row_idx >= 0) & (row_idx < grid.height)
    col_idx = col_idx[valid]
    row_idx = row_idx[valid]
    z_valid = z[valid]

    surface = np.full(grid.shape, np.inf, dtype=np.float64)
    np.minimum.at(surface, (row_idx, col_idx), z_valid)

    # Пустые ячейки -> nan, потом заполним ближайшим.
    empty = ~np.isfinite(surface)
    if empty.any():
        logger.debug("Пустых ячеек в min surface: %d", int(empty.sum()))
        surface[empty] = np.nan
        # Заполняем через дистанционное преобразование: ближайший валидный.
        indices = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
        surface = surface[tuple(indices)]

    return surface.astype(np.float32)


def _compute_slope(surface: np.ndarray, cell_size: float) -> np.ndarray:
    """Считает наклон поверхности.

    Наклон — модуль градиента. В единицах tan(угол).

    Args:
        surface: 2D массив высот.
        cell_size: размер ячейки в тех же единицах, что и высота.

    Returns:
        np.ndarray float32 формы surface.shape с наклоном.
    """
    gy, gx = np.gradient(surface, cell_size)
    slope = np.sqrt(gx * gx + gy * gy)
    return slope.astype(np.float32)


def _progressive_opening(
    surface: np.ndarray,
    window_sizes: tuple[int, ...],
) -> np.ndarray:
    """Прогрессивное морфологическое открытие.

    Последовательно применяет grey_opening с растущим окном.
    Открытие = эрозия (min filter) + дилатация (max filter).
    Сглаживает положительные выступы, сохраняя крупные формы.

    Args:
        surface: 2D массив высот.
        window_sizes: размеры окон (нечётные) в пикселях.

    Returns:
        Сглаженная поверхность.
    """
    result = surface.copy()
    for size in window_sizes:
        if size % 2 == 0:
            size += 1  # гарантируем нечётность
        result = ndimage.grey_opening(result, size=(size, size), mode="nearest")
        logger.debug("Открытие с окном %d завершено", size)
    return result


def _classify_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    surface: np.ndarray,
    slope: np.ndarray,
    grid: Grid,
    height_threshold: float,
    slope_threshold: float,
) -> np.ndarray:
    """Классифицирует точки: земля или нет.

    Точка — земля, если:
        Z - surface <= height_threshold
    и   наклон в её ячейке <= slope_threshold.

    Args:
        x, y, z: координаты точек.
        surface: сглаженная поверхность.
        slope: наклон поверхности.
        grid: внутренняя сетка.
        height_threshold: порог по высоте.
        slope_threshold: порог по наклону.

    Returns:
        Булев массив формы (N,) — True для земли.
    """
    col, row = grid.transform_to_pixel(x=x, y=y)
    col_idx = np.clip(np.floor(col).astype(np.int64), 0, grid.width - 1)
    row_idx = np.clip(np.floor(row).astype(np.int64), 0, grid.height - 1)

    surface_at_point = surface[row_idx, col_idx]
    slope_at_point = slope[row_idx, col_idx]

    height_ok = (z - surface_at_point) <= height_threshold
    slope_ok = slope_at_point <= slope_threshold

    return height_ok & slope_ok


def _refine_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    grid: Grid,
    current_surface: np.ndarray,
) -> np.ndarray:
    """Уточняет поверхность по текущей маске земли.

    Строит min surface только по точкам, помеченным как земля,
    и заполняет пустые ячейки ближайшим значением.

    Args:
        x, y, z: координаты точек.
        ground_mask: текущая маска земли.
        grid: внутренняя сетка.
        current_surface: текущая поверхность (используется как fallback).

    Returns:
        Уточнённая поверхность.
    """
    if not ground_mask.any():
        logger.warning("Маска земли пуста, поверхность не уточняется")
        return current_surface

    x_g = x[ground_mask]
    y_g = y[ground_mask]
    z_g = z[ground_mask]

    col, row = grid.transform_to_pixel(x=x_g, y=y_g)
    col_idx = np.clip(np.floor(col).astype(np.int64), 0, grid.width - 1)
    row_idx = np.clip(np.floor(row).astype(np.int64), 0, grid.height - 1)

    surface = np.full(grid.shape, np.inf, dtype=np.float64)
    np.minimum.at(surface, (row_idx, col_idx), z_g)

    empty = ~np.isfinite(surface)
    if empty.any():
        surface[empty] = np.nan
        indices = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
        surface = surface[tuple(indices)]

    return surface.astype(np.float32)


def filter_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: GroundConfig,
    crs: str,
) -> GroundResult:
    """Фильтрует облако точек: отделяет землю от всего остального.

    Args:
        x: X-координаты точек (метры, в CRS).
        y: Y-координаты точек (метры, в CRS).
        z: Z-координаты точек (метры).
        config: параметры фильтра (с пресетом).
        crs: CRS точек (для внутренней сетки).

    Returns:
        GroundResult с маской земли, поверхностью, наклоном, числом итераций.

    Raises:
        ValueError: если массивы разной длины или preset неизвестен.
    """
    if not (len(x) == len(y) == len(z)):
        raise ValueError(f"Длины x, y, z должны совпадать: {len(x)}, {len(y)}, {len(z)}")
    if len(x) == 0:
        raise ValueError("Пустое облако точек")

    resolved = _resolve_config(config=config)
    logger.info(
        "Фильтр земли: preset=%s, точек=%d, cell_size=%.2f, "
        "height_threshold=%.2f, slope_threshold=%.2f",
        resolved.preset,
        len(x),
        resolved.cell_size,
        resolved.height_threshold,
        resolved.slope_threshold,
    )

    # 1. Внутренняя сетка.
    grid = _build_grid_from_points(x=x, y=y, cell_size=resolved.cell_size, crs=crs)

    # 2. Min surface и прогрессивное открытие.
    surface = _build_min_surface(x=x, y=y, z=z, grid=grid)
    surface = _progressive_opening(surface=surface, window_sizes=resolved.window_sizes)

    # 3. Итеративное уточнение.
    ground_mask = np.zeros(len(x), dtype=bool)
    n_iterations = 0

    for iteration in range(resolved.max_iterations):
        slope = _compute_slope(surface=surface, cell_size=resolved.cell_size)
        new_mask = _classify_points(
            x=x,
            y=y,
            z=z,
            surface=surface,
            slope=slope,
            grid=grid,
            height_threshold=resolved.height_threshold,
            slope_threshold=resolved.slope_threshold,
        )

        n_ground = int(new_mask.sum())
        n_prev = int(ground_mask.sum())
        change_ratio = abs(n_ground - n_prev) / max(n_prev, 1)

        logger.info(
            "Итерация %d: земли %d (%.1f%%), изменение %.4f",
            iteration + 1,
            n_ground,
            100.0 * n_ground / len(x),
            change_ratio,
        )

        ground_mask = new_mask
        n_iterations = iteration + 1

        if change_ratio < resolved.min_change_ratio and iteration > 0:
            logger.info("Стабилизация достигнута на итерации %d", n_iterations)
            break

        surface = _refine_surface(
            x=x,
            y=y,
            z=z,
            ground_mask=ground_mask,
            grid=grid,
            current_surface=surface,
        )
        surface = _progressive_opening(surface=surface, window_sizes=resolved.window_sizes)

    slope = _compute_slope(surface=surface, cell_size=resolved.cell_size)

    return GroundResult(
        ground_mask=ground_mask,
        surface=surface,
        slope=slope,
        n_iterations=n_iterations,
    )
