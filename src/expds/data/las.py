"""Чтение LAS/LAZ: заголовок, инвентаризация, чанковое чтение точек."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import laspy
import numpy as np
from pyproj import Transformer

logger = logging.getLogger(__name__)

# Константы: коэффициенты перевода единиц высоты.
_FEET_TO_METERS = 0.3048


@dataclass(frozen=True)
class LasHeaderInfo:
    """Информация из заголовка LAS-файла."""

    path: Path
    point_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    scales: tuple[float, float, float]
    offsets: tuple[float, float, float]
    crs: str | None


@dataclass(frozen=True)
class LasSummary:
    """Сводка по содержимому LAS-файла.

    Границы XY и диапазон Z — в метрах, в target_crs.
    z_range_raw — в исходных единицах файла, для контроля.
    """

    header: LasHeaderInfo
    bounds_xy_m: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    z_range_raw: tuple[float, float]
    z_range_meters: tuple[float, float]
    unit: str


def _build_transformer(
    source_crs: str,
    target_crs: str,
) -> Transformer:
    """Создаёт трансформер координат.

    Args:
        source_crs: исходный CRS (например, "EPSG:3358").
        target_crs: целевой CRS (например, "EPSG:32617").

    Returns:
        Transformer из pyproj.
    """
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _detect_z_unit(z_range_raw: tuple[float, float], declared_unit: str) -> str:
    """Проверяет, соответствует ли заявленная единица высоты фактическому диапазону.

    Args:
        z_range_raw: min и max Z в исходных единицах файла.
        declared_unit: единица из конфига (например, "feet" или "meters").

    Returns:
        Строка с фактической единицей: "feet" или "meters".
    """
    span = abs(z_range_raw[1] - z_range_raw[0])
    # Эвристика: если разброс высот больше 100, это почти наверняка футы.
    # Для Raleigh (Северная Каролина) рельеф пологий, метры дали бы десятки.
    if declared_unit == "feet" and span > 100:
        return "feet"
    if declared_unit == "meters" and span > 100:
        logger.warning(
            "Заявлены метры, но разброс Z=%.1f. Похоже на футы. Проверь конфиг.",
            span,
        )
        return "feet"
    return declared_unit


def _convert_z_to_meters(z_values: np.ndarray, unit: str) -> np.ndarray:
    """Переводит массив высот в метры.

    Args:
        z_values: массив высот в исходных единицах.
        unit: исходная единица ("feet" или "meters").

    Returns:
        Массив высот в метрах.
    """
    if unit == "feet":
        return z_values * _FEET_TO_METERS
    return z_values


def read_header(path: Path) -> LasHeaderInfo:
    """Читает только заголовок LAS/LAZ, без точек.

    Args:
        path: путь к LAS/LAZ-файлу.

    Returns:
        LasHeaderInfo с границами, числом точек, CRS, scale и offset.

    Raises:
        FileNotFoundError: если файл не найден.
        laspy.errors.LaspyException: если файл повреждён или не поддерживается.
    """
    if not path.exists():
        raise FileNotFoundError(f"LAS-файл не найден: {path}")

    with laspy.open(path) as las:
        header = las.header
        crs_obj = header.parse_crs()
        crs_str = crs_obj.to_string() if crs_obj is not None else None

        info = LasHeaderInfo(
            path=path,
            point_count=int(header.point_count),
            bounds_min=(
                float(header.mins[0]),
                float(header.mins[1]),
                float(header.mins[2]),
            ),
            bounds_max=(
                float(header.maxs[0]),
                float(header.maxs[1]),
                float(header.maxs[2]),
            ),
            scales=(
                float(header.scales[0]),
                float(header.scales[1]),
                float(header.scales[2]),
            ),
            offsets=(
                float(header.offsets[0]),
                float(header.offsets[1]),
                float(header.offsets[2]),
            ),
            crs=crs_str,
        )

    logger.info(
        "Заголовок прочитан: %s | точек=%d | CRS=%s",
        path.name,
        info.point_count,
        info.crs,
    )
    return info


def read_points(
    path: Path,
    chunk_size: int,
    source_crs: str,
    target_crs: str,
    source_z_unit: str,
    max_points: int | None = None,
) -> Iterator[np.ndarray]:
    """Читает точки LAS/LAZ чанками, возвращает метры в target_crs.

    Yields:
        np.ndarray формы (N, 6): X, Y, Z, intensity, return_number, classification.
        X, Y, Z — в метрах в target_crs.
    """
    if not path.exists():
        raise FileNotFoundError(f"LAS-файл не найден: {path}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size должен быть > 0, получено {chunk_size}")

    transformer = _build_transformer(source_crs=source_crs, target_crs=target_crs)
    produced = 0

    with laspy.open(path) as las:
        for chunk in las.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)

            # Z: единицы файла → метры.
            z = _convert_z_to_meters(z_values=z, unit=source_z_unit)

            # X, Y: перепроецирование в internal_crs.
            x_m, y_m = transformer.transform(x, y)

            intensity = np.asarray(chunk.intensity, dtype=np.float32)
            return_number = np.asarray(chunk.return_number, dtype=np.uint8)
            classification = np.asarray(chunk.classification, dtype=np.uint8)

            block = np.column_stack((x_m, y_m, z, intensity, return_number, classification))

            if max_points is not None:
                remaining = max_points - produced
                if remaining <= 0:
                    logger.info("Достигнут max_points=%d", max_points)
                    return
                if block.shape[0] > remaining:
                    block = block[:remaining]

            produced += block.shape[0]
            yield block


def summarize(
    path: Path,
    source_z_unit: str,
    chunk_size: int,
    source_crs: str,
    target_crs: str,
    max_points: int | None = None,
) -> LasSummary:
    """Собирает сводку по LAS-файлу: заголовок, границы XY и диапазон Z.

    Читает точки в target_crs и метрах, поэтому границы XY и Z — уже
    в метрах. Отдельно сохраняет исходный диапазон Z для контроля единиц.

    Args:
        path: путь к LAS/LAZ-файлу.
        source_z_unit: заявленная единица высоты из конфига.
        chunk_size: размер чанка при чтении.
        source_crs: CRS исходных данных (например, "EPSG:3358").
        target_crs: CRS для внутренних расчётов (например, "EPSG:32617").
        max_points: ограничение числа точек (None = все).

    Returns:
        LasSummary с заголовком, границами XY в target_crs (метры)
        и диапазонами Z в исходных единицах и метрах.
    """
    header = read_header(path=path)

    # Границы XY в метрах, в target_crs.
    x_min_m = float("inf")
    x_max_m = float("-inf")
    y_min_m = float("inf")
    y_max_m = float("-inf")
    # Диапазон Z в метрах (после конвертации единиц).
    z_min_m = float("inf")
    z_max_m = float("-inf")

    for block in read_points(
        path=path,
        chunk_size=chunk_size,
        source_crs=source_crs,
        target_crs=target_crs,
        source_z_unit=source_z_unit,
        max_points=max_points,
    ):
        x = block[:, 0]
        y = block[:, 1]
        z = block[:, 2]

        x_min_m = min(x_min_m, float(x.min()))
        x_max_m = max(x_max_m, float(x.max()))
        y_min_m = min(y_min_m, float(y.min()))
        y_max_m = max(y_max_m, float(y.max()))
        z_min_m = min(z_min_m, float(z.min()))
        z_max_m = max(z_max_m, float(z.max()))

    z_range_meters = (z_min_m, z_max_m)

    # Обратный пересчёт Z в исходные единицы — только для отчёта.
    if source_z_unit == "feet":
        z_range_raw = (z_min_m / _FEET_TO_METERS, z_max_m / _FEET_TO_METERS)
    else:
        z_range_raw = z_range_meters

    actual_unit = _detect_z_unit(z_range_raw=z_range_raw, declared_unit=source_z_unit)

    summary = LasSummary(
        header=header,
        bounds_xy_m=(x_min_m, y_min_m, x_max_m, y_max_m),
        z_range_raw=z_range_raw,
        z_range_meters=z_range_meters,
        unit=actual_unit,
    )

    width_km = (x_max_m - x_min_m) / 1000.0
    height_km = (y_max_m - y_min_m) / 1000.0
    logger.info(
        "Сводка: точек=%d | X_m=(%.1f, %.1f) | Y_m=(%.1f, %.1f) | "
        "размер=%.2f x %.2f км | Z_m=(%.2f, %.2f) | unit=%s",
        header.point_count,
        x_min_m,
        x_max_m,
        y_min_m,
        y_max_m,
        width_km,
        height_km,
        z_min_m,
        z_max_m,
        actual_unit,
    )
    return summary
