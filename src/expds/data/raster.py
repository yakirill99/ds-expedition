"""Чтение и запись растровых слоёв в GeoTIFF.

Контракт: Layer <-> GeoTIFF без потерь. Все растры хранятся в float32,
nodata кодируется как в Layer, CRS и геопривязка — из Grid.

Ответственность модуля:
- читать GeoTIFF в Layer (с опциональным ресэмплингом под заданный Grid);
- писать Layer в GeoTIFF с корректными метаданными;
- читать только метаданные (для инвентаризации), без данных.

Перепроецирование между CRS — НЕ здесь. Это задача prep/align.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)

# Константы: значения по умолчанию.
_DEFAULT_DTYPE = "float32"
_DEFAULT_NODATA = -9999.0
_DEFAULT_COMPRESS = "deflate"

# Константы: допуск при сравнении pixel_size и CRS.
_PIXEL_SIZE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RasterMeta:
    """Метаданные GeoTIFF без данных.

    Полезно для инвентаризации: узнать, что внутри файла, не читая массив.
    """

    path: Path
    crs: str | None
    pixel_size_x: float
    pixel_size_y: float
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    nodata: float | None
    dtype: str


def _build_grid_from_raster(
    crs: str,
    transform: Affine,
    width: int,
    height: int,
) -> Grid:
    """Строит наш Grid из параметров rasterio.

    rasterio хранит верхний левый угол. Наш Grid — нижний левый.
    Пересчитываем y_min = y_max - height * pixel_size.

    Args:
        crs: строка CRS (например, "EPSG:32617").
        transform: affine-трансформация rasterio.
        width: ширина в пикселях.
        height: высота в пикселях.

    Returns:
        Grid, соответствующий растру.
    """
    pixel_size_x = transform.a
    pixel_size_y = -transform.e

    if abs(pixel_size_x - pixel_size_y) > _PIXEL_SIZE_TOLERANCE:
        logger.warning(
            "Анизотропный пиксель: dx=%.6f, dy=%.6f. Берём dx как pixel_size.",
            pixel_size_x,
            pixel_size_y,
        )

    x_min = transform.c
    y_max = transform.f
    y_min = y_max - height * pixel_size_y

    return Grid(
        crs=crs,
        pixel_size=float(pixel_size_x),
        x_min=float(x_min),
        y_min=float(y_min),
        width=int(width),
        height=int(height),
    )


def _build_transform(grid: Grid) -> Affine:
    """Строит affine-трансформацию rasterio из нашего Grid.

    rasterio ожидает верхний левый угол:
        transform = Affine(pixel_size, 0, x_min, 0, -pixel_size, y_max)

    Args:
        grid: наша сетка.

    Returns:
        Affine для rasterio.
    """
    _, _, _, y_max = grid.bounds
    return Affine(
        grid.pixel_size,
        0.0,
        grid.x_min,
        0.0,
        -grid.pixel_size,
        y_max,
    )


def _resample_to_grid(
    source: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    src_nodata: float | None,
    target_grid: Grid,
    target_nodata: float,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Ресэмплит массив в целевую сетку.

    Nodata исходника передаётся в reproject, чтобы он не смешивал
    nodata-пиксели с валидными. После ресэмплинга пиксели, которые
    не получили валидного значения, заполняются target_nodata.

    Args:
        source: исходный 2D массив.
        src_transform: affine исходного растра.
        src_crs: CRS исходного растра.
        src_nodata: nodata исходного растра (или None).
        target_grid: целевая сетка.
        target_nodata: значение nodata для результата.
        resampling: метод ресэмплинга rasterio.

    Returns:
        np.ndarray float32 формы target_grid.shape.
    """
    dst_transform = _build_transform(target_grid)
    dst_crs = CRS.from_user_input(target_grid.crs)
    dst = np.full(target_grid.shape, target_nodata, dtype=np.float32)

    reproject(
        source=source,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=target_nodata,
        resampling=resampling,
    )
    return dst


def read_raster_meta(path: Path) -> RasterMeta:
    """Читает метаданные GeoTIFF без данных.

    Args:
        path: путь к GeoTIFF.

    Returns:
        RasterMeta с CRS, разрешением, границами, nodata и dtype.

    Raises:
        FileNotFoundError: если файл не найден.
    """
    if not path.exists():
        raise FileNotFoundError(f"Растр не найден: {path}")

    with rasterio.open(path) as src:
        transform = src.transform
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        crs_str = src.crs.to_string() if src.crs is not None else None

        meta = RasterMeta(
            path=path,
            crs=crs_str,
            pixel_size_x=float(abs(transform.a)),
            pixel_size_y=float(abs(transform.e)),
            width=int(src.width),
            height=int(src.height),
            bounds=(float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])),
            nodata=float(src.nodata) if src.nodata is not None else None,
            dtype=str(src.dtypes[0]),
        )

    logger.debug(
        "Метаданные растра: %s | CRS=%s | %dx%d | pixel=%.3f",
        path.name,
        meta.crs,
        meta.width,
        meta.height,
        meta.pixel_size_x,
    )
    return meta


def read_raster(
    path: Path,
    grid: Grid | None = None,
    nodata: float = _DEFAULT_NODATA,
    resampling: Resampling = Resampling.bilinear,
) -> Layer:
    """Читает GeoTIFF в Layer.

    Если grid задан и совпадает с растром (CRS, pixel_size, extent) —
    данные читаются как есть. Если grid задан и не совпадает — выполняется
    ресэмплинг с предупреждением в лог. Если grid не задан — строится
    из метаданных файла.

    Перепроецирование между CRS не делается: если CRS файла не совпадает
    с grid.crs, будет ошибка. Это задача prep/align.py.

    Args:
        path: путь к GeoTIFF.
        grid: целевая сетка (или None — взять из файла).
        nodata: значение nodata в результирующем Layer.
        resampling: метод ресэмплинга, если нужен.

    Returns:
        Layer с data float32 формы grid.shape.

    Raises:
        FileNotFoundError: если файл не найден.
        ValueError: если CRS файла не совпадает с grid.crs.
    """
    if not path.exists():
        raise FileNotFoundError(f"Растр не найден: {path}")

    with rasterio.open(path) as src:
        src_crs_str = src.crs.to_string() if src.crs is not None else None
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        data = src.read(1).astype(np.float32)

        if grid is None:
            if src_crs_str is None:
                raise ValueError(
                    f"Растр {path.name} не имеет CRS, и grid не задан. "
                    f"Укажи CRS в конфиге или передай grid."
                )
            grid = _build_grid_from_raster(
                crs=src_crs_str,
                transform=src_transform,
                width=src.width,
                height=src.height,
            )
            result = data
            logger.info(
                "Чтение без ресэмплинга: %s | %s | %dx%d",
                path.name,
                grid.crs,
                grid.width,
                grid.height,
            )
        else:
            if src_crs_str != grid.crs:
                raise ValueError(
                    f"CRS растра ({src_crs_str}) не совпадает с grid "
                    f"({grid.crs}). Перепроецирование — задача prep/align."
                )
            same_grid = (
                src.width == grid.width
                and src.height == grid.height
                and abs(src_transform.c - grid.x_min) < _PIXEL_SIZE_TOLERANCE
                and abs((src_transform.f - src.height * abs(src_transform.e)) - grid.y_min)
                < _PIXEL_SIZE_TOLERANCE
                and abs(abs(src_transform.a) - grid.pixel_size) < _PIXEL_SIZE_TOLERANCE
            )
            if same_grid:
                result = data
                logger.info("Чтение без ресэмплинга (grid совпадает): %s", path.name)
            else:
                logger.warning(
                    "Ресэмплинг: %s из %dx%d в %dx%d",
                    path.name,
                    src.width,
                    src.height,
                    grid.width,
                    grid.height,
                )
                result = _resample_to_grid(
                    source=data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    target_grid=grid,
                    target_nodata=nodata,
                    resampling=resampling,
                )

        # Если ресэмплинга не было, но в исходнике есть nodata — заменяем
        # значения nodata исходника на наш nodata.
        if src_nodata is not None and result is data:
            result = data.copy()
            result[data == src_nodata] = nodata

    layer = Layer(
        name=path.stem,
        data=result.astype(np.float32, copy=False),
        grid=grid,
        nodata=nodata,
        source=str(path),
        license="unknown",
        unit="unitless",
    )
    return layer


def write_raster(
    layer: Layer,
    path: Path,
    compress: str = _DEFAULT_COMPRESS,
    dtype: str = _DEFAULT_DTYPE,
) -> None:
    """Сохраняет Layer в GeoTIFF.

    Args:
        layer: слой для сохранения.
        path: путь к выходному GeoTIFF. Родительская папка создаётся.
        compress: метод сжатия ("deflate", "lzw", "none").
        dtype: тип данных на диске (по умолчанию float32).

    Raises:
        ValueError: если dtype не поддерживается.
    """
    if dtype not in ("float32", "float64", "int16", "uint8"):
        raise ValueError(f"Неподдерживаемый dtype: {dtype}")

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = _build_transform(layer.grid)

    profile = {
        "driver": "GTiff",
        "height": layer.grid.height,
        "width": layer.grid.width,
        "count": 1,
        "dtype": dtype,
        "crs": CRS.from_user_input(layer.grid.crs),
        "transform": transform,
        "nodata": layer.nodata,
        "compress": compress,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(layer.data.astype(dtype, copy=False), 1)

    logger.info(
        "Записан растр: %s | %dx%d | %s | compress=%s",
        path.name,
        layer.grid.width,
        layer.grid.height,
        dtype,
        compress,
    )
