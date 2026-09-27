"""Регрессия: ориентация Y на границе Grid <-> GeoTIFF.

Контракт Grid: row 0 = y_min (юг). GeoTIFF north-up: строка 0 файла = y_max (север).
raster.py обязан переворачивать массив при записи и чтении.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from expds.data.raster import read_raster, write_raster
from expds.features.grid import Grid, Layer

_GRID = Grid(crs="EPSG:32617", pixel_size=1.0, x_min=0.0, y_min=0.0, width=4, height=3)


def _south_row_ones() -> np.ndarray:
    """Единицы в row 0 по конвенции Grid (южная строка)."""
    data = np.zeros(_GRID.shape, dtype=np.float32)
    data[0, :] = 1.0
    return data


def test_write_row0_lands_on_south(tmp_path: Path) -> None:
    """Row 0 массива Grid после записи лежит на y центра row 0 из Grid."""
    path = tmp_path / "t.tif"
    write_raster(layer=Layer(name="t", data=_south_row_ones(), grid=_GRID), path=path)

    with rasterio.open(path) as src:
        arr = src.read(1)
        rows = np.where(arr[:, 0] == 1.0)[0]
        assert len(rows) == 1
        _, y_file = src.xy(int(rows[0]), 0)

    _, y_grid = _GRID.transform_to_world(0, 0)
    assert y_file == float(y_grid)


def test_read_external_north_up(tmp_path: Path) -> None:
    """Внешний north-up tif: северная строка файла -> последняя строка Grid."""
    path = tmp_path / "ext.tif"
    arr = np.zeros(_GRID.shape, dtype=np.float32)
    arr[0, :] = 1.0  # строка 0 файла = север
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_GRID.height,
        width=_GRID.width,
        count=1,
        dtype="float32",
        crs=_GRID.crs,
        transform=from_origin(_GRID.x_min, _GRID.y_max, _GRID.pixel_size, _GRID.pixel_size),
    ) as dst:
        dst.write(arr, 1)

    layer = read_raster(path=path, grid=_GRID)
    assert layer.data[-1].tolist() == [1.0] * _GRID.width
    assert float(layer.data[0].sum()) == 0.0


def test_roundtrip_preserves_array(tmp_path: Path) -> None:
    """write -> read возвращает тот же массив."""
    path = tmp_path / "rt.tif"
    data = np.arange(_GRID.height * _GRID.width, dtype=np.float32).reshape(_GRID.shape)
    write_raster(layer=Layer(name="rt", data=data, grid=_GRID), path=path)
    assert np.array_equal(read_raster(path=path, grid=_GRID).data, data)
