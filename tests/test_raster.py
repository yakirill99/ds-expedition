"""Тесты чтения/записи GeoTIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from expds.data.raster import (
    read_raster,
    read_raster_meta,
    write_raster,
)
from expds.features.grid import Grid, Layer


@pytest.fixture()
def grid() -> Grid:
    """Сетка 8x6, pixel_size=1.0, CRS UTM 17N."""
    return Grid(
        crs="EPSG:32617",
        pixel_size=1.0,
        x_min=500_000.0,
        y_min=4_000_000.0,
        width=8,
        height=6,
    )


@pytest.fixture()
def layer(grid: Grid) -> Layer:
    """Слой с предсказуемыми значениями."""
    data = np.arange(grid.height * grid.width, dtype=np.float32).reshape(grid.shape)
    return Layer(
        name="test",
        data=data,
        grid=grid,
        nodata=-9999.0,
        source="synthetic",
        license="CC0",
        unit="meters",
    )


class TestRoundTrip:
    """Layer -> GeoTIFF -> Layer без потерь."""

    def test_roundtrip_values(self, layer: Layer, tmp_path: Path) -> None:
        path = tmp_path / "test.tif"
        write_raster(layer=layer, path=path)
        restored = read_raster(path=path, grid=layer.grid)
        np.testing.assert_allclose(restored.data, layer.data)
        assert restored.grid == layer.grid

    def test_roundtrip_grid_from_file(self, layer: Layer, tmp_path: Path) -> None:
        path = tmp_path / "test.tif"
        write_raster(layer=layer, path=path)
        restored = read_raster(path=path, grid=None)
        assert restored.grid.crs == layer.grid.crs
        assert restored.grid.width == layer.grid.width
        assert restored.grid.height == layer.grid.height
        assert restored.grid.pixel_size == pytest.approx(layer.grid.pixel_size)
        assert restored.grid.x_min == pytest.approx(layer.grid.x_min)
        assert restored.grid.y_min == pytest.approx(layer.grid.y_min)

    def test_roundtrip_nodata(self, grid: Grid, tmp_path: Path) -> None:
        data = np.ones(grid.shape, dtype=np.float32)
        data[0, 0] = -9999.0
        layer = Layer(name="x", data=data, grid=grid, nodata=-9999.0)
        path = tmp_path / "x.tif"
        write_raster(layer=layer, path=path)
        restored = read_raster(path=path, grid=grid)
        assert restored.data[0, 0] == pytest.approx(-9999.0)
        assert restored.data[1, 1] == pytest.approx(1.0)


class TestReadMeta:
    """read_raster_meta не читает данные."""

    def test_meta_fields(self, layer: Layer, tmp_path: Path) -> None:
        path = tmp_path / "test.tif"
        write_raster(layer=layer, path=path)
        meta = read_raster_meta(path=path)
        assert meta.crs == "EPSG:32617"
        assert meta.width == layer.grid.width
        assert meta.height == layer.grid.height
        assert meta.pixel_size_x == pytest.approx(layer.grid.pixel_size)
        assert meta.dtype == "float32"
        assert meta.nodata == pytest.approx(-9999.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_raster_meta(path=tmp_path / "nope.tif")


class TestErrors:
    """Ошибочные сценарии."""

    def test_missing_file_raises(self, tmp_path: Path, grid: Grid) -> None:
        with pytest.raises(FileNotFoundError):
            read_raster(path=tmp_path / "nope.tif", grid=grid)

    def test_crs_mismatch_raises(self, layer: Layer, tmp_path: Path) -> None:
        path = tmp_path / "test.tif"
        write_raster(layer=layer, path=path)
        other_grid = Grid(
            crs="EPSG:32618",  # другой UTM zone
            pixel_size=1.0,
            x_min=500_000.0,
            y_min=4_000_000.0,
            width=8,
            height=6,
        )
        with pytest.raises(ValueError, match="CRS"):
            read_raster(path=path, grid=other_grid)

    def test_bad_dtype_raises(self, layer: Layer, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="dtype"):
            write_raster(
                layer=layer,
                path=tmp_path / "x.tif",
                dtype="int32",
            )


class TestResample:
    """Ресэмплинг под другую сетку."""

    def test_resample_downscales(self, layer: Layer, tmp_path: Path) -> None:
        path = tmp_path / "test.tif"
        write_raster(layer=layer, path=path)

        # Целевая сетка в 2 раза крупнее по пикселю.
        coarse = Grid(
            crs=layer.grid.crs,
            pixel_size=2.0,
            x_min=layer.grid.x_min,
            y_min=layer.grid.y_min,
            width=layer.grid.width // 2,
            height=layer.grid.height // 2,
        )
        resampled = read_raster(path=path, grid=coarse)
        assert resampled.grid.shape == coarse.shape
        assert resampled.data.dtype == np.float32
