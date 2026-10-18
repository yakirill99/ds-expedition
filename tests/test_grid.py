"""Тесты контрактов Grid, Layer, Sample, DataSpec."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.grid import DataSpec, Grid, Layer, Sample, stack_layers


@pytest.fixture()
def grid() -> Grid:
    """Простая сетка 10x20 пикселей, pixel_size=1.0."""
    return Grid(
        crs="EPSG:32617",
        pixel_size=1.0,
        x_min=0.0,
        y_min=0.0,
        width=20,
        height=10,
    )


class TestGrid:
    """Тесты Grid."""

    def test_shape_and_bounds(self, grid: Grid) -> None:
        assert grid.shape == (10, 20)
        assert grid.bounds == (0.0, 0.0, 20.0, 10.0)
        assert grid.x_max == 20.0
        assert grid.y_max == 10.0

    def test_center_of_first_pixel(self, grid: Grid) -> None:
        # Пиксель (0, 0) — центр ячейки в (0.5, 0.5).
        x, y = grid.transform_to_world(col=0, row=0)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)

    def test_roundtrip_pixel_world(self, grid: Grid) -> None:
        cols = np.array([0.0, 5.5, 19.0])
        rows = np.array([0.0, 3.5, 9.0])
        x, y = grid.transform_to_world(col=cols, row=rows)
        col_back, row_back = grid.transform_to_pixel(x=x, y=y)
        np.testing.assert_allclose(col_back, cols, atol=1e-9)
        np.testing.assert_allclose(row_back, rows, atol=1e-9)

    def test_from_bounds_rounds_up(self) -> None:
        # 10.5 / 1.0 -> 11 пикселей.
        g = Grid.from_bounds(
            crs="EPSG:32617",
            pixel_size=1.0,
            bounds=(0.0, 0.0, 10.5, 7.2),
        )
        assert g.width == 11
        assert g.height == 8

    def test_invalid_pixel_size(self) -> None:
        with pytest.raises(ValueError, match="pixel_size"):
            Grid(
                crs="EPSG:32617",
                pixel_size=0.0,
                x_min=0.0,
                y_min=0.0,
                width=10,
                height=10,
            )

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="width/height"):
            Grid(
                crs="EPSG:32617",
                pixel_size=1.0,
                x_min=0.0,
                y_min=0.0,
                width=0,
                height=10,
            )

    def test_contains_point(self, grid: Grid) -> None:
        assert grid.contains_point(5.0, 5.0)
        assert not grid.contains_point(-1.0, 5.0)
        assert not grid.contains_point(21.0, 5.0)


class TestLayer:
    """Тесты Layer."""

    def test_valid_layer(self, grid: Grid) -> None:
        data = np.zeros(grid.shape, dtype=np.float32)
        layer = Layer(name="dtm", data=data, grid=grid)
        assert layer.name == "dtm"
        assert layer.valid_mask.all()

    def test_nodata_detected(self, grid: Grid) -> None:
        data = np.full(grid.shape, -9999.0, dtype=np.float32)
        data[0, 0] = 10.0
        layer = Layer(name="dtm", data=data, grid=grid, nodata=-9999.0)
        assert layer.valid_mask.sum() == 1
        assert layer.valid_mask[0, 0]

    def test_wrong_shape_raises(self, grid: Grid) -> None:
        data = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            Layer(name="dtm", data=data, grid=grid)

    def test_wrong_dtype_raises(self, grid: Grid) -> None:
        data = np.zeros(grid.shape, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            Layer(name="dtm", data=data, grid=grid)

    def test_wrong_ndim_raises(self, grid: Grid) -> None:
        data = np.zeros((1, *grid.shape), dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            Layer(name="dtm", data=data, grid=grid)


class TestSample:
    """Тесты Sample."""

    def test_valid_sample(self, grid: Grid) -> None:
        tensor = np.zeros((3, *grid.shape), dtype=np.float32)
        sample = Sample(
            tensor=tensor,
            channel_names=("dtm", "hillshade", "slope"),
            channel_mask=np.array([True, True, True]),
            grid=grid,
        )
        assert sample.n_channels == 3
        assert sample.n_available_channels == 3

    def test_channel_names_mismatch(self, grid: Grid) -> None:
        tensor = np.zeros((3, *grid.shape), dtype=np.float32)
        with pytest.raises(ValueError, match="channel_names"):
            Sample(
                tensor=tensor,
                channel_names=("dtm", "hillshade"),
                channel_mask=np.array([True, True, True]),
                grid=grid,
            )

    def test_channel_mask_mismatch(self, grid: Grid) -> None:
        tensor = np.zeros((3, *grid.shape), dtype=np.float32)
        with pytest.raises(ValueError, match="channel_mask"):
            Sample(
                tensor=tensor,
                channel_names=("a", "b", "c"),
                channel_mask=np.array([True, True]),
                grid=grid,
            )


class TestDataSpec:
    """Тесты DataSpec."""

    def test_valid_spec(self, grid: Grid) -> None:
        spec = DataSpec(
            grid=grid,
            channel_names=("dtm", "hillshade", "slope"),
        )
        assert spec.channel_index("hillshade") == 1

    def test_duplicate_channels_raise(self, grid: Grid) -> None:
        with pytest.raises(ValueError, match="дублирующиеся"):
            DataSpec(grid=grid, channel_names=("dtm", "dtm"))

    def test_unknown_channel_raises(self, grid: Grid) -> None:
        spec = DataSpec(grid=grid, channel_names=("dtm",))
        with pytest.raises(KeyError, match="hillshade"):
            spec.channel_index("hillshade")


class TestStackLayers:
    """Тесты сборки Sample из слоёв."""

    def test_stack_full(self, grid: Grid) -> None:
        spec = DataSpec(grid=grid, channel_names=("dtm", "hillshade"))
        dtm = Layer(
            name="dtm",
            data=np.ones(grid.shape, dtype=np.float32),
            grid=grid,
        )
        hs = Layer(
            name="hillshade",
            data=np.full(grid.shape, 2.0, dtype=np.float32),
            grid=grid,
        )
        sample = stack_layers(layers=[dtm, hs], spec=spec)
        assert sample.n_available_channels == 2
        np.testing.assert_allclose(sample.tensor[0], 1.0)
        np.testing.assert_allclose(sample.tensor[1], 2.0)

    def test_stack_missing_channel_filled_with_zeros(self, grid: Grid) -> None:
        spec = DataSpec(grid=grid, channel_names=("dtm", "hillshade"))
        dtm = Layer(
            name="dtm",
            data=np.ones(grid.shape, dtype=np.float32),
            grid=grid,
        )
        sample = stack_layers(layers=[dtm], spec=spec)
        assert sample.n_available_channels == 1
        assert sample.channel_mask[0]
        assert not sample.channel_mask[1]
        np.testing.assert_allclose(sample.tensor[1], 0.0)

    def test_stack_extra_layer_warns(self, grid: Grid, caplog) -> None:
        import logging

        spec = DataSpec(grid=grid, channel_names=("dtm",))
        dtm = Layer(
            name="dtm",
            data=np.ones(grid.shape, dtype=np.float32),
            grid=grid,
        )
        extra = Layer(
            name="unknown",
            data=np.ones(grid.shape, dtype=np.float32),
            grid=grid,
        )
        with caplog.at_level(logging.WARNING):
            stack_layers(layers=[dtm, extra], spec=spec)
        assert "Лишние слои" in caplog.text

    def test_stack_nodata_becomes_zero(self, grid: Grid) -> None:
        spec = DataSpec(grid=grid, channel_names=("dtm",))
        data = np.full(grid.shape, -9999.0, dtype=np.float32)
        data[0, 0] = 42.0
        dtm = Layer(name="dtm", data=data, grid=grid, nodata=-9999.0)
        sample = stack_layers(layers=[dtm], spec=spec)
        assert sample.tensor[0, 0, 0] == pytest.approx(42.0)
        assert sample.tensor[0, 1, 1] == pytest.approx(0.0)


class TestGridToAffine:
    """Тесты grid_to_affine."""

    def test_known_grid(self, grid: Grid) -> None:
        """Проверяет transform для известного Grid."""
        from expds.features.grid import grid_to_affine

        affine = grid_to_affine(grid)
        # pixel_size по X и Y.
        assert affine.a == pytest.approx(grid.pixel_size)
        assert affine.e == pytest.approx(-grid.pixel_size)
        # Верхний левый угол.
        assert affine.c == pytest.approx(grid.x_min)
        assert affine.f == pytest.approx(grid.y_max)
        # Нет поворота.
        assert affine.b == 0.0
        assert affine.d == 0.0

    def test_matches_north_up_convention(self, grid: Grid) -> None:
        """grid_to_affine даёт transform для north-up (rasterio-конвенция).

        Grid использует row 0 = юг, rasterio — row 0 = север.
        Совпадение по Y достигается через переворот:
            y_grid = y_min + (row + 0.5) * ps
            y_rasterio = y_max - (row + 0.5) * ps
        где row_rasterio = height - 1 - row_grid.
        """
        from expds.features.grid import grid_to_affine

        affine = grid_to_affine(grid)
        col, row_grid = 5, 3

        # Координата в Grid-конвенции.
        x_grid, y_grid = grid.transform_to_world(col=col, row=row_grid)

        # Тот же пиксель в rasterio-конвенции (row 0 = север).
        row_rasterio = grid.height - 1 - row_grid

        x_rasterio = affine.c + (col + 0.5) * affine.a
        y_rasterio = affine.f + (row_rasterio + 0.5) * affine.e

        assert x_rasterio == pytest.approx(x_grid)
        assert y_rasterio == pytest.approx(y_grid)
