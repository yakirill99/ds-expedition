"""Тесты выравнивания слоёв."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.align import (
    METHOD_BILINEAR,
    METHOD_NEAREST,
    AlignmentReport,
    ShiftEstimate,
    apply_shift,
    check_alignment,
    estimate_shift,
    resample_layer,
    resample_layers,
)
from expds.features.grid import Grid, Layer


@pytest.fixture()
def grid() -> Grid:
    """Сетка 20x20, pixel_size=1.0, UTM 17N."""
    return Grid(
        crs="EPSG:32617",
        pixel_size=1.0,
        x_min=500_000.0,
        y_min=4_000_000.0,
        width=20,
        height=20,
    )


def _smooth_layer(grid: Grid, name: str = "smooth") -> Layer:
    """Слой с локальными формами: несколько гауссовых холмов.

    Не линейный градиент, а именно локальные формы — так кросс-корреляция
    даёт острый пик, и сдвиг определяется надёжно. Это ближе к реальному
    DTM с курганами и валами.
    """
    x = np.arange(grid.width)
    y = np.arange(grid.height)
    xx, yy = np.meshgrid(x, y)

    # Три гауссовых холма в разных углах.
    h1 = 5.0 * np.exp(-((xx - 4) ** 2 + (yy - 4) ** 2) / 8.0)
    h2 = 7.0 * np.exp(-((xx - 15) ** 2 + (yy - 6) ** 2) / 10.0)
    h3 = 4.0 * np.exp(-((xx - 10) ** 2 + (yy - 15) ** 2) / 6.0)

    data = (h1 + h2 + h3 + 0.1 * xx).astype(np.float32)
    return Layer(name=name, data=data, grid=grid, nodata=-9999.0)


class TestResampleLayer:
    """Тесты ресэмплинга одного слоя."""

    def test_same_grid_returns_same(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        result = resample_layer(layer=layer, target_grid=grid)
        assert result is layer

    def test_different_resolution(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        # Целевая сетка в 2 раза крупнее.
        coarse = Grid(
            crs=grid.crs,
            pixel_size=2.0,
            x_min=grid.x_min,
            y_min=grid.y_min,
            width=grid.width // 2,
            height=grid.height // 2,
        )
        result = resample_layer(layer=layer, target_grid=coarse)
        assert result.grid == coarse
        assert result.data.shape == coarse.shape
        assert result.data.dtype == np.float32

    def test_crs_mismatch_raises(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        other_crs = Grid(
            crs="EPSG:32618",
            pixel_size=1.0,
            x_min=500_000.0,
            y_min=4_000_000.0,
            width=20,
            height=20,
        )
        with pytest.raises(ValueError, match="CRS"):
            resample_layer(layer=layer, target_grid=other_crs)

    def test_unknown_method_raises(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        coarse = Grid(
            crs=grid.crs,
            pixel_size=2.0,
            x_min=grid.x_min,
            y_min=grid.y_min,
            width=grid.width // 2,
            height=grid.height // 2,
        )
        with pytest.raises(ValueError, match="Метод"):
            resample_layer(layer=layer, target_grid=coarse, method="unknown")

    def test_nearest_method(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        coarse = Grid(
            crs=grid.crs,
            pixel_size=2.0,
            x_min=grid.x_min,
            y_min=grid.y_min,
            width=grid.width // 2,
            height=grid.height // 2,
        )
        result = resample_layer(layer=layer, target_grid=coarse, method=METHOD_NEAREST)
        assert result.grid == coarse

    def test_preserves_metadata(self, grid: Grid) -> None:
        layer = Layer(
            name="test",
            data=np.ones(grid.shape, dtype=np.float32),
            grid=grid,
            nodata=-9999.0,
            source="source_x",
            license="CC0",
            unit="meters",
        )
        coarse = Grid(
            crs=grid.crs,
            pixel_size=2.0,
            x_min=grid.x_min,
            y_min=grid.y_min,
            width=grid.width // 2,
            height=grid.height // 2,
        )
        result = resample_layer(layer=layer, target_grid=coarse)
        assert result.source == "source_x"
        assert result.license == "CC0"
        assert result.unit == "meters"
        assert result.nodata == -9999.0


class TestResampleLayers:
    """Тесты ресэмплинга нескольких слоёв."""

    def test_multiple_layers(self, grid: Grid) -> None:
        layers = {
            "a": _smooth_layer(grid=grid, name="a"),
            "b": _smooth_layer(grid=grid, name="b"),
        }
        coarse = Grid(
            crs=grid.crs,
            pixel_size=2.0,
            x_min=grid.x_min,
            y_min=grid.y_min,
            width=grid.width // 2,
            height=grid.height // 2,
        )
        result = resample_layers(layers=layers, target_grid=coarse)
        assert set(result.keys()) == {"a", "b"}
        for layer in result.values():
            assert layer.grid == coarse

    def test_per_layer_methods(self, grid: Grid) -> None:
        layers = {
            "a": _smooth_layer(grid=grid, name="a"),
            "b": _smooth_layer(grid=grid, name="b"),
        }
        coarse = Grid(
            crs=grid.crs,
            pixel_size=2.0,
            x_min=grid.x_min,
            y_min=grid.y_min,
            width=grid.width // 2,
            height=grid.height // 2,
        )
        result = resample_layers(
            layers=layers,
            target_grid=coarse,
            methods={"a": METHOD_NEAREST, "b": METHOD_BILINEAR},
        )
        assert result["a"].grid == coarse
        assert result["b"].grid == coarse


class TestEstimateShift:
    """Тесты оценки сдвига."""

    def test_no_shift(self, grid: Grid) -> None:
        layer_a = _smooth_layer(grid=grid, name="a")
        layer_b = _smooth_layer(grid=grid, name="b")
        shift = estimate_shift(layer_a=layer_a, layer_b=layer_b, max_shift=5)
        assert shift.dx == 0
        assert shift.dy == 0
        # Идеальное совпадение => NCC ~1.0.
        assert shift.confidence > 0.9

    def test_known_shift_x(self, grid: Grid) -> None:
        layer_a = _smooth_layer(grid=grid, name="a")
        # Сдвигаем B на 3 пикселя вправо.
        data_b = np.roll(layer_a.data, 3, axis=1)
        layer_b = Layer(name="b", data=data_b, grid=grid, nodata=-9999.0)
        shift = estimate_shift(layer_a=layer_a, layer_b=layer_b, max_shift=5)
        assert shift.dx == 3
        assert shift.dy == 0
        # NCC для коррелированных слоёв высокий.
        assert shift.confidence > 0.3

    def test_known_shift_y(self, grid: Grid) -> None:
        layer_a = _smooth_layer(grid=grid, name="a")
        # Сдвигаем B на 2 пикселя вверх (axis=0).
        data_b = np.roll(layer_a.data, 2, axis=0)
        layer_b = Layer(name="b", data=data_b, grid=grid, nodata=-9999.0)
        shift = estimate_shift(layer_a=layer_a, layer_b=layer_b, max_shift=5)
        assert shift.dy == 2
        assert shift.dx == 0

    def test_known_shift_both(self, grid: Grid) -> None:
        layer_a = _smooth_layer(grid=grid, name="a")
        data_b = np.roll(np.roll(layer_a.data, 2, axis=0), -3, axis=1)
        layer_b = Layer(name="b", data=data_b, grid=grid, nodata=-9999.0)
        shift = estimate_shift(layer_a=layer_a, layer_b=layer_b, max_shift=5)
        assert shift.dx == -3
        assert shift.dy == 2

    def test_uncorrelated_layers(self, grid: Grid) -> None:
        rng = np.random.default_rng(42)
        layer_a = Layer(
            name="a",
            data=rng.normal(0, 1, grid.shape).astype(np.float32),
            grid=grid,
            nodata=-9999.0,
        )
        layer_b = Layer(
            name="b",
            data=rng.normal(0, 1, grid.shape).astype(np.float32),
            grid=grid,
            nodata=-9999.0,
        )
        shift = estimate_shift(layer_a=layer_a, layer_b=layer_b, max_shift=5)
        # NCC для шума низкий.
        assert shift.confidence < 0.3

    def test_grid_mismatch_raises(self, grid: Grid) -> None:
        layer_a = _smooth_layer(grid=grid, name="a")
        other = Grid(
            crs="EPSG:32617",
            pixel_size=2.0,
            x_min=grid.x_min,
            y_min=grid.y_min,
            width=grid.width // 2,
            height=grid.height // 2,
        )
        layer_b = Layer(
            name="b",
            data=np.ones(other.shape, dtype=np.float32),
            grid=other,
            nodata=-9999.0,
        )
        with pytest.raises(ValueError, match="Сетки"):
            estimate_shift(layer_a=layer_a, layer_b=layer_b)

    def test_magnitude(self) -> None:
        shift = ShiftEstimate(dx=3, dy=4, confidence=5.0)
        assert shift.magnitude == pytest.approx(5.0)


class TestApplyShift:
    """Тесты применения сдвига."""

    def test_zero_shift_returns_same(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        shift = ShiftEstimate(dx=0, dy=0, confidence=10.0)
        result = apply_shift(layer=layer, shift=shift)
        assert result is layer

    def test_shift_x(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        shift = ShiftEstimate(dx=3, dy=0, confidence=10.0)
        result = apply_shift(layer=layer, shift=shift)
        # Левые 3 столбца — nodata.
        assert (result.data[:, :3] == -9999.0).all()
        # Значения сместились вправо.
        assert result.data[0, 3] == layer.data[0, 0]

    def test_shift_y(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        shift = ShiftEstimate(dx=0, dy=2, confidence=10.0)
        result = apply_shift(layer=layer, shift=shift)
        # Верхние 2 строки — nodata.
        assert (result.data[:2, :] == -9999.0).all()
        # Значения сместились вниз.
        assert result.data[2, 0] == layer.data[0, 0]

    def test_negative_shift_x(self, grid: Grid) -> None:
        layer = _smooth_layer(grid=grid)
        shift = ShiftEstimate(dx=-3, dy=0, confidence=10.0)
        result = apply_shift(layer=layer, shift=shift)
        # Правые 3 столбца — nodata.
        assert (result.data[:, -3:] == -9999.0).all()


class TestCheckAlignment:
    """Тесты проверки совмещения."""

    def test_all_aligned(self, grid: Grid) -> None:
        layers = {
            "dtm": _smooth_layer(grid=grid, name="dtm"),
            "slope": _smooth_layer(grid=grid, name="slope"),
            "tpi": _smooth_layer(grid=grid, name="tpi"),
        }
        report = check_alignment(layers=layers, reference_name="dtm")
        assert isinstance(report, AlignmentReport)
        assert report.reference == "dtm"
        assert len(report.shifts) == 2
        for shift in report.shifts.values():
            assert shift.dx == 0
            assert shift.dy == 0

    def test_detects_shift(self, grid: Grid) -> None:
        reference = _smooth_layer(grid=grid, name="dtm")
        shifted_data = np.roll(reference.data, 2, axis=1)
        shifted = Layer(
            name="shifted",
            data=shifted_data,
            grid=grid,
            nodata=-9999.0,
        )
        report = check_alignment(
            layers={"dtm": reference, "shifted": shifted},
            reference_name="dtm",
            confidence_threshold=0.3,
        )
        assert report.shifts["shifted"].dx == 2
        assert report.shifts["shifted"].confidence > 0.3

    def test_missing_reference_raises(self, grid: Grid) -> None:
        layers = {"a": _smooth_layer(grid=grid, name="a")}
        with pytest.raises(KeyError, match="не найден"):
            check_alignment(layers=layers, reference_name="nonexistent")

    def test_low_confidence_treated_as_zero(self, grid: Grid) -> None:
        rng = np.random.default_rng(42)
        reference = Layer(
            name="ref",
            data=rng.normal(0, 1, grid.shape).astype(np.float32),
            grid=grid,
            nodata=-9999.0,
        )
        other = Layer(
            name="other",
            data=rng.normal(0, 1, grid.shape).astype(np.float32),
            grid=grid,
            nodata=-9999.0,
        )
        report = check_alignment(
            layers={"ref": reference, "other": other},
            reference_name="ref",
            confidence_threshold=0.3,
        )
        # Низкий NCC => сдвиг обнулён.
        assert report.shifts["other"].dx == 0
        assert report.shifts["other"].dy == 0
