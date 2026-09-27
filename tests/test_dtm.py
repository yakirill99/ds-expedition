"""Тесты построения DTM."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.config import DtmConfig
from expds.features.dtm import (
    _bin_points,
    _fill_idw,
    _fill_nearest,
    _filter_large_holes,
    build_dtm,
    detect_artifacts,
)
from expds.features.grid import Grid


@pytest.fixture()
def grid() -> Grid:
    """Сетка 20x20 с pixel_size=1.0, покрывающая [0..20] x [0..20]."""
    return Grid(
        crs="EPSG:32617",
        pixel_size=1.0,
        x_min=0.0,
        y_min=0.0,
        width=20,
        height=20,
    )


@pytest.fixture()
def config() -> DtmConfig:
    """Стандартный конфиг DTM для тестов."""
    return DtmConfig(
        statistic="mean",
        fill_method="idw",
        idw_k=4,
        idw_power=2.0,
        max_hole_pixels=100,
        artifact_window=3,
        artifact_threshold=2.0,
    )


def _make_plane(
    grid: Grid,
    n: int = 5000,
    seed: int = 42,
    slope: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Плоскость Z = slope * X + шум."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(grid.x_min, grid.x_max, n)
    y = rng.uniform(grid.y_min, grid.y_max, n)
    z = slope * x + rng.normal(0, 0.01, n)
    return x, y, z


class TestBinPoints:
    """Тесты биннинга."""

    def test_mean_on_plane(self, grid: Grid) -> None:
        x, y, z = _make_plane(grid=grid, slope=0.0)
        values, count = _bin_points(x=x, y=y, z=z, grid=grid, statistic="mean")
        assert values.shape == grid.shape
        assert count.shape == grid.shape
        # Точки есть почти везде, значения близки к 0.
        valid = ~np.isnan(values)
        assert valid.mean() > 0.95
        assert np.nanmean(values) == pytest.approx(0.0, abs=0.05)

    def test_min_on_plane(self, grid: Grid) -> None:
        x, y, z = _make_plane(grid=grid, slope=0.0)
        values, _ = _bin_points(x=x, y=y, z=z, grid=grid, statistic="min")
        valid = ~np.isnan(values)
        # min должен быть <= mean.
        assert np.nanmean(values[valid]) < 0.0

    def test_median_on_plane(self, grid: Grid) -> None:
        x, y, z = _make_plane(grid=grid, slope=0.0)
        values, _ = _bin_points(x=x, y=y, z=z, grid=grid, statistic="median")
        valid = ~np.isnan(values)
        assert np.nanmean(values[valid]) == pytest.approx(0.0, abs=0.05)

    def test_empty_grid(self, grid: Grid) -> None:
        x = np.array([100.0, 200.0])  # вне сетки
        y = np.array([100.0, 200.0])
        z = np.array([0.0, 0.0])
        values, count = _bin_points(x=x, y=y, z=z, grid=grid, statistic="mean")
        assert count.sum() == 0
        assert np.isnan(values).all()

    def test_unknown_statistic_raises(self, grid: Grid) -> None:
        x, y, z = _make_plane(grid=grid)
        with pytest.raises(ValueError, match="Статистика"):
            _bin_points(x=x, y=y, z=z, grid=grid, statistic="unknown")


class TestFillHoles:
    """Тесты заполнения дыр."""

    def test_nearest_fills_all(self) -> None:
        values = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
        mask = np.isnan(values)
        filled = _fill_nearest(values=values, mask=mask)
        assert not np.isnan(filled).any()
        assert (filled == 1.0).all()

    def test_idw_fills_all(self, grid: Grid) -> None:
        values = np.full(grid.shape, np.nan, dtype=np.float32)
        values[10, 10] = 5.0
        values[10, 11] = 6.0
        mask = np.isnan(values)
        filled = _fill_idw(values=values, mask=mask, grid=grid, k=2, power=2.0)
        assert not np.isnan(filled).any()
        # Значения должны быть между 5 и 6.
        assert filled.min() >= 5.0
        assert filled.max() <= 6.0

    def test_no_holes_returns_same(self) -> None:
        values = np.ones((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=bool)
        filled = _fill_nearest(values=values, mask=mask)
        np.testing.assert_allclose(filled, values)


class TestLargeHoles:
    """Тесты фильтра больших дыр."""

    def test_small_hole_kept(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 2:4] = True  # 4 пикселя
        big = _filter_large_holes(mask=mask, max_hole_pixels=10)
        assert not big.any()

    def test_large_hole_flagged(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True  # 36 пикселей
        big = _filter_large_holes(mask=mask, max_hole_pixels=10)
        assert big.sum() == 36

    def test_no_holes(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        big = _filter_large_holes(mask=mask, max_hole_pixels=10)
        assert not big.any()


class TestArtifacts:
    """Тесты детектора артефактов."""

    def test_smooth_surface_no_artifacts(self) -> None:
        dtm = np.linspace(0, 10, 400).reshape(20, 20)
        artifacts = detect_artifacts(dtm=dtm, window=3, threshold=1.0)
        assert artifacts.sum() == 0

    def test_spike_detected(self) -> None:
        dtm = np.zeros((20, 20), dtype=np.float32)
        dtm[10, 10] = 100.0  # выброс
        artifacts = detect_artifacts(dtm=dtm, window=3, threshold=1.0)
        assert artifacts[10, 10]


class TestBuildDtm:
    """Тесты основного входа."""

    def test_dtm_on_plane(self, grid: Grid, config: DtmConfig) -> None:
        x, y, z = _make_plane(grid=grid, slope=0.0)
        result = build_dtm(x=x, y=y, z=z, grid=grid, config=config)
        assert result.dtm.grid == grid
        assert result.dtm.data.shape == grid.shape
        assert result.hole_mask.data.shape == grid.shape
        assert result.count.data.shape == grid.shape
        # Большинство ячеек заполнено.
        valid = result.dtm.data != result.dtm.nodata
        assert valid.mean() > 0.9

    def test_dtm_preserves_slope(self, grid: Grid, config: DtmConfig) -> None:
        slope = 0.2
        x, y, z = _make_plane(grid=grid, slope=slope)
        result = build_dtm(x=x, y=y, z=z, grid=grid, config=config)

        data = result.dtm.data
        valid = data != result.dtm.nodata

        # Средний Z в первом и последнем столбце (только валидные ячейки).
        col_first = data[:, 0][valid[:, 0]].mean()
        col_last = data[:, -1][valid[:, -1]].mean()

        # Расстояние между центрами крайних столбцов.
        distance = (grid.width - 1) * grid.pixel_size
        observed_slope = (col_last - col_first) / distance

        assert observed_slope == pytest.approx(slope, abs=0.02)

    def test_count_layer(self, grid: Grid, config: DtmConfig) -> None:
        x, y, z = _make_plane(grid=grid, n=2000)
        result = build_dtm(x=x, y=y, z=z, grid=grid, config=config)
        assert result.count.data.sum() > 0
        assert result.count.data.max() > 0

    def test_hole_mask_marks_empty(self, grid: Grid, config: DtmConfig) -> None:
        # Точки только в левой половине.
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, 500)
        y = rng.uniform(0, 20, 500)
        z = rng.normal(0, 0.01, 500)
        result = build_dtm(x=x, y=y, z=z, grid=grid, config=config)
        # Правая половина должна быть помечена как дыра.
        assert result.hole_mask.data[:, 10:].mean() > 0.9
        assert result.hole_mask.data[:, :5].mean() < 0.1

    def test_mismatched_lengths(self, grid: Grid, config: DtmConfig) -> None:
        x = np.zeros(10)
        y = np.zeros(5)
        z = np.zeros(10)
        with pytest.raises(ValueError, match="Длины"):
            build_dtm(x=x, y=y, z=z, grid=grid, config=config)

    def test_empty_points(self, grid: Grid, config: DtmConfig) -> None:
        x = np.zeros(0)
        y = np.zeros(0)
        z = np.zeros(0)
        with pytest.raises(ValueError, match="Пустой"):
            build_dtm(x=x, y=y, z=z, grid=grid, config=config)

    def test_unknown_fill_method(self, grid: Grid, config: DtmConfig) -> None:
        bad = DtmConfig(
            statistic="mean",
            fill_method="unknown",
            idw_k=4,
            idw_power=2.0,
            max_hole_pixels=100,
            artifact_window=3,
            artifact_threshold=2.0,
        )
        x, y, z = _make_plane(grid=grid)
        with pytest.raises(ValueError, match="Метод"):
            build_dtm(x=x, y=y, z=z, grid=grid, config=bad)
