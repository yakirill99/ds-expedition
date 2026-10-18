"""Тесты спектральных индексов."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.config import SpectralConfig
from expds.features.grid import Grid
from expds.features.spectral import (
    INDEX_ARI,
    INDEX_GNDVI,
    INDEX_NDVI,
    INDEX_NDWI,
    INDEX_SAVI,
    INDEX_SR,
    build_spectral_indices,
    compute_ari,
    compute_gndvi,
    compute_ndvi,
    compute_ndwi,
    compute_savi,
    compute_sr,
)

# Локальные копии лимитов: если поменяем лимиты в модуле — тесты явно упадут.
_SR_MAX_EXPECTED = 20.0
_ARI_ABS_MAX_EXPECTED = 10.0
_SAVI_ABS_MAX_EXPECTED = 1.5


@pytest.fixture()
def grid() -> Grid:
    return Grid(
        crs="EPSG:32617",
        pixel_size=1.0,
        x_min=0.0,
        y_min=0.0,
        width=10,
        height=10,
    )


@pytest.fixture()
def config() -> SpectralConfig:
    return SpectralConfig(
        enabled=(
            INDEX_NDVI,
            INDEX_SR,
            INDEX_ARI,
            INDEX_GNDVI,
            INDEX_NDWI,
            INDEX_SAVI,
        ),
        red_channel=0,
        nir_channel=3,
        green_channel=1,
        blue_channel=2,
        nodata=-9999.0,
        channel_std_min=0.01,
    )


def _noisy_channel(
    rng: np.random.Generator,
    grid: Grid,
    mean: float,
    std: float = 5.0,
) -> np.ndarray:
    """Канал с шумом вокруг mean: std > channel_std_min.

    Константный канал фильтр каналов (channel_std_min) отбросит как
    alpha-маску. Нам нужны каналы с вариацией, иначе все индексы
    окажутся в skipped.
    """
    return (mean + rng.normal(0.0, std, grid.shape)).astype(np.float32)


def _make_channels(grid: Grid) -> dict[int, np.ndarray]:
    """Синтетические каналы с вариацией (std > channel_std_min).

    Средние значения: red≈100, green≈150, blue≈50, nir≈200.
    """
    rng = np.random.default_rng(42)
    return {
        0: _noisy_channel(rng, grid, 100.0),  # red
        1: _noisy_channel(rng, grid, 150.0),  # green
        2: _noisy_channel(rng, grid, 50.0),  # blue
        3: _noisy_channel(rng, grid, 200.0),  # nir
    }


class TestNdvi:
    """Тесты NDVI."""

    def test_known_value(self, grid: Grid) -> None:
        red = np.full(grid.shape, 100.0, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        ndvi = compute_ndvi(red=red, nir=nir, valid=valid, nodata=-9999.0)
        np.testing.assert_allclose(ndvi, 1.0 / 3.0, atol=1e-5)

    def test_range(self, grid: Grid) -> None:
        rng = np.random.default_rng(42)
        red = rng.uniform(1, 255, grid.shape).astype(np.float32)
        nir = rng.uniform(1, 255, grid.shape).astype(np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        ndvi = compute_ndvi(red=red, nir=nir, valid=valid, nodata=-9999.0)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_nodata_applied(self, grid: Grid) -> None:
        red = np.full(grid.shape, 100.0, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        valid[0, 0] = False
        ndvi = compute_ndvi(red=red, nir=nir, valid=valid, nodata=-9999.0)
        assert ndvi[0, 0] == pytest.approx(-9999.0)
        assert ndvi[0, 1] == pytest.approx(1.0 / 3.0, abs=1e-5)


class TestSr:
    """Тесты SR."""

    def test_known_value(self, grid: Grid) -> None:
        red = np.full(grid.shape, 100.0, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        sr = compute_sr(red=red, nir=nir, valid=valid, nodata=-9999.0)
        np.testing.assert_allclose(sr, 2.0, atol=1e-5)

    def test_non_negative(self, grid: Grid) -> None:
        rng = np.random.default_rng(42)
        red = rng.uniform(1, 255, grid.shape).astype(np.float32)
        nir = rng.uniform(1, 255, grid.shape).astype(np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        sr = compute_sr(red=red, nir=nir, valid=valid, nodata=-9999.0)
        assert sr.min() >= 0.0

    def test_clip_max(self, grid: Grid) -> None:
        red = np.full(grid.shape, 1.0, dtype=np.float32)
        nir = np.full(grid.shape, 255.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        sr = compute_sr(red=red, nir=nir, valid=valid, nodata=-9999.0)
        np.testing.assert_allclose(sr, _SR_MAX_EXPECTED, atol=1e-5)

    def test_zero_red_no_inf(self, grid: Grid) -> None:
        red = np.zeros(grid.shape, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        sr = compute_sr(red=red, nir=nir, valid=valid, nodata=-9999.0)
        assert np.isfinite(sr).all()


class TestAri:
    """Тесты ARI."""

    def test_known_value(self, grid: Grid) -> None:
        green = np.full(grid.shape, 100.0, dtype=np.float32)
        red = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        ari = compute_ari(green=green, red=red, valid=valid, nodata=-9999.0)
        np.testing.assert_allclose(ari, 0.005, atol=1e-6)

    def test_clip_extreme(self, grid: Grid) -> None:
        green = np.full(grid.shape, 0.001, dtype=np.float32)
        red = np.full(grid.shape, 255.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        ari = compute_ari(green=green, red=red, valid=valid, nodata=-9999.0)
        assert ari.max() <= _ARI_ABS_MAX_EXPECTED + 1e-5
        assert ari.min() >= -_ARI_ABS_MAX_EXPECTED - 1e-5

    def test_zero_green_no_inf(self, grid: Grid) -> None:
        green = np.zeros(grid.shape, dtype=np.float32)
        red = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        ari = compute_ari(green=green, red=red, valid=valid, nodata=-9999.0)
        assert np.isfinite(ari).all()


class TestGndvi:
    """Тесты GNDVI."""

    def test_known_value(self, grid: Grid) -> None:
        green = np.full(grid.shape, 100.0, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        gndvi = compute_gndvi(green=green, nir=nir, valid=valid, nodata=-9999.0)
        np.testing.assert_allclose(gndvi, 1.0 / 3.0, atol=1e-5)


class TestNdwi:
    """Тесты NDWI."""

    def test_known_value(self, grid: Grid) -> None:
        green = np.full(grid.shape, 100.0, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        ndwi = compute_ndwi(green=green, nir=nir, valid=valid, nodata=-9999.0)
        np.testing.assert_allclose(ndwi, -1.0 / 3.0, atol=1e-5)


class TestSavi:
    """Тесты SAVI."""

    def test_known_value(self, grid: Grid) -> None:
        # L=0.5, Red=100, NIR=200:
        # SAVI = (200-100)/(200+100+0.5) * 1.5 = 100/300.5 * 1.5 ≈ 0.4992
        red = np.full(grid.shape, 100.0, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        savi = compute_savi(red=red, nir=nir, valid=valid, nodata=-9999.0)
        expected = (200.0 - 100.0) / (200.0 + 100.0 + 0.5) * 1.5
        np.testing.assert_allclose(savi, expected, atol=1e-5)

    def test_zero_red_no_inf(self, grid: Grid) -> None:
        red = np.zeros(grid.shape, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        savi = compute_savi(red=red, nir=nir, valid=valid, nodata=-9999.0)
        assert np.isfinite(savi).all()

    def test_clip_extreme(self, grid: Grid) -> None:
        # Огромная разность -> должен обрезаться до _SAVI_MAX_EXPECTED.
        red = np.full(grid.shape, 0.0, dtype=np.float32)
        nir = np.full(grid.shape, 255.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        savi = compute_savi(red=red, nir=nir, valid=valid, nodata=-9999.0)
        assert savi.max() <= _SAVI_ABS_MAX_EXPECTED + 1e-5
        assert savi.min() >= -_SAVI_ABS_MAX_EXPECTED - 1e-5

    def test_nodata_applied(self, grid: Grid) -> None:
        red = np.full(grid.shape, 100.0, dtype=np.float32)
        nir = np.full(grid.shape, 200.0, dtype=np.float32)
        valid = np.ones(grid.shape, dtype=bool)
        valid[0, 0] = False
        savi = compute_savi(red=red, nir=nir, valid=valid, nodata=-9999.0)
        assert savi[0, 0] == pytest.approx(-9999.0)


class TestBuildSpectral:
    """Тесты сборки индексов."""

    def test_all_indices(self, grid: Grid, config: SpectralConfig) -> None:
        channels = _make_channels(grid)
        valid = np.ones(grid.shape, dtype=bool)
        result = build_spectral_indices(
            channels=channels,
            valid_mask=valid,
            grid=grid,
            config=config,
        )
        assert set(result.layers.keys()) == {
            INDEX_NDVI,
            INDEX_SR,
            INDEX_ARI,
            INDEX_GNDVI,
            INDEX_NDWI,
            INDEX_SAVI,
        }
        assert result.skipped == ()

    def test_no_nir_skips_ndvi(self, grid: Grid) -> None:
        config = SpectralConfig(
            enabled=(INDEX_NDVI, INDEX_SR, INDEX_SAVI),
            red_channel=0,
            nir_channel=None,
            green_channel=1,
            blue_channel=2,
            nodata=-9999.0,
            channel_std_min=0.01,
        )
        channels = _make_channels(grid)
        valid = np.ones(grid.shape, dtype=bool)
        result = build_spectral_indices(
            channels=channels,
            valid_mask=valid,
            grid=grid,
            config=config,
        )
        assert result.layers == {}
        assert INDEX_NDVI in result.skipped
        assert INDEX_SR in result.skipped
        assert INDEX_SAVI in result.skipped

    def test_no_red_skips_ari(self, grid: Grid) -> None:
        config = SpectralConfig(
            enabled=(INDEX_ARI,),
            red_channel=None,
            nir_channel=3,
            green_channel=1,
            blue_channel=2,
            nodata=-9999.0,
            channel_std_min=0.01,
        )
        channels = _make_channels(grid)
        valid = np.ones(grid.shape, dtype=bool)
        result = build_spectral_indices(
            channels=channels,
            valid_mask=valid,
            grid=grid,
            config=config,
        )
        assert result.layers == {}
        assert INDEX_ARI in result.skipped

    def test_empty_enabled(self, grid: Grid) -> None:
        config = SpectralConfig(
            enabled=(),
            red_channel=0,
            nir_channel=3,
            green_channel=1,
            blue_channel=2,
            nodata=-9999.0,
            channel_std_min=0.01,
        )
        channels = _make_channels(grid)
        valid = np.ones(grid.shape, dtype=bool)
        result = build_spectral_indices(
            channels=channels,
            valid_mask=valid,
            grid=grid,
            config=config,
        )
        assert result.layers == {}
        assert result.skipped == ()

    def test_unknown_index_ignored(self, grid: Grid) -> None:
        config = SpectralConfig(
            enabled=("unknown_index", INDEX_NDVI),
            red_channel=0,
            nir_channel=3,
            green_channel=1,
            blue_channel=2,
            nodata=-9999.0,
            channel_std_min=0.01,
        )
        channels = _make_channels(grid)
        valid = np.ones(grid.shape, dtype=bool)
        result = build_spectral_indices(
            channels=channels,
            valid_mask=valid,
            grid=grid,
            config=config,
        )
        assert set(result.layers.keys()) == {INDEX_NDVI}

    def test_shape_mismatch_raises(self, grid: Grid, config: SpectralConfig) -> None:
        channels = {0: np.zeros((5, 5), dtype=np.float32)}
        valid = np.ones(grid.shape, dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            build_spectral_indices(
                channels=channels,
                valid_mask=valid,
                grid=grid,
                config=config,
            )

    def test_valid_mask_shape_raises(self, grid: Grid, config: SpectralConfig) -> None:
        channels = _make_channels(grid)
        valid = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="valid_mask"):
            build_spectral_indices(
                channels=channels,
                valid_mask=valid,
                grid=grid,
                config=config,
            )

    def test_constant_nir_skips_ndvi(self, grid: Grid) -> None:
        """Константный NIR (alpha-маска=255) не должен давать NIR-индексы."""
        config = SpectralConfig(
            enabled=(INDEX_NDVI, INDEX_SR, INDEX_SAVI),
            red_channel=0,
            nir_channel=3,
            green_channel=1,
            blue_channel=2,
            nodata=-9999.0,
            channel_std_min=0.01,
        )
        rng = np.random.default_rng(1)
        channels = {
            0: rng.uniform(1, 255, grid.shape).astype(np.float32),
            1: rng.uniform(1, 255, grid.shape).astype(np.float32),
            2: rng.uniform(1, 255, grid.shape).astype(np.float32),
            3: np.full(grid.shape, 255.0, dtype=np.float32),
        }
        valid = np.ones(grid.shape, dtype=bool)
        result = build_spectral_indices(
            channels=channels,
            valid_mask=valid,
            grid=grid,
            config=config,
        )
        assert INDEX_NDVI in result.skipped
        assert INDEX_SR in result.skipped
        assert INDEX_SAVI in result.skipped
        assert result.layers == {}

    def test_constant_red_skips_red_indices(self, grid: Grid) -> None:
        """Константный Red — все red-индексы пропущены."""
        config = SpectralConfig(
            enabled=(INDEX_NDVI, INDEX_SR, INDEX_ARI, INDEX_SAVI),
            red_channel=0,
            nir_channel=3,
            green_channel=1,
            blue_channel=2,
            nodata=-9999.0,
            channel_std_min=0.01,
        )
        rng = np.random.default_rng(2)
        channels = {
            0: np.zeros(grid.shape, dtype=np.float32),
            1: rng.uniform(1, 255, grid.shape).astype(np.float32),
            2: rng.uniform(1, 255, grid.shape).astype(np.float32),
            3: rng.uniform(1, 255, grid.shape).astype(np.float32),
        }
        valid = np.ones(grid.shape, dtype=bool)
        result = build_spectral_indices(
            channels=channels,
            valid_mask=valid,
            grid=grid,
            config=config,
        )
        assert INDEX_NDVI in result.skipped
        assert INDEX_SR in result.skipped
        assert INDEX_ARI in result.skipped
        assert INDEX_SAVI in result.skipped
        assert result.layers == {}
