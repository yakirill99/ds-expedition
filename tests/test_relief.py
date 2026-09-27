"""Тесты производных рельефа (итерации 1, 2, 3)."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.relief import (
    build_relief_layers,
    compute_aspect,
    compute_curvature,
    compute_hillshade,
    compute_openness,
    compute_sky_view_factor,
    compute_slope,
    compute_slope_aspect,
    compute_slrm,
    compute_tpi,
    compute_tri,
)


def _plane(
    slope_x: float = 0.0,
    slope_y: float = 0.0,
    size: int = 50,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """Плоскость Z = slope_x * X + slope_y * Y."""
    x = np.arange(size) * pixel_size
    y = np.arange(size) * pixel_size
    xx, yy = np.meshgrid(x, y)
    return (slope_x * xx + slope_y * yy).astype(np.float32)


def _gaussian_hill(
    height: float = 5.0,
    sigma: float = 5.0,
    size: int = 50,
) -> np.ndarray:
    """Гауссов холм в центре."""
    center = size // 2
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    d2 = (xx - center) ** 2 + (yy - center) ** 2
    return (height * np.exp(-d2 / (2 * sigma * sigma))).astype(np.float32)


# ============================================================
# Итерация 1: slope, aspect, hillshade, SLRM
# ============================================================


class TestSlope:
    """Тесты наклона."""

    def test_flat_is_zero(self) -> None:
        dtm = _plane()
        slope = compute_slope(dtm=dtm, pixel_size=1.0)
        assert np.abs(slope[1:-1, 1:-1]).max() < 1e-3

    def test_constant_slope(self) -> None:
        dtm = _plane(slope_x=0.1)
        slope = compute_slope(dtm=dtm, pixel_size=1.0)
        expected = np.degrees(np.arctan(0.1))
        assert slope[1:-1, 1:-1].mean() == pytest.approx(expected, abs=0.1)

    def test_45_degrees(self) -> None:
        dtm = _plane(slope_x=1.0)
        slope = compute_slope(dtm=dtm, pixel_size=1.0)
        assert slope[1:-1, 1:-1].mean() == pytest.approx(45.0, abs=0.5)


class TestAspect:
    """Тесты экспозиции."""

    def test_flat_undefined(self) -> None:
        dtm = _plane()
        aspect = compute_aspect(dtm=dtm, pixel_size=1.0)
        assert aspect.shape == dtm.shape

    def test_west_facing(self) -> None:
        # Высота растёт вправо => склон смотрит на запад (270°).
        dtm = _plane(slope_x=0.1)
        aspect = compute_aspect(dtm=dtm, pixel_size=1.0)
        assert aspect[1:-1, 1:-1].mean() == pytest.approx(270.0, abs=1.0)

    def test_north_facing(self) -> None:
        # Grid: row растёт на север. Высота падает к северу => склон смотрит на север (0°).
        dtm = _plane(slope_y=-0.1)
        aspect = compute_aspect(dtm=dtm, pixel_size=1.0)
        mean_aspect = aspect[1:-1, 1:-1].mean()
        assert min(abs(mean_aspect), abs(mean_aspect - 360.0)) < 1.0


class TestHillshade:
    """Тесты hillshade."""

    def test_flat_fully_lit(self) -> None:
        dtm = _plane()
        hs = compute_hillshade(dtm=dtm, pixel_size=1.0, azimuth_deg=315.0, altitude_deg=45.0)
        expected = np.sin(np.radians(45.0)) * 255.0
        assert hs[1:-1, 1:-1].mean() == pytest.approx(expected, abs=1.0)

    def test_range_0_255(self) -> None:
        dtm = _gaussian_hill()
        hs = compute_hillshade(dtm=dtm, pixel_size=1.0, azimuth_deg=315.0)
        assert hs.min() >= 0.0
        assert hs.max() <= 255.0

    def test_different_azimuths_differ(self) -> None:
        dtm = _gaussian_hill()
        hs_ne = compute_hillshade(dtm=dtm, pixel_size=1.0, azimuth_deg=45.0)
        hs_sw = compute_hillshade(dtm=dtm, pixel_size=1.0, azimuth_deg=225.0)
        assert not np.allclose(hs_ne, hs_sw)


class TestSlrm:
    """Тесты SLRM."""

    def test_flat_returns_zero(self) -> None:
        dtm = _plane()
        slrm = compute_slrm(dtm=dtm, sigma=5.0)
        assert np.abs(slrm[5:-5, 5:-5]).max() < 1e-3

    def test_hill_positive_at_center(self) -> None:
        dtm = _gaussian_hill(height=5.0, sigma=5.0)
        slrm = compute_slrm(dtm=dtm, sigma=10.0)
        center = dtm.shape[0] // 2
        assert slrm[center, center] > 0.0

    def test_hill_negative_at_edges(self) -> None:
        dtm = _gaussian_hill(height=5.0, sigma=5.0)
        slrm = compute_slrm(dtm=dtm, sigma=10.0)
        center = dtm.shape[0] // 2
        edge = center + 15
        assert slrm[center, edge] < 0.0

    def test_removes_large_scale(self) -> None:
        size = 200
        x = np.arange(size)
        xx, yy = np.meshgrid(x, x)
        trend = 0.5 * xx
        hill = 3.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / 200.0)
        dtm = (trend + hill).astype(np.float32)

        slrm = compute_slrm(dtm=dtm, sigma=20.0)
        assert abs(slrm[70, 70]) < 0.5
        assert abs(slrm[130, 130]) < 0.5
        assert slrm[100, 100] > 0.5


class TestSlopeAspect:
    """Тесты совместного вычисления."""

    def test_matches_separate(self) -> None:
        dtm = _gaussian_hill()
        slope_joint, aspect_joint = compute_slope_aspect(dtm=dtm, pixel_size=1.0)
        slope_sep = compute_slope(dtm=dtm, pixel_size=1.0)
        aspect_sep = compute_aspect(dtm=dtm, pixel_size=1.0)
        np.testing.assert_allclose(slope_joint, slope_sep, atol=1e-5)
        np.testing.assert_allclose(aspect_joint, aspect_sep, atol=1e-5)


# ============================================================
# Итерация 2: openness, sky-view factor, curvature
# ============================================================


class TestOpenness:
    """Тесты openness."""

    def test_flat_surface_openness_near_90(self) -> None:
        dtm = _plane()
        pos, neg = compute_openness(dtm=dtm, pixel_size=1.0, radius=5, n_directions=8)
        # На плоскости обе openness ≈ 90°.
        assert pos[10:-10, 10:-10].mean() == pytest.approx(90.0, abs=1.0)
        assert neg[10:-10, 10:-10].mean() == pytest.approx(90.0, abs=1.0)

    def test_hill_positive_openness_high(self) -> None:
        dtm = _gaussian_hill(height=5.0, sigma=5.0)
        pos, _ = compute_openness(dtm=dtm, pixel_size=1.0, radius=10, n_directions=16)
        center = dtm.shape[0] // 2
        # На вершине positive openness > 90° (горизонт ниже).
        assert pos[center, center] > 90.0

    def test_valley_negative_openness_high(self) -> None:
        dtm = -_gaussian_hill(height=5.0, sigma=5.0)
        _, neg = compute_openness(dtm=dtm, pixel_size=1.0, radius=10, n_directions=16)
        center = dtm.shape[0] // 2
        # В яме negative openness > 90° (низ ниже горизонта).
        assert neg[center, center] > 90.0

    def test_returns_correct_shapes(self) -> None:
        dtm = _gaussian_hill()
        pos, neg = compute_openness(dtm=dtm, pixel_size=1.0, radius=5, n_directions=8)
        assert pos.shape == dtm.shape
        assert neg.shape == dtm.shape


class TestSkyViewFactor:
    """Тесты sky-view factor."""

    def test_flat_surface_svf_near_half(self) -> None:
        dtm = _plane()
        pos, neg = compute_openness(dtm=dtm, pixel_size=1.0, radius=5, n_directions=8)
        svf = compute_sky_view_factor(positive_openness=pos, negative_openness=neg)
        assert 0.4 < svf[10:-10, 10:-10].mean() < 0.6

    def test_svf_range(self) -> None:
        dtm = _gaussian_hill()
        pos, neg = compute_openness(dtm=dtm, pixel_size=1.0, radius=10, n_directions=16)
        svf = compute_sky_view_factor(positive_openness=pos, negative_openness=neg)
        assert svf.min() >= 0.0
        assert svf.max() <= 1.0


class TestCurvature:
    """Тесты кривизны."""

    def test_flat_curvature_zero(self) -> None:
        dtm = _plane()
        curv = compute_curvature(dtm=dtm, pixel_size=1.0, sigma=2.0)
        assert abs(curv[10:-10, 10:-10]).max() < 1e-3

    def test_hill_negative_curvature(self) -> None:
        dtm = _gaussian_hill(height=5.0, sigma=5.0)
        curv = compute_curvature(dtm=dtm, pixel_size=1.0, sigma=2.0)
        center = dtm.shape[0] // 2
        assert curv[center, center] < 0.0

    def test_valley_positive_curvature(self) -> None:
        dtm = -_gaussian_hill(height=5.0, sigma=5.0)
        curv = compute_curvature(dtm=dtm, pixel_size=1.0, sigma=2.0)
        center = dtm.shape[0] // 2
        assert curv[center, center] > 0.0


# ============================================================
# Итерация 3: TPI, TRI
# ============================================================


class TestTpi:
    """Тесты TPI."""

    def test_flat_returns_zero(self) -> None:
        dtm = _plane()
        tpi = compute_tpi(dtm=dtm, radius=5)
        assert np.abs(tpi[10:-10, 10:-10]).max() < 1e-3

    def test_hill_positive_at_center(self) -> None:
        dtm = _gaussian_hill(height=5.0, sigma=5.0)
        tpi = compute_tpi(dtm=dtm, radius=10)
        center = dtm.shape[0] // 2
        # В центре холма TPI > 0 (выше окружения).
        assert tpi[center, center] > 0.0

    def test_hill_negative_at_edges(self) -> None:
        dtm = _gaussian_hill(height=5.0, sigma=5.0)
        tpi = compute_tpi(dtm=dtm, radius=10)
        center = dtm.shape[0] // 2
        # На периферии холма TPI < 0 (ниже среднего в окне).
        edge = center + 12
        assert tpi[center, edge] < 0.0

    def test_valley_negative_at_center(self) -> None:
        dtm = -_gaussian_hill(height=5.0, sigma=5.0)
        tpi = compute_tpi(dtm=dtm, radius=10)
        center = dtm.shape[0] // 2
        # В яме TPI < 0.
        assert tpi[center, center] < 0.0

    def test_returns_correct_shape(self) -> None:
        dtm = _gaussian_hill()
        tpi = compute_tpi(dtm=dtm, radius=5)
        assert tpi.shape == dtm.shape


class TestTri:
    """Тесты TRI."""

    def test_flat_returns_zero(self) -> None:
        dtm = _plane()
        tri = compute_tri(dtm=dtm, radius=5)
        assert np.abs(tri[10:-10, 10:-10]).max() < 1e-3

    def test_noisy_surface_positive(self) -> None:
        rng = np.random.default_rng(42)
        dtm = rng.normal(0, 1.0, (50, 50)).astype(np.float32)
        tri = compute_tri(dtm=dtm, radius=5)
        assert tri[10:-10, 10:-10].mean() > 0.0

    def test_returns_correct_shape(self) -> None:
        dtm = _gaussian_hill()
        tri = compute_tri(dtm=dtm, radius=5)
        assert tri.shape == dtm.shape

    def test_tri_scales_with_roughness(self) -> None:
        # Гладкая поверхность vs шумная.
        rng = np.random.default_rng(42)
        smooth = _plane() + rng.normal(0, 0.01, (50, 50)).astype(np.float32)
        rough = _plane() + rng.normal(0, 1.0, (50, 50)).astype(np.float32)
        tri_smooth = compute_tri(dtm=smooth, radius=5)
        tri_rough = compute_tri(dtm=rough, radius=5)
        # TRI на шумной поверхности должен быть больше.
        assert tri_rough[10:-10, 10:-10].mean() > tri_smooth[10:-10, 10:-10].mean() * 10


# ============================================================
# Сборка всех слоёв
# ============================================================


class TestBuildReliefLayers:
    """Тесты сборки всех слоёв (итерации 1, 2, 3)."""

    def test_all_layers_present(self) -> None:
        dtm = _gaussian_hill()
        layers = build_relief_layers(
            dtm=dtm,
            pixel_size=1.0,
            hillshade_azimuths=(45.0, 135.0),
            hillshade_altitude=45.0,
            hillshade_z_factor=1.0,
            slrm_sigmas=(2.0, 8.0),
            openness_radius=5,
            openness_n_directions=8,
            curvature_sigmas=(2.0, 8.0),
            tpi_radii=(5, 15),
            tri_radii=(3, 9),
        )
        expected_keys = {
            "slope",
            "aspect",
            "hillshade_az_45",
            "hillshade_az_135",
            "slrm_sigma_2.0",
            "slrm_sigma_8.0",
            "positive_openness",
            "negative_openness",
            "sky_view_factor",
            "curvature_sigma_2.0",
            "curvature_sigma_8.0",
            "tpi_radius_5",
            "tpi_radius_15",
            "tri_radius_3",
            "tri_radius_9",
        }
        assert set(layers.keys()) == expected_keys

    def test_all_layers_shape_and_dtype(self) -> None:
        dtm = _gaussian_hill()
        layers = build_relief_layers(
            dtm=dtm,
            pixel_size=1.0,
            hillshade_azimuths=(45.0,),
            hillshade_altitude=45.0,
            hillshade_z_factor=1.0,
            slrm_sigmas=(2.0,),
            openness_radius=5,
            openness_n_directions=8,
            curvature_sigmas=(2.0,),
            tpi_radii=(5,),
            tri_radii=(3,),
        )
        for name, arr in layers.items():
            assert arr.shape == dtm.shape, f"{name}: shape {arr.shape}"
            assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype}"

    def test_empty_config_produces_base_layers(self) -> None:
        # Без hillshade, SLRM, кривизны, TPI, TRI — только slope/aspect/openness.
        dtm = _gaussian_hill()
        layers = build_relief_layers(
            dtm=dtm,
            pixel_size=1.0,
            hillshade_azimuths=(),
            hillshade_altitude=45.0,
            hillshade_z_factor=1.0,
            slrm_sigmas=(),
            openness_radius=5,
            openness_n_directions=8,
            curvature_sigmas=(),
            tpi_radii=(),
            tri_radii=(),
        )
        expected_keys = {
            "slope",
            "aspect",
            "positive_openness",
            "negative_openness",
            "sky_view_factor",
        }
        assert set(layers.keys()) == expected_keys
