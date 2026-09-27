"""Регрессия: aspect/hillshade в конвенции Grid (row 0 = юг, row растёт на север)."""

from __future__ import annotations

import numpy as np

from expds.features.relief import compute_aspect, compute_hillshade, compute_slope_aspect

_N = 16
_INNER = (slice(2, -2), slice(2, -2))


def _plane_rising_north() -> np.ndarray:
    """z = row: высота растёт на север -> склон смотрит на юг."""
    return np.tile(np.arange(_N, dtype=np.float32)[:, None], (1, _N))


def _plane_rising_east() -> np.ndarray:
    """z = col: высота растёт на восток -> склон смотрит на запад."""
    return np.tile(np.arange(_N, dtype=np.float32)[None, :], (_N, 1))


def test_aspect_north_rising_faces_south() -> None:
    aspect = compute_aspect(_plane_rising_north(), 1.0)
    assert np.allclose(aspect[_INNER], 180.0, atol=1e-3)


def test_aspect_east_rising_faces_west() -> None:
    aspect = compute_aspect(_plane_rising_east(), 1.0)
    assert np.allclose(aspect[_INNER], 270.0, atol=1e-3)


def test_slope_aspect_consistent_with_aspect() -> None:
    dtm = _plane_rising_north()
    _, aspect = compute_slope_aspect(dtm, 1.0)
    assert np.allclose(aspect[_INNER], compute_aspect(dtm, 1.0)[_INNER])


def test_hillshade_sun_from_south_lights_south_slope() -> None:
    dtm = _plane_rising_north()
    from_south = compute_hillshade(dtm, 1.0, azimuth_deg=180.0)
    from_north = compute_hillshade(dtm, 1.0, azimuth_deg=0.0)
    assert float(from_south[_INNER].mean()) > float(from_north[_INNER].mean())
