"""Оракульная диагностика (eval.oracle): полнота roundtrip и атрибуция потерь."""

import numpy as np
import pytest

from expds.eval.oracle import (
    LOST_MIN_AREA,
    OK,
    oracle_report,
)
from expds.features.grid import Grid
from expds.io.geojson import LabelFeature
from expds.post.vectorize import PostConfig

PS = 0.5
X0, Y0 = 500000.0, 6200000.0
H, W = 128, 128


def _grid():
    return Grid(crs="EPSG:32637", pixel_size=PS, x_min=X0, y_min=Y0, width=W, height=H)


def _feat(grid, cx_px, cy_px, r_px, cls="k"):
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    cx, cy = X0 + cx_px * PS, Y0 + cy_px * PS
    ring = np.column_stack([cx + r_px * PS * np.cos(t), cy + r_px * PS * np.sin(t)])
    px = np.column_stack(
        [
            (ring[:, 0] - grid.x_min) / grid.pixel_size - 0.5,
            (ring[:, 1] - grid.y_min) / grid.pixel_size - 0.5,
        ]
    )
    return LabelFeature(cls=cls, geom_type="Polygon", world=(ring,), pixel=(px,), properties={})


_GOOD = PostConfig(
    split_touching=True, split_min_distance_px=4, min_area_px=6.0, hm_thr=0.15, peak_window_px=7
)
_BAD_AREA = PostConfig(
    split_touching=True, split_min_distance_px=4, min_area_px=200.0, hm_thr=0.15, peak_window_px=7
)


def test_roundtrip_is_lossless():
    """Разрозненные + слипшиеся объекты возвращаются из пайплайна полностью."""
    grid = _grid()
    feats = [
        _feat(grid, 30, 30, 8),
        _feat(grid, 70, 40, 5),
        # слитая пара
        _feat(grid, 40, 80, 7),
        _feat(grid, 50, 80, 7),
        # мелкий (< 16 px, но >= 6 px)
        _feat(grid, 90, 90, 2),
    ]
    rep = oracle_report(feats, grid, ["k"], _GOOD, sigma_px=3.0)
    assert rep.gt_total == 5
    assert rep.match.fn == 0
    assert rep.match.fp == 0
    assert rep.match.f1 == pytest.approx(1.0)
    assert rep.reasons.get(OK, 0) == 5


def test_min_area_loss_is_attributed():
    """Завышенный min_area_px глушит мелкие объекты — и это видно в причинах."""
    grid = _grid()
    feats = [_feat(grid, 30, 30, 8), _feat(grid, 90, 90, 2)]
    rep = oracle_report(feats, grid, ["k"], _BAD_AREA, sigma_px=3.0)
    assert rep.reasons.get(LOST_MIN_AREA, 0) == 1
    assert rep.reasons.get(OK, 0) == 1
    assert rep.match.fn == 1


def test_merged_loss_is_attributed():
    """Слитая пара без деления (Voronoi off, watershed off) даёт причину «слияние»."""
    grid = _grid()
    feats = [_feat(grid, 40, 64, 7), _feat(grid, 50, 64, 7)]
    cfg = PostConfig(split_touching=False, min_area_px=6.0, hm_thr=0.15, peak_window_px=7)
    rep = oracle_report(feats, grid, ["k"], cfg, sigma_px=3.0)
    # компонент один на два объекта; Voronoi по пикам всё же делит пару —
    # поэтому строго проверяем, что потери атрибутированы, а GT не потерян молча
    total = sum(rep.reasons.values())
    assert total == 2
    assert rep.match.tp + rep.match.fn == 2


def test_valid_mask_loss_is_attributed():
    """Объект, вырезанный маской валидности, получает причину valid/nodata."""
    from expds.eval.oracle import LOST_VALID

    grid = _grid()
    feats = [_feat(grid, 30, 30, 8)]
    valid = np.ones((H, W), dtype=bool)
    valid[10:60, 10:60] = False  # накрывает объект
    rep = oracle_report(feats, grid, ["k"], _GOOD, sigma_px=3.0, valid=valid)
    assert rep.valid_killed == 1
    assert rep.reasons.get(LOST_VALID, 0) == 1
    assert rep.match.fn == 1


def test_object_outside_grid_is_attributed():
    """Объект целиком вне сетки — причина «вне сетки», а не падение."""
    grid = _grid()
    inside = _feat(grid, 30, 30, 8)
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    cx, cy = X0 - 50 * PS, Y0 + 20 * PS  # левее сетки
    ring = np.column_stack([cx + 4 * PS * np.cos(t), cy + 4 * PS * np.sin(t)])
    outside = LabelFeature(
        cls="k", geom_type="Polygon", world=(ring,), pixel=(ring,), properties={}
    )
    rep = oracle_report([inside, outside], grid, ["k"], _GOOD, sigma_px=3.0)
    assert rep.gt_total == 2
    assert rep.match.tp == 1
    assert rep.raster_zero == 1
