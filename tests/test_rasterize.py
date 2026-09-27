"""Тесты expds.targets.rasterize (вход — GeoJSON в 3857, как у организаторов)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

from expds.features.grid import Grid
from expds.io.geojson import LabelFeature, read_labels
from expds.targets.rasterize import labels_to_targets, points_to_heatmap, polygons_to_mask

_INTERNAL = "EPSG:32637"
_OUTPUT = "EPSG:3857"
_GRID = Grid(crs=_INTERNAL, pixel_size=1.0, x_min=500000.0, y_min=6200000.0, width=64, height=48)
_TO_OUT = Transformer.from_crs(_INTERNAL, _OUTPUT, always_xy=True)
_CLASSES = ("mound", "ditch")


def _out(x: float, y: float) -> list[float]:
    ox, oy = _TO_OUT.transform(_GRID.x_min + x, _GRID.y_min + y)
    return [float(ox), float(oy)]


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Прямоугольник в локальных метрах от (x_min, y_min) -> замкнутое кольцо 3857."""
    ring = [_out(x0, y0), _out(x1, y0), _out(x1, y1), _out(x0, y1)]
    return ring + [ring[0]]


def _feats(tmp_path: Path, features: list[tuple[str, dict[str, Any]]]) -> list[LabelFeature]:
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": [
            {"type": "Feature", "properties": {"class": cls}, "geometry": geom}
            for cls, geom in features
        ],
    }
    path = tmp_path / "gt.geojson"
    path.write_text(json.dumps(fc), encoding="utf-8")
    return read_labels(path, _GRID, src_crs=_OUTPUT)


def test_square_10x10_is_100_pixels_in_grid_orientation(tmp_path: Path) -> None:
    # Края по границам пикселей: x 10..20 м, y 30..40 м (северная часть сетки).
    feats = _feats(
        tmp_path, [("ditch", {"type": "Polygon", "coordinates": [_rect(10, 30, 20, 40)]})]
    )
    mask = polygons_to_mask(feats, _GRID, _CLASSES)
    assert mask.shape == (2, 48, 64)
    assert mask[0].sum() == 0
    assert mask[1].sum() == 100
    rows, cols = np.nonzero(mask[1])
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (30, 39, 10, 19)  # row 0 = юг


def test_polygon_with_hole(tmp_path: Path) -> None:
    geom = {"type": "Polygon", "coordinates": [_rect(0, 0, 20, 20), _rect(5, 5, 11, 11)]}
    mask = polygons_to_mask(_feats(tmp_path, [("mound", geom)]), _GRID, _CLASSES)
    assert mask[0].sum() == 400 - 36
    assert mask[0, 7, 7] == 0.0 and mask[0, 2, 2] == 1.0


def test_polygon_clipped_by_grid(tmp_path: Path) -> None:
    geom = {"type": "Polygon", "coordinates": [_rect(-5, -5, 5, 5)]}
    mask = polygons_to_mask(_feats(tmp_path, [("mound", geom)]), _GRID, _CLASSES)
    assert mask[0].sum() == 25


def test_point_heatmap_peak(tmp_path: Path) -> None:
    # Центр пикселя (col 7, row 40) = мир (7.5, 40.5).
    feats = _feats(tmp_path, [("mound", {"type": "Point", "coordinates": _out(7.5, 40.5)})])
    heat = points_to_heatmap(feats, _GRID, _CLASSES, sigma_px=2.0)
    assert np.unravel_index(np.argmax(heat[0]), heat[0].shape) == (40, 7)
    assert heat[0, 40, 7] == np.float32(1.0)
    assert heat[1].sum() == 0
    assert np.isclose(heat[0, 40, 7 + 2], np.exp(-0.5), atol=1e-6)  # один σ в сторону


def test_heatmap_overlap_is_max_not_sum(tmp_path: Path) -> None:
    pt = {"type": "Point", "coordinates": _out(20.5, 20.5)}
    heat = points_to_heatmap(_feats(tmp_path, [("mound", pt), ("mound", pt)]), _GRID, _CLASSES, 2.0)
    assert float(heat.max()) == 1.0


def test_unknown_class_skipped_and_types_split(tmp_path: Path) -> None:
    feats = _feats(
        tmp_path,
        [
            ("rampart", {"type": "Polygon", "coordinates": [_rect(0, 0, 4, 4)]}),
            ("mound", {"type": "Point", "coordinates": _out(0.5, 0.5)}),
            ("mound", {"type": "Polygon", "coordinates": [_rect(30, 0, 34, 4)]}),
        ],
    )
    y_seg, y_hm = labels_to_targets(feats, _GRID, _CLASSES, sigma_px=1.5)
    assert y_seg[0].sum() == 16 and y_seg[1].sum() == 0  # rampart пропущен
    assert y_hm[0, 0, 0] == np.float32(1.0)  # точка у края — без падения
    assert y_hm[0, :, 30:].sum() == 0  # полигон в heatmap не попадает
