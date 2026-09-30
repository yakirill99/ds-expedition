"""Тесты expds.io.geojson."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyproj import Transformer

from expds.features.grid import Grid
from expds.io.geojson import (
    GEOM_POINT,
    GEOM_POLYGON,
    Detection,
    read_labels,
    ring_centroid,
    write_detections,
)

_INTERNAL = "EPSG:32637"
_OUTPUT = "EPSG:3857"
_GRID = Grid(crs=_INTERNAL, pixel_size=1.0, x_min=500000.0, y_min=6200000.0, width=64, height=48)
_TO_OUT = Transformer.from_crs(_INTERNAL, _OUTPUT, always_xy=True)


def _ring_out(ring: list[tuple[float, float]]) -> list[list[float]]:
    xs, ys = _TO_OUT.transform([p[0] for p in ring], [p[1] for p in ring])
    out = [[float(a), float(b)] for a, b in zip(xs, ys, strict=True)]
    return out + [out[0]]


def _write_fc(path: Path, features: list[dict[str, Any]], crs_name: str | None) -> Path:
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs_name is not None:
        fc["crs"] = {"type": "name", "properties": {"name": crs_name}}
    path.write_text(json.dumps(fc), encoding="utf-8")
    return path


def test_ring_centroid() -> None:
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    assert np.allclose(ring_centroid(square), [1.0, 1.0])
    triangle = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    assert np.allclose(ring_centroid(triangle), [1.0, 1.0])
    far = square + np.array([4.0e6, 7.0e6])  # точность на координатах 3857
    assert np.allclose(ring_centroid(far), [4.0e6 + 1.0, 7.0e6 + 1.0], atol=1e-6)


def test_roundtrip_detections(tmp_path: Path) -> None:
    x0, y0 = _GRID.x_min, _GRID.y_min
    poly = np.array([[x0 + 5, y0 + 5], [x0 + 15, y0 + 5], [x0 + 15, y0 + 12], [x0 + 5, y0 + 12]])
    point = np.array([[x0 + 30.5, y0 + 40.5]])
    dets = [
        Detection(cls="rampart", score=0.9, geom_type=GEOM_POLYGON, geometry=poly),
        Detection(cls="mound", score=0.4, geom_type=GEOM_POINT, geometry=point),
    ]
    path = tmp_path / "pred.geojson"
    assert write_detections(path, dets, src_crs=_INTERNAL, dst_crs=_OUTPUT) == 2

    feats = read_labels(path, _GRID, src_crs=_OUTPUT)
    assert [f.cls for f in feats] == ["rampart", "mound"]
    assert np.allclose(feats[0].world[0], poly, atol=1e-6)
    assert np.allclose(feats[1].world[0], point, atol=1e-6)
    assert feats[0].properties["score"] == pytest.approx(0.9)
    # Пиксели: центр пикселя (30, 40) Grid = мир (x0 + 30.5, y0 + 40.5), row растёт на север.
    assert np.allclose(feats[1].pixel[0][0], [30.0, 40.0], atol=1e-6)


def test_multipolygon_exploded_holes_kept(tmp_path: Path) -> None:
    x0, y0 = _GRID.x_min, _GRID.y_min
    outer = [(x0 + 2, y0 + 2), (x0 + 12, y0 + 2), (x0 + 12, y0 + 12), (x0 + 2, y0 + 12)]
    hole = [(x0 + 5, y0 + 5), (x0 + 7, y0 + 5), (x0 + 7, y0 + 7), (x0 + 5, y0 + 7)]
    other = [(x0 + 20, y0 + 20), (x0 + 25, y0 + 20), (x0 + 25, y0 + 25)]
    geom = {
        "type": "MultiPolygon",
        "coordinates": [[_ring_out(outer), _ring_out(hole)], [_ring_out(other)]],
    }
    path = _write_fc(
        tmp_path / "gt.geojson",
        [{"type": "Feature", "properties": {"class": "ditch"}, "geometry": geom}],
        "urn:ogc:def:crs:EPSG::3857",
    )
    feats = read_labels(path, _GRID, src_crs=_OUTPUT)
    assert len(feats) == 2
    assert len(feats[0].world) == 2 and len(feats[1].world) == 1
    assert len(feats[0].world[0]) == 4  # замыкающая точка снята


def test_declared_crs_mismatch_raises(tmp_path: Path) -> None:
    path = _write_fc(tmp_path / "gt.geojson", [], "urn:ogc:def:crs:EPSG::4326")
    with pytest.raises(ValueError, match="CRS"):
        read_labels(path, _GRID, src_crs=_OUTPUT)


def test_missing_class_raises(tmp_path: Path) -> None:
    geom = {"type": "Point", "coordinates": _ring_out([(_GRID.x_min + 1, _GRID.y_min + 1)])[0]}
    path = _write_fc(
        tmp_path / "gt.geojson", [{"type": "Feature", "properties": {}, "geometry": geom}], None
    )
    with pytest.raises(ValueError, match="class"):
        read_labels(path, _GRID, src_crs=_OUTPUT)
