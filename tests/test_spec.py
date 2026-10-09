"""Тесты контрактов spec.py (Блок 1)."""

from __future__ import annotations

import textwrap

import pytest

from expds.spec import DataSpec, Detection, Grid, load_spec


def _write(p, text: str):
    p.write_text(textwrap.dedent(text), encoding="utf-8")


# --- Grid ---------------------------------------------------------------------


def test_grid_pixel_to_world_centre():
    # origin (1000, 2000), пиксель 2 м, north-up
    g = Grid(crs="EPSG:3857", transform=(2.0, 0.0, 1000.0, 0.0, -2.0, 2000.0), width=10, height=10)
    xs, ys = g.pixel_to_world([0], [0])
    # центр пикселя (0,0) => (1000 + 0.5*2, 2000 - 0.5*2) = (1001, 1999)
    assert xs[0] == pytest.approx(1001.0)
    assert ys[0] == pytest.approx(1999.0)


def test_grid_pixel_size_and_bounds():
    g = Grid(crs="EPSG:3857", transform=(2.0, 0.0, 1000.0, 0.0, -2.0, 2000.0), width=10, height=10)
    assert g.pixel_size == pytest.approx(2.0)
    assert g.bounds == pytest.approx((1000.0, 1980.0, 1020.0, 2000.0))


def test_grid_roundtrip_world_pixel():
    g = Grid(crs="EPSG:3857", transform=(0.5, 0.0, 100.0, 0.0, -0.5, 200.0), width=100, height=100)
    xs, ys = g.pixel_to_world([3, 7], [4, 9])
    rows, cols = g.world_to_pixel(xs, ys)
    assert list(rows) == pytest.approx([3.0, 7.0], abs=1e-9)
    assert list(cols) == pytest.approx([4.0, 9.0], abs=1e-9)


# --- Detection -----------------------------------------------------------------


def test_detection_to_geojson_feature():
    d = Detection(
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        score=0.9,
        class_id=1,
        class_name="kurgan",
        source="relief",
    )
    f = d.to_geojson_feature()
    assert f["type"] == "Feature"
    assert f["properties"]["class_name"] == "kurgan"
    assert f["geometry"]["type"] == "Polygon"


# --- load_spec -----------------------------------------------------------------

MINIMAL = """
    name: t
    tile: 512
    overlap: 128
    r_tolerance: 10.0
    iou_threshold: 0.5
    classes:
      0: background
      1: kurgan
    layers:
      - name: dtm
        group: relief
        glob: "dtm.tif"
        kind: raster
        optional: true
    sites:
      - site_id: site_1
        root: data/site_1
        grid: null
        layers: {}
"""


def test_load_spec_minimal_optional(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "dataset.yaml"
    _write(cfg, MINIMAL)
    spec = load_spec(cfg)
    assert isinstance(spec, DataSpec)
    assert spec.tile == 512
    assert spec.classes[1] == "kurgan"
    assert len(spec.sites) == 1
    assert spec.sites[0].site_id == "site_1"


def test_load_spec_missing_required_layer_fails(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "dataset.yaml"
    _write(cfg, MINIMAL.replace("optional: true", "optional: false"))
    with pytest.raises(ValueError, match="обязательный слой"):
        load_spec(cfg)


def test_load_spec_bad_group(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "dataset.yaml"
    _write(cfg, MINIMAL.replace("group: relief", "group: bogus"))
    with pytest.raises(ValueError, match="group"):
        load_spec(cfg)


def test_load_spec_bad_tile_overlap(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "dataset.yaml"
    _write(cfg, MINIMAL.replace("tile: 512", "tile: 64").replace("overlap: 128", "overlap: 256"))
    with pytest.raises(ValueError, match="tile > overlap"):
        load_spec(cfg)


def test_load_spec_tile_not_multiple_of_32(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "dataset.yaml"
    _write(cfg, MINIMAL.replace("tile: 512", "tile: 1000"))
    with pytest.raises(ValueError, match="делиться на 32"):
        load_spec(cfg)


def test_load_spec_bad_iou(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "dataset.yaml"
    _write(cfg, MINIMAL.replace("iou_threshold: 0.5", "iou_threshold: 0.0"))
    with pytest.raises(ValueError, match="iou_threshold"):
        load_spec(cfg)


def test_load_spec_file_not_found(tmp_path):
    with pytest.raises(ValueError, match="не найден"):
        load_spec(tmp_path / "nope.yaml")
