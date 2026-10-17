"""Тесты контрактов spec.py: load_spec читает формат костяка.

Grid/Detection здесь не тестируются — они живут в своих модулях
(tests/test_grid.py, tests/test_geojson_io.py). Здесь проверяется только то,
что spec.py на них ссылается и не заводит второй версии.
"""

from __future__ import annotations

import textwrap

import pytest

from expds.features.grid import Grid
from expds.io.geojson import Detection
from expds.spec import DataSpec, load_spec


def _write(p, text: str) -> None:
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _cfg(tmp_path, body: str):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(exist_ok=True)
    cfg = cfg_dir / "dataset.yaml"
    _write(cfg, body)
    return cfg


HEAD = """
    data:
      sites_dir: data/raw
      labels_file: labels.geojson
      channels: [dtm, hillshade]
      normalize: site_robust
    targets:
      classes: [mound, rampart, ditch]
      heatmap_sigma_px: 3.0
    tiles:
      tile_px: 512
      overlap_px: 128
      pos_fraction: 0.7
    eval:
      iou_thr: 0.5
      r_tol: 10.0
    post:
      nms_radius_px: 9.0
"""

LAYERS = """
    layers:
      - name: dtm
        group: relief
        glob: "dtm.tif"
        kind: raster
        optional: true
"""

SITES = """
    sites:
      - site_id: site_1
"""

MINIMAL = HEAD + LAYERS + SITES


# --- масштаб и метрика берутся из секций костяка -------------------------------


def test_load_spec_reads_pipeline_sections(tmp_path):
    spec = load_spec(_cfg(tmp_path, MINIMAL))
    assert isinstance(spec, DataSpec)
    assert spec.tile == 512
    assert spec.overlap == 128
    assert spec.r_tolerance == pytest.approx(10.0)
    assert spec.iou_threshold == pytest.approx(0.5)
    assert spec.sigma_px == pytest.approx(3.0)
    assert spec.r_nms == pytest.approx(9.0)
    assert spec.classes == ("mound", "rampart", "ditch")


def test_spec_reexports_canonical_classes():
    # spec.py не заводит вторую версию Grid/Detection, а ссылается на канон.
    import expds.spec as spec_mod

    assert spec_mod.Grid is Grid
    assert spec_mod.Detection is Detection


# --- участки ------------------------------------------------------------------


def test_load_spec_default_root_from_sites_dir(tmp_path):
    spec = load_spec(_cfg(tmp_path, MINIMAL))
    assert len(spec.sites) == 1
    assert spec.sites[0].site_id == "site_1"
    assert spec.sites[0].root == (tmp_path / "data/raw/site_1").resolve()
    assert spec.sites[0].grid is None


EXPLICIT_ROOT = """
    sites:
      - site_id: site_1
        root: data/elsewhere/site_9
"""


def test_load_spec_explicit_root_wins(tmp_path):
    spec = load_spec(_cfg(tmp_path, HEAD + LAYERS + EXPLICIT_ROOT))
    assert spec.sites[0].root == (tmp_path / "data/elsewhere/site_9").resolve()


GRID_SITES = """
    sites:
      - site_id: site_1
        grid:
          crs: EPSG:32637
          pixel_size: 1.0
          x_min: 0.0
          y_min: 0.0
          width: 100
          height: 200
"""


def test_load_spec_grid_from_dict(tmp_path):
    spec = load_spec(_cfg(tmp_path, HEAD + LAYERS + GRID_SITES))
    g = spec.sites[0].grid
    assert isinstance(g, Grid)
    assert g.crs == "EPSG:32637"
    assert g.shape == (200, 100)
    # Единая конвенция пайплайна: row 0 = юг, y растёт вверх.
    assert g.transform_to_world(0, 0)[1] == pytest.approx(0.5)


BAD_GRID = """
    sites:
      - site_id: site_1
        grid:
          crs: EPSG:32637
          pixel_size: 1.0
"""


def test_load_spec_grid_missing_field(tmp_path):
    with pytest.raises(ValueError, match="не хватает полей"):
        load_spec(_cfg(tmp_path, HEAD + LAYERS + BAD_GRID))


TWO_CRS = """
    sites:
      - site_id: s1
        grid: {crs: EPSG:32637, pixel_size: 1.0, x_min: 0.0, y_min: 0.0, width: 8, height: 8}
      - site_id: s2
        grid: {crs: EPSG:32638, pixel_size: 1.0, x_min: 0.0, y_min: 0.0, width: 8, height: 8}
"""


def test_load_spec_crs_conflict(tmp_path):
    with pytest.raises(ValueError, match="одинаковый crs"):
        load_spec(_cfg(tmp_path, HEAD + LAYERS + TWO_CRS))


# --- слои ---------------------------------------------------------------------


def test_load_spec_optional_layer_missing_ok(tmp_path):
    spec = load_spec(_cfg(tmp_path, MINIMAL))
    assert spec.sites[0].layers == {}
    assert spec.layer_defs[0].optional is True


def test_load_spec_required_layer_missing(tmp_path):
    with pytest.raises(ValueError, match="обязательный слой"):
        load_spec(_cfg(tmp_path, MINIMAL.replace("optional: true", "optional: false")))


def test_load_spec_layer_bad_group(tmp_path):
    with pytest.raises(ValueError, match="group"):
        load_spec(_cfg(tmp_path, MINIMAL.replace("group: relief", "group: nope")))


DUPLICATE_LAYERS = """
    layers:
      - name: dtm
        group: relief
        glob: "dtm.tif"
        kind: raster
        optional: true
      - name: dtm
        group: optic
        glob: "dtm2.tif"
        kind: raster
        optional: true
"""


def test_load_spec_duplicate_layer_names(tmp_path):
    with pytest.raises(ValueError, match="дублирующиеся имена слоёв"):
        load_spec(_cfg(tmp_path, HEAD + DUPLICATE_LAYERS + SITES))


# --- валидация масштаба -------------------------------------------------------


def test_load_spec_tile_le_overlap(tmp_path):
    body = MINIMAL.replace("tile_px: 512", "tile_px: 64").replace(
        "overlap_px: 128", "overlap_px: 256"
    )
    with pytest.raises(ValueError, match="tile_px > tiles.overlap_px"):
        load_spec(_cfg(tmp_path, body))


def test_load_spec_tile_not_multiple_of_32(tmp_path):
    # 1000 > 128 (первая проверка проходит) и 1000 % 32 != 0 (ловит вторая).
    body = MINIMAL.replace("tile_px: 512", "tile_px: 1000")
    with pytest.raises(ValueError, match="делиться на 32"):
        load_spec(_cfg(tmp_path, body))


# --- прочие ошибки загрузки ---------------------------------------------------


def test_load_spec_missing_file(tmp_path):
    with pytest.raises(ValueError, match="не найден"):
        load_spec(tmp_path / "nope.yaml")


def test_load_spec_missing_required_section(tmp_path):
    with pytest.raises(ValueError, match="нет секции 'targets'"):
        load_spec(_cfg(tmp_path, MINIMAL.replace("    targets:", "    tmprgts:")))


def test_load_spec_bad_iou(tmp_path):
    with pytest.raises(ValueError, match="iou_thr"):
        load_spec(_cfg(tmp_path, MINIMAL.replace("iou_thr: 0.5", "iou_thr: 1.5")))


def test_load_spec_empty_classes(tmp_path):
    body = MINIMAL.replace("classes: [mound, rampart, ditch]", "classes: []")
    with pytest.raises(ValueError, match="непустым списком"):
        load_spec(_cfg(tmp_path, body))
