"""Тесты генератора синтетики (tools/make_synthetic.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import rasterio
from pyproj import Transformer

from expds.features.config import load_config

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "dataset_synthetic.yaml"
_TOOL = _ROOT / "tools" / "make_synthetic.py"
_SIZE = 256
_N_SITES = 2


def _load_tool() -> ModuleType:
    name = "make_synthetic"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(out_dir: Path, seed: int = 0) -> None:
    argv = [
        "--config", str(_CONFIG),
        "--n-sites", str(_N_SITES),
        "--size", str(_SIZE),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]  # fmt: skip
    assert _load_tool().main(argv) == 0


def _labels(site_dir: Path) -> dict[str, Any]:
    return json.loads((site_dir / "labels.geojson").read_text(encoding="utf-8"))


def _center(feature: dict[str, Any]) -> tuple[float, float]:
    geom = feature["geometry"]
    if geom["type"] == "Point":
        return float(geom["coordinates"][0]), float(geom["coordinates"][1])
    ring = np.asarray(geom["coordinates"][0][:-1])
    return float(ring[:, 0].mean()), float(ring[:, 1].mean())


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synth")
    _run(out)
    return out


def test_config_readable_by_role1_loader() -> None:
    app = load_config(_CONFIG)
    assert app.crs.output == "EPSG:3857"
    assert app.grid.pixel_size == 1.0


def test_site_structure(synth_dir: Path) -> None:
    for i in range(1, _N_SITES + 1):
        site = synth_dir / f"site_{i}"
        for name in ("dtm", "slope", "aspect", "rgb_r", "rgb_g", "rgb_b", "mag"):
            assert (site / "layers" / f"{name}.tif").exists(), name
        assert list((site / "layers").glob("hillshade_az_*.tif"))
        assert list((site / "layers").glob("slrm_sigma_*.tif"))

        meta = json.loads((site / "samples" / "sample_meta.json").read_text(encoding="utf-8"))
        assert meta["channel_names"][0] == "dtm"
        assert {"rgb_r", "rgb_g", "rgb_b", "mag"} <= set(meta["channel_names"])
        with np.load(site / "samples" / "sample.npz") as npz:
            assert npz["tensor"].shape == (len(meta["channel_names"]), _SIZE, _SIZE)
            assert bool(npz["channel_mask"].all())


def test_labels_format(synth_dir: Path) -> None:
    fc = _labels(synth_dir / "site_1")
    assert fc["crs"]["properties"]["name"].endswith("3857")
    classes = {f["properties"]["class"] for f in fc["features"]}
    assert "mound" in classes
    assert classes <= {"mound", "rampart", "ditch"}
    for f in fc["features"]:
        geom = f["geometry"]
        assert geom["type"] in {"Point", "Polygon"}
        if geom["type"] == "Polygon":
            ring = geom["coordinates"][0]
            assert ring[0] == ring[-1]
            assert len(ring) >= 5


def test_mag_partial_coverage(synth_dir: Path) -> None:
    with rasterio.open(synth_dir / "site_1" / "layers" / "mag.tif") as src:
        arr = src.read(1)
        nodata = src.nodata
    assert nodata is not None
    frac_valid = float((arr != nodata).mean())
    assert 0.5 < frac_valid < 1.0


def test_deterministic(synth_dir: Path, tmp_path: Path) -> None:
    _run(tmp_path)
    for i in range(1, _N_SITES + 1):
        site_a, site_b = synth_dir / f"site_{i}", tmp_path / f"site_{i}"
        assert (site_a / "labels.geojson").read_bytes() == (site_b / "labels.geojson").read_bytes()
        with (
            rasterio.open(site_a / "layers" / "dtm.tif") as a,
            rasterio.open(site_b / "layers" / "dtm.tif") as b,
        ):
            assert np.array_equal(a.read(1), b.read(1))


def test_sites_differ(synth_dir: Path) -> None:
    assert _labels(synth_dir / "site_1")["features"] != _labels(synth_dir / "site_2")["features"]


def test_mounds_are_hills_in_file_orientation(synth_dir: Path) -> None:
    """Независимый путь: 3857 -> internal (pyproj) -> индекс по north-up transform файла.

    Если Y где-то отражён, центр кургана попадёт на фон и SLRM будет ~0.
    """
    app = load_config(_CONFIG)
    to_internal = Transformer.from_crs(app.crs.output, app.crs.internal, always_xy=True)
    slrm_name = f"slrm_sigma_{app.relief.slrm_sigmas[0]}"

    checked = passed = 0
    for i in range(1, _N_SITES + 1):
        site = synth_dir / f"site_{i}"
        with rasterio.open(site / "layers" / f"{slrm_name}.tif") as src:
            arr = src.read(1)
            for f in _labels(site)["features"]:
                if f["properties"]["class"] != "mound":
                    continue
                x, y = to_internal.transform(*_center(f))
                row, col = src.index(x, y)
                checked += 1
                if arr[row, col] >= 0.3 * f["properties"]["height_m"]:
                    passed += 1

    assert checked > 0
    assert passed / checked >= 0.9, f"{passed}/{checked}"
