from pathlib import Path

import pytest
import yaml

from expds.pipeline_config import load_pipeline_config

SYNTH = Path("configs/dataset_synthetic.yaml")


def _dump(tmp_path, raw):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return p


def _raw():
    return yaml.safe_load(SYNTH.read_text(encoding="utf-8"))


def test_synthetic_sections():
    cfg = load_pipeline_config(SYNTH)
    assert cfg.model.arch == "unet"
    assert cfg.loss.hm_weight == 1.0
    assert cfg.stitch.window == "hann"
    assert cfg.post.seg_thr == 0.5
    assert cfg.eval.r_tol_units in ("crs", "meters")
    assert cfg.train.train_sites and cfg.train.val_sites
    assert cfg.data.labels_crs == "EPSG:3857"


def test_optional_sections_default(tmp_path):
    raw = _raw()
    for k in ("model", "loss", "stitch", "post", "eval", "train"):
        raw.pop(k, None)
    cfg = load_pipeline_config(_dump(tmp_path, raw))
    assert cfg.model.arch == "unet" and cfg.stitch.window == "hann"


def test_tile_divisor(tmp_path):
    raw = _raw()
    raw["tiles"]["tile_px"] = 250
    raw["model"] = {"arch": "unet"}
    with pytest.raises(ValueError, match="делится"):
        load_pipeline_config(_dump(tmp_path, raw))
    raw["model"] = {"arch": "stub"}
    assert load_pipeline_config(_dump(tmp_path, raw)).tiles.tile_px == 250


@pytest.mark.parametrize("key", ["model", "loss", "stitch", "post", "eval", "train"])
def test_unknown_key_rejected(tmp_path, key):
    raw = _raw()
    raw[key] = {**(raw.get(key) or {}), "typo_key": 1}
    with pytest.raises(ValueError, match="unknown keys"):
        load_pipeline_config(_dump(tmp_path, raw))
