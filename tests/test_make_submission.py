import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from expds.models import ModelConfig, build_model, save_checkpoint
from expds.pipeline_config import load_pipeline_config
from expds.submission import build_submission

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "dataset_synthetic.yaml"


def _make_synthetic():
    spec = importlib.util.spec_from_file_location(
        "make_synthetic", ROOT / "tools" / "make_synthetic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _weights(tmp_path, channels=None):
    cfg = load_pipeline_config(CONFIG)
    channels = channels or cfg.data.channels
    mcfg = ModelConfig(arch="stub")
    model = build_model(mcfg, len(channels), len(cfg.targets.classes))
    path = tmp_path / "w.pt"
    save_checkpoint(path, model, mcfg, channels, cfg.targets.classes, extra={"epoch": 1})
    return path, cfg


def test_build_submission_smoke(tmp_path):
    sites = tmp_path / "sites"
    _make_synthetic().main(
        ["--config", str(CONFIG), "--n-sites", "1", "--size", "256", "--seed", "1",
         "--out-dir", str(sites)]
    )  # fmt: skip
    w, cfg = _weights(tmp_path)
    res = build_submission(CONFIG, w, tmp_path / "sub.zip", smoke_data=sites / "site_1")
    with zipfile.ZipFile(res["zip"]) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("inference/MANIFEST.json"))
    for n in (
        "entrypoint.py",
        "config.yaml",
        "weights.pt",
        "expds/predict.py",
        "expds/__init__.py",
    ):
        assert f"inference/{n}" in names
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert manifest["classes"] == list(cfg.targets.classes)
    assert manifest["checkpoint_extra"]["epoch"] == 1
    assert res["smoke"]["n_features"] >= 0 and res["smoke"]["seconds"] > 0


def test_size_limit(tmp_path):
    w, _ = _weights(tmp_path)
    with pytest.raises(ValueError, match="лимита"):
        build_submission(CONFIG, w, tmp_path / "sub.zip", max_bytes=1000)


def test_channel_mismatch(tmp_path):
    w, _ = _weights(tmp_path, channels=tuple(f"c{i}" for i in range(19)))
    with pytest.raises(ValueError, match="channel_names"):
        build_submission(CONFIG, w, tmp_path / "sub.zip")
