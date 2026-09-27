import json

import numpy as np
import pytest
import torch

from expds.eval import objects_from_detections
from expds.features.grid import Grid
from expds.infer import detect, predict_probs
from expds.models import ModelConfig, build_model, load_checkpoint
from expds.pipeline_config import DataConfig, PipelineConfig, TargetsConfig, TilesConfig
from expds.tiles.dataset import SiteData
from expds.train import TrainConfig, fit
from expds.utils.seed import seed_everything

CLASSES = ("mound", "ditch")
CH = ("a", "b", "c")


def _site(name, seed, h=64, w=80):
    rng = np.random.default_rng(seed)
    grid = Grid(
        crs="EPSG:32637", pixel_size=1.0, x_min=500000.0, y_min=6200000.0, width=w, height=h
    )
    rr, cc = np.mgrid[:h, :w]
    y_seg = np.zeros((2, h, w), np.float32)
    y_hm = np.zeros((2, h, w), np.float32)
    for _ in range(3):
        r, c = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
        y_seg[0] = np.maximum(y_seg[0], np.hypot(rr - r, cc - c) <= 5)
        r, c = rng.uniform(4, h - 4), rng.uniform(4, w - 4)
        y_hm[1] = np.maximum(y_hm[1], np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 8.0))
    x = np.stack([2 * y_seg[0], 2 * y_hm[1], np.zeros((h, w))]).astype(np.float32)
    x += rng.normal(0, 0.05, x.shape).astype(np.float32)
    return SiteData(
        name=name,
        grid=grid,
        x=x,
        channel_names=CH,
        channel_mask=np.ones(3, bool),
        valid=np.ones((h, w), bool),
        y_seg=y_seg,
        y_hm=y_hm,
    )


def _cfg(tmp_path, **train):
    kw = {"epochs": 5, "batch_size": 4, "lr": 0.02, "device": "cpu", "num_workers": 0}
    kw.update(amp=False, **train)
    return PipelineConfig(
        data=DataConfig(
            sites_dir=tmp_path, labels_file="labels.geojson", channels=CH, normalize="none"
        ),
        targets=TargetsConfig(classes=CLASSES, heatmap_sigma_px=2.0),
        tiles=TilesConfig(tile_px=32, overlap_px=8, pos_fraction=0.5),
        model=ModelConfig(arch="stub"),
        train=TrainConfig(**kw),
    )


def test_seed_everything():
    seed_everything(1)
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(1)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_train_config_from_dict():
    c = TrainConfig.from_dict({"lr": "1e-3", "train_sites": ["s1", "s2"], "val_sites": ["s3"]})
    assert c.lr == 1e-3 and c.train_sites == ("s1", "s2")
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"epocs": 3})
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"train_sites": ["s1"], "val_sites": ["s1"]})


def test_predict_probs_matches_full_forward():
    """Pointwise-модель: сшитые тайлы == прогон целого участка (геометрия окон и паддинга)."""
    site = _site("s", 0)
    torch.manual_seed(0)
    m = build_model(ModelConfig(arch="stub", stub_kernel=1), len(CH), len(CLASSES))
    probs = predict_probs(m, site, len(CLASSES), tile_px=32, overlap_px=8, batch_size=3)
    with torch.no_grad():
        ref = torch.sigmoid(m(torch.from_numpy(site.x)[None]))[0].numpy()
    assert probs.shape == (4, 64, 80)
    assert np.allclose(probs, ref, atol=1e-5)
    assert m.training  # режим восстановлен


def test_fit_smoke(tmp_path):
    cfg = _cfg(tmp_path)
    val = _site("v", 3)
    gt_probs = np.concatenate([val.y_seg, val.y_hm])
    gts = objects_from_detections(detect(gt_probs, val, CLASSES, cfg.post))
    assert len(gts) >= 2
    run = tmp_path / "run"
    res = fit(cfg, [_site("t1", 1), _site("t2", 2)], [(val, gts)], run, eval_crs=val.grid.crs)
    assert res["oracle"]["f1"] == 1.0  # GT построены той же цепочкой
    h = res["history"]
    assert len(h) == 5
    assert h[-1]["train_loss"] < h[0]["train_loss"]
    assert 0.0 <= h[-1]["val_f1"] <= 1.0
    lines = (run / "log.jsonl").read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["epoch"] == 5
    m, meta = load_checkpoint(run / "best.pt", channel_names=CH, classes=CLASSES)
    assert 1 <= meta["extra"]["epoch"] <= 5
    assert (run / "last.pt").exists()
