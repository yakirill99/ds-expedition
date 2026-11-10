"""Потолок постобработки: oracle и модель при разных min_area_px.

Вероятности считаются один раз, затем перебирается только post.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from expds.infer import detect, predict_probs, resolve_device
from expds.io.geojson import write_detections
from expds.models.checkpoint import load_checkpoint
from expds.pipeline_config import load_pipeline_config
from expds.tiles.dataset import load_site

cfg = load_pipeline_config(Path("runs/n2_rgb/config.yaml"))
dev = resolve_device("auto")
tmp = Path(tempfile.mkdtemp())

sites, gts, feats = [], [], []
for name in cfg.train.val_sites:
    site = load_site(
        cfg.data.sites_dir / name,
        channels=cfg.data.channels,
        classes=cfg.targets.classes,
        heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
        labels_crs=cfg.data.labels_crs,
        labels_file=cfg.data.labels_file,
        normalize=cfg.data.normalize,
    )
    sites.append(site)
    d = json.loads((cfg.data.sites_dir / name / cfg.data.labels_file).read_text(encoding="utf-8"))
    for f in d["features"]:
        f.setdefault("properties", {})["region_name"] = name
        feats.append(f)
print(f"val: {len(sites)} участков, GT {len(feats)}", flush=True)

(tmp / "g.geojson").write_text(
    json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
    encoding="utf-8",
)
gd = json.loads((tmp / "g.geojson").read_text(encoding="utf-8"))
for f in gd["features"]:
    p = f.setdefault("properties", {})
    p["class_name"] = p.get("class_name") or p.get("class")
    p["markup_type"] = p.get("markup_type") or "лидар"
(tmp / "g_fix.geojson").write_text(json.dumps(gd, ensure_ascii=False), encoding="utf-8")

# вероятности один раз
oracle_probs = [np.concatenate([s.y_seg, s.y_hm]).astype(np.float32) for s in sites]

k = len(cfg.targets.classes)
model, _ = load_checkpoint(
    Path("runs/n2_rgb/last.pt"), channel_names=cfg.data.channels, classes=cfg.targets.classes
)
model = model.to(dev).eval()
model_probs = [
    predict_probs(
        model,
        s,
        k,
        cfg.tiles.tile_px,
        cfg.tiles.overlap_px,
        cfg.stitch,
        dev,
        cfg.train.batch_size,
        cfg.train.amp,
    )
    for s in sites
]
print("вероятности готовы", flush=True)


def score(probs_list, post):
    dets = []
    for site, probs in zip(sites, probs_list, strict=False):
        dets += detect(probs, site, cfg.targets.classes, post)
    write_detections(tmp / "p.geojson", dets, cfg.data.labels_crs, "EPSG:3857")
    d = json.loads((tmp / "p.geojson").read_text(encoding="utf-8"))
    for f in d["features"]:
        p = f.setdefault("properties", {})
        p["class_name"] = p.get("class_name") or p.get("class")
        p["region_name"] = p.get("region_name") or "val"
        p["markup_type"] = p.get("markup_type") or "лидар"
    (tmp / "p_fix.geojson").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    out = subprocess.run(
        [
            sys.executable,
            "compute_metrics.py",
            "-p",
            str(tmp / "p_fix.geojson"),
            "-g",
            str(tmp / "g_fix.geojson"),
        ],
        capture_output=True,
        text=True,
    )
    nums = [s for s in out.stdout.strip().splitlines() if s.replace(".", "").isdigit()]
    return float(nums[-1]) if nums else 0.0, len(dets)


print(f"\n{'min_area_px':12} {'seg_thr':8} {'oracle':>9} {'model':>9} {'дет':>6}")
for min_area in (4, 8, 16, 32, 64):
    for seg_thr in (0.3, 0.5):
        post = replace(cfg.post, min_area_px=float(min_area), seg_thr=seg_thr)
        so, _ = score(oracle_probs, post)
        sm, nd = score(model_probs, post)
        print(f"{min_area:<12} {seg_thr:<8} {so:9.6f} {sm:9.6f} {nd:6d}", flush=True)
