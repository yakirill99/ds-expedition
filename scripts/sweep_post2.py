"""Этап 2 перебора post на весах n2_rgb: размер точки, hm_thr, NMS, min_area."""

import itertools
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from expds.infer import detect, predict_probs, resolve_device
from expds.io.geojson import write_detections
from expds.models.checkpoint import load_checkpoint
from expds.pipeline_config import load_pipeline_config
from expds.tiles.dataset import load_site

RUN = Path("runs/n2_rgb")
cfg = load_pipeline_config(RUN / "config.yaml")
dev = resolve_device("auto")
tmp = Path(tempfile.mkdtemp())


def fix(fc, region):
    for f in fc["features"]:
        p = f.setdefault("properties", {})
        p["class_name"] = p.get("class_name") or p.get("class")
        p["region_name"] = p.get("region_name") or region
        p["markup_type"] = p.get("markup_type") or "лидар"
    return fc


sites, feats = [], []
for name in cfg.train.val_sites:
    sites.append(
        load_site(
            cfg.data.sites_dir / name,
            channels=cfg.data.channels,
            classes=cfg.targets.classes,
            heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
            labels_crs=cfg.data.labels_crs,
            labels_file=cfg.data.labels_file,
            normalize=cfg.data.normalize,
        )
    )
    d = json.loads((cfg.data.sites_dir / name / cfg.data.labels_file).read_text(encoding="utf-8"))
    for f in d["features"]:
        f.setdefault("properties", {})["region_name"] = name
        feats.append(f)
(tmp / "g.geojson").write_text(
    json.dumps(fix({"type": "FeatureCollection", "features": feats}, "gt"), ensure_ascii=False),
    encoding="utf-8",
)

model, _ = load_checkpoint(
    RUN / "last.pt", channel_names=cfg.data.channels, classes=cfg.targets.classes
)
model = model.to(dev).eval()
k = len(cfg.targets.classes)
probs = [
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
print(f"вероятности готовы: {len(probs)} участков", flush=True)


def score(post):
    dets = []
    for site, pr in zip(sites, probs, strict=False):
        dets += detect(pr, site, cfg.targets.classes, post)
    write_detections(tmp / "p.geojson", dets, cfg.data.labels_crs, "EPSG:3857")
    d = fix(json.loads((tmp / "p.geojson").read_text(encoding="utf-8")), "val")
    (tmp / "p_fix.geojson").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    out = subprocess.run(
        [
            sys.executable,
            "compute_metrics.py",
            "-p",
            str(tmp / "p_fix.geojson"),
            "-g",
            str(tmp / "g.geojson"),
        ],
        capture_output=True,
        text=True,
    )
    nums = [s for s in out.stdout.strip().splitlines() if s.replace(".", "").isdigit()]
    return float(nums[-1]) if nums else 0.0, len(dets)


best = [0.0, {}]


def run(label, **kw):
    s, n = score(replace(cfg.post, **kw))
    mark = ""
    if s > best[0]:
        best[:] = [s, kw]
        mark = "  <-- лучший"
    print(f"{label:44} {s:.6f}  дет {n:5d}{mark}", flush=True)


print("\n--- этап 1: point_radius_px x min_area_px (seg_thr 0.5) ---")
for pr, ma in itertools.product((4.0, 8.0, 12.0, 16.0), (16.0, 32.0, 48.0)):
    run(f"radius {pr:4.0f} min_area {ma:4.0f}", point_radius_px=pr, min_area_px=ma, seg_thr=0.5)

base = dict(best[1])
print(f"\n--- этап 2: hm_thr x nms_radius_px (лучший этап 1: {base}) ---")
for hm, nms in itertools.product((0.3, 0.5, 0.7, 0.85), (6.0, 12.0, 20.0, 30.0)):
    run(f"hm {hm:.2f} nms {nms:4.0f}", hm_thr=hm, nms_radius_px=nms, **base)

print(f"\nИТОГО: {best[0]:.6f} при {best[1]}")
