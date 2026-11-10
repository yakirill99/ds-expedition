"""Официальная метрика на val-участках прогона для одного чекпоинта."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from expds.infer import detect, predict_probs, resolve_device
from expds.io.geojson import write_detections
from expds.models.checkpoint import load_checkpoint
from expds.pipeline_config import load_pipeline_config
from expds.tiles.dataset import load_site


def fix_fields(src: Path, dst: Path, region: str) -> None:
    d = json.loads(src.read_text(encoding="utf-8"))
    for f in d["features"]:
        p = f.setdefault("properties", {})
        p["class_name"] = p.get("class_name") or p.get("class")
        p["region_name"] = p.get("region_name") or region
        p["markup_type"] = p.get("markup_type") or "лидар"
    dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def score(ckpt: Path, run: Path, sites: list[str] | None = None) -> float:
    cfg = load_pipeline_config(run / "config.yaml")
    dev = resolve_device("auto")
    k = len(cfg.targets.classes)
    model, _ = load_checkpoint(ckpt, channel_names=cfg.data.channels, classes=cfg.targets.classes)
    model = model.to(dev).eval()
    targets = sites or list(cfg.train.val_sites)
    tmp = Path(tempfile.mkdtemp())

    dets, feats = [], []
    for name in targets:
        site = load_site(
            cfg.data.sites_dir / name,
            channels=cfg.data.channels,
            classes=cfg.targets.classes,
            heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
            labels_crs=cfg.data.labels_crs,
            labels_file=cfg.data.labels_file,
            normalize=cfg.data.normalize,
        )
        probs = predict_probs(
            model,
            site,
            k,
            cfg.tiles.tile_px,
            cfg.tiles.overlap_px,
            cfg.stitch,
            dev,
            cfg.train.batch_size,
            cfg.train.amp,
        )
        dets += detect(probs, site, cfg.targets.classes, cfg.post)
        d = json.loads(
            (cfg.data.sites_dir / name / cfg.data.labels_file).read_text(encoding="utf-8")
        )
        for f in d["features"]:
            f.setdefault("properties", {})["region_name"] = name
            feats.append(f)

    write_detections(tmp / "p.geojson", dets, cfg.data.labels_crs, "EPSG:3857")
    (tmp / "g.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
        encoding="utf-8",
    )
    fix_fields(tmp / "p.geojson", tmp / "p_fix.geojson", "val")
    fix_fields(tmp / "g.geojson", tmp / "g_fix.geojson", "gt")

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
    if not nums:
        print(f"[{ckpt}] метрика не ответила: {out.stderr.strip()[-300:]}", flush=True)
        return 0.0
    return float(nums[-1])


if __name__ == "__main__":
    ckpt = Path(sys.argv[1])
    run = Path(sys.argv[2]) if len(sys.argv) > 2 else ckpt.parent
    print(f"{ckpt.name}: {score(ckpt, run):.6f}", flush=True)
