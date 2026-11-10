"""Диагностика входа: территории, где лежат tif, сколько и каких GT."""

import json
from collections import Counter
from pathlib import Path

from expds.data.build_raster import RGB_PREFIXES
from expds.pipeline_config import load_pipeline_config

cfg = load_pipeline_config(Path("runs/n2_rgb/config.yaml"))
root = Path(cfg.data.sites_dir)
labels_file = cfg.data.labels_file

print("sites_dir:", root, "| labels:", labels_file)
print("каналов:", len(cfg.data.channels))
print("каналы:", list(cfg.data.channels))
print("RGB_PREFIXES:", RGB_PREFIXES)
print("train.sites:", getattr(cfg.train, "sites", "—"))
print("train.val_sites:", getattr(cfg.train, "val_sites", "—"))
print()

hdr = f"{'территория':40} {'tif@кор':>8} {'tif@влож':>9} {'GT':>6}  классы"
print(hdr)
print("-" * 100)
tot = Counter()
for site in sorted(p for p in root.iterdir() if p.is_dir()):
    tifs_root = list(site.glob("*.tif"))
    tifs_nested = list(site.glob("*/*.tif"))
    hist, n_gt = Counter(), 0
    lp = site / labels_file
    if lp.is_file():
        try:
            d = json.loads(lp.read_text(encoding="utf-8"))
            for f in d.get("features", []):
                pr = f.get("properties") or {}
                cls = pr.get("class") or pr.get("class_name") or "?"
                hist[str(cls)] += 1
                n_gt += 1
        except Exception as exc:
            hist[f"ОШИБКА:{type(exc).__name__}"] += 1
    tot.update(hist)
    print(f"{site.name:40} {len(tifs_root):8} {len(tifs_nested):9} {n_gt:6}  {dict(hist)}")

print("\nИТОГО по классам:", dict(tot), "| всего", sum(tot.values()))
