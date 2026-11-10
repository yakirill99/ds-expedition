"""rgb_* для участков, где сборка не прошла: причина + пересборка в sites_dir.

uv run python scripts/rgb_rebuild.py 005_Боровичи_2      # один участок
uv run python scripts/rgb_rebuild.py                     # все train/val без rgb
"""

import inspect
import sys
import traceback
from pathlib import Path

from expds.data.build_raster import RGB_PREFIXES, build_site_rasters
from expds.pipeline_config import load_pipeline_config
from expds.tiles.dataset import load_site

cfg = load_pipeline_config(Path("configs/night.yaml"))
RAW, RB = Path("data/raw"), Path(cfg.data.sites_dir)

sig = inspect.signature(build_site_rasters)
print("build_site_rasters", sig)
kw = {}
mp = sig.parameters.get("max_pixels")
if mp is not None and mp.default is inspect.Parameter.empty:
    kw["max_pixels"] = 200_000_000
if "log" in sig.parameters:
    kw["log"] = lambda *a: print("   ", *a)

names = sys.argv[1:] or [
    n
    for n in list(cfg.train.train_sites) + list(cfg.train.val_sites)
    if not (RB / n / "layers" / "rgb_r.tif").is_file()
    and (RAW / n).is_dir()
    and any(q.is_dir() and q.name[:3] in RGB_PREFIXES for q in (RAW / n).iterdir())
]
print("участки:", names)

for name in names:
    rb, raw = RB / name, RAW / name
    print(f"\n=== {name}")
    if not rb.is_dir():
        print("   нет в", RB)
        continue
    if (rb / "layers" / "rgb_r.tif").is_file():
        print("   rgb уже есть, пропуск")
        continue
    srcs = [q for q in raw.iterdir() if q.is_dir() and q.name[:3] in RGB_PREFIXES]
    for q in srcs:
        files = [f for f in q.rglob("*") if f.is_file()]
        print(f"   {q.name}: {len(files)} файлов, расширения {sorted({f.suffix for f in files})}")
        dst = rb / q.name
        if not dst.exists():
            dst.symlink_to(q.resolve(), target_is_directory=True)  # исходники только читаются
    try:
        res = build_site_rasters(rb, cfg.data.labels_crs, **kw)
        print("   результат:", res)
    except Exception:
        traceback.print_exc()
        continue
    if not (rb / "layers" / "rgb_r.tif").is_file():
        print("   НЕ СОБРАН")
        continue
    try:
        site = load_site(
            rb,
            channels=cfg.data.channels,
            classes=cfg.targets.classes,
            heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
            labels_crs=cfg.data.labels_crs,
            labels_file=cfg.data.labels_file,
            normalize=cfg.data.normalize,
        )
        print(f"   OK: каналов {int(site.channel_mask.sum())}/{len(cfg.data.channels)}")
    except Exception as exc:
        print("   собран, но load_site падает:", exc)
