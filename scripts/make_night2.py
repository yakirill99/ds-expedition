"""night2.yaml: обучение с RGB-участками, val без пересечения с train.

Участки без dtm (только rgb) грузятся в память как 31 канал, из них 3 реальных.
"""

from pathlib import Path

import yaml

from expds.eval.metric import read_objects
from expds.pipeline_config import load_pipeline_config

cfg = load_pipeline_config(Path("configs/night.yaml"))
RAW, RB = Path("data/raw"), Path(cfg.data.sites_dir)
used = set(cfg.train.train_sites) | set(cfg.train.val_sites)


def usable(name: str) -> str | None:
    for base in (RB / name, RAW / name):
        if (base / "layers" / "rgb_r.tif").is_file() and (base / cfg.data.labels_file).is_file():
            return str(base)
    return None


cand = []
for d in sorted(RAW.iterdir()):
    if not d.is_dir() or d.name in used:
        continue
    base = usable(d.name)
    if base is None:
        continue
    try:
        n = len(read_objects(Path(base) / cfg.data.labels_file, cfg.data.labels_crs))
    except Exception:
        n = 0
    if n == 0:
        continue
    cand.append((n, d.name, base))

# Память: тензор участка = 31 * H * W * 4 байт. Считаем по реальному размеру.
import rasterio


def mb(base: str) -> float:
    with rasterio.open(Path(base) / "layers" / "rgb_r.tif") as r:
        return 31 * r.height * r.width * 4 / 1e6


cand.sort(reverse=True)  # больше объектов — раньше
print(f"кандидатов с RGB и GT: {len(cand)}")

MAX_SITES, MAX_MB = 60, 120.0
extra, skipped = [], []
for n, name, base in cand:
    m = mb(base)
    if m > MAX_MB:
        skipped.append((name, round(m)))
        continue
    if len(extra) >= MAX_SITES:
        break
    extra.append(name)
print(f"в train берём {len(extra)}, пропущено по размеру: {skipped[:5]}")

holdout = [c[1] for c in cand if c[1] in extra][:12]

raw = yaml.safe_load(Path("configs/night.yaml").read_text(encoding="utf-8"))
raw["train"]["train_sites"] = (
    list(cfg.train.train_sites) + list(cfg.train.val_sites) + [s for s in extra if s not in holdout]
)
raw["train"]["val_sites"] = holdout  # только новые, старых здесь нет
raw["train"]["epochs"] = 60
raw["train"]["lr"] = 0.0003
raw["model"]["prior_prob"] = 0.05
raw["loss"]["focal_alpha"] = 0.75
raw["loss"]["hm_weight"] = 0.3
Path("configs/night2.yaml").write_text(
    yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
)

tr, va = raw["train"]["train_sites"], raw["train"]["val_sites"]
assert not (set(tr) & set(va)), "пересечение train/val"
print(f"train_sites {len(tr)}, val_sites {len(va)}, пересечение: нет")
print("RAM на загрузку, ГБ ~", round(sum(mb(c[2]) for c in cand if c[1] in tr) / 1000, 1))
