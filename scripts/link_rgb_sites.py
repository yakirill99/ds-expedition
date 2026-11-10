"""Симлинки data/rebuilt/<name> -> data/raw/<name> и чистка configs/night2.yaml.

Имена участков пишутся в конфиг без пути, а load_site всегда смотрит в
data.sites_dir (data/rebuilt). Участки, у которых rgb_* собран только в
data/raw, падали с FileNotFoundError: нет ни sample_meta.json, ни layers/.
"""

import shutil
from pathlib import Path

import yaml

from expds.pipeline_config import load_pipeline_config

cfg = load_pipeline_config(Path("configs/night2.yaml"))
RAW, RB = Path("data/raw"), Path(cfg.data.sites_dir)
names = list(cfg.train.train_sites) + list(cfg.train.val_sites)

ok, linked, bad = [], [], []
for name in names:
    rb, raw = RB / name, RAW / name
    if (rb / "layers" / "rgb_r.tif").is_file() or (rb / "layers" / "dtm.tif").is_file():
        ok.append(name)
        continue
    if not (raw / "layers").is_dir():
        bad.append(name)
        continue
    if rb.is_symlink():
        ok.append(name)
        continue
    if rb.exists():  # пустая папка без слоёв — отодвигаем
        bak = rb.with_name(rb.name + ".bak")
        if bak.is_symlink() or bak.is_file():
            bak.unlink()
        elif bak.is_dir():
            shutil.rmtree(bak)
        rb.rename(bak)
    rb.symlink_to(raw.resolve(), target_is_directory=True)
    linked.append(name)

keep = set(ok) | set(linked)
print(f"уже было: {len(ok)} | слинковано: {len(linked)} | без слоёв: {len(bad)}")
if bad:
    print("выброшены из конфига:", bad[:15])

raw_cfg = yaml.safe_load(Path("configs/night2.yaml").read_text(encoding="utf-8"))
raw_cfg["train"]["train_sites"] = [s for s in raw_cfg["train"]["train_sites"] if s in keep]
raw_cfg["train"]["val_sites"] = [s for s in raw_cfg["train"]["val_sites"] if s in keep]
Path("configs/night2.yaml").write_text(
    yaml.safe_dump(raw_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
)
tr, va = raw_cfg["train"]["train_sites"], raw_cfg["train"]["val_sites"]
assert not (set(tr) & set(va))
print(f"train_sites {len(tr)}, val_sites {len(va)}")
