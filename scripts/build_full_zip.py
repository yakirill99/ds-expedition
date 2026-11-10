"""Архив: solution.py + metadata + config.yaml + weights.pt (+ expds/ по --with-code).

ЭТО НЕ АРХИВ ДЛЯ ПЛАТФОРМЫ. Платформа (Приложение 2 регламента) ждёт zip
ровно из двух файлов в корне: solution.py + metadata.json; заливка этого
архива = отклонение сабмита. Он для локального прогона распакованного
архива (порядок поиска в solution.py: _HERE -> pkg_dir.parent -> /app).
Для платформы: scripts/build_platform_solution.py.

_config.yaml из архива перебивает /app/config.yaml, weights.pt — образ.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import yaml

from expds.models.checkpoint import load_checkpoint
from expds.pipeline_config import load_pipeline_config

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--weights", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--min-area", type=float, default=32.0)
ap.add_argument("--post", default="{}", help="JSON поверх post")
ap.add_argument("--with-code", action="store_true", help="положить expds/ в архив")
ap.add_argument(
    "--allow-for-platform",
    action="store_true",
    help="заглушить предупреждение о неплатформенном формате архива",
)
a = ap.parse_args()

if not a.allow_for_platform:
    print(
        "ВНИМАНИЕ: этот архив НЕ для платформы (4 файла в корне вместо двух — "
        "платформа его отклонит). Для платформы: scripts/build_platform_solution.py. "
        "Заглушить: --allow-for-platform",
        file=sys.stderr,
        flush=True,
    )

cfg = load_pipeline_config(a.run / "config.yaml")
load_checkpoint(a.weights, channel_names=cfg.data.channels, classes=cfg.targets.classes)
print(f"веса совместимы: {a.weights} ({a.weights.stat().st_size / 1e6:.0f} МБ)")

raw = yaml.safe_load((a.run / "config.yaml").read_text(encoding="utf-8"))
raw["post"]["min_area_px"] = a.min_area
raw["post"].update(json.loads(a.post))
tmp = a.out.parent / "_config.yaml"
a.out.parent.mkdir(parents=True, exist_ok=True)
tmp.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

src = Path("inference/solution.py").read_text(encoding="utf-8")
compile(src, "solution.py", "exec")

with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("solution.py", src)
    zf.writestr(
        "metadata.json",
        json.dumps(
            {"image": "yakirill99/ds-expedition:latest", "entry_point": "python solution.py"}
        ),
    )
    zf.write(tmp, "config.yaml")
    zf.write(a.weights, "weights.pt")
    if a.with_code:
        for f in sorted(Path("src/expds").rglob("*.py")):
            zf.write(f, str(f.relative_to("src")))

names = zipfile.ZipFile(a.out).namelist()
print(f"{a.out}: {a.out.stat().st_size / 1e6:.1f} МБ, файлов {len(names)}")
print("корень:", [n for n in names if "/" not in n])
assert "weights.pt" in names and "config.yaml" in names and "solution.py" in names
print(f"post.min_area_px = {raw['post']['min_area_px']}")
