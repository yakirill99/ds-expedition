"""Сколько объектов разметки лежит вне сетки слоёв участка.

Таргеты вне сетки отбрасываются при растрировании молча. У 180_Гнёздово_Лесная_2
так теряется 33 из 483 объектов (6.8%): сетку задаёт bbox LAS, а разметка шире.
Это потолок recall для участка, а не ошибка сборки.

    uv run python scripts/check_label_bounds.py
    uv run python scripts/check_label_bounds.py --only 180,001,035
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio
from pyproj import Transformer

from expds.pipeline_config import load_pipeline_config


def _points(coords: object):
    """Все (x, y) из geometry.coordinates любой вложенности."""
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        yield coords[0], coords[1]
    else:
        for part in coords:
            yield from _points(part)


def main() -> int:
    """Точка входа."""
    ap = argparse.ArgumentParser(description="Разметка вне сетки слоёв")
    ap.add_argument("--config", type=Path, default=Path("configs/dataset_real.yaml"))
    ap.add_argument("--only", default="", help="номера участков: 180,001")
    args = ap.parse_args()

    cfg = load_pipeline_config(args.config)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    rows = []
    for site in sorted(Path(cfg.data.sites_dir).iterdir()):
        if not site.is_dir():
            continue
        if only and site.name[:3] not in only:
            continue
        dtm = site / "layers" / "dtm.tif"
        labels = site / "labels.geojson"
        if not dtm.exists() or not labels.exists():
            continue

        with rasterio.open(dtm) as ds:
            b, crs = ds.bounds, ds.crs
        feats = json.loads(labels.read_text(encoding="utf-8")).get("features") or []
        tr = Transformer.from_crs(cfg.data.labels_crs, crs, always_xy=True)

        inside = outside = 0
        for f in feats:
            xy = list(_points((f.get("geometry") or {}).get("coordinates") or []))
            if not xy:
                continue
            x, y = tr.transform(sum(p[0] for p in xy) / len(xy), sum(p[1] for p in xy) / len(xy))
            if b.left <= x <= b.right and b.bottom <= y <= b.top:
                inside += 1
            else:
                outside += 1
        total = inside + outside
        rows.append((site.name, inside, outside, outside / total if total else 0.0))

    if not rows:
        print("нечего проверять: нет участков с layers/dtm.tif и labels.geojson")
        return 0

    print(f"{'участок':38} {'внутри':>7} {'вне':>5} {'потеря':>7}")
    for name, ins, out, frac in rows:
        mark = "  <--" if frac > 0.02 else ""
        print(f"{name:38} {ins:7d} {out:5d} {frac:7.1%}{mark}")

    bad = [r for r in rows if r[3] > 0.02]
    loss = sum(r[2] for r in rows)
    print(
        f"\nучастков {len(rows)} | объектов вне сетки {loss} | участков с потерей >2%: {len(bad)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
