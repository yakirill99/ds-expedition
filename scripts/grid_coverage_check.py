"""Аудит покрытия: сколько разметки каждого участка попадает в сетку.

Ответ на вопрос, откуда в оракуле «вне сетки» (sweep 2026-11-11: 456 из
461 FN — объекты GT за пределами растровой сетки участка). Скрипт не тянет
torch: только yaml + pyproj + json.

Для каждого участка:
  - сетка из samples/sample_meta.json, иначе из первого tif в layers/;
  - разметка labels.geojson (labels_crs, обычно EPSG:3857) репроецируется
    в CRS сетки;
  - каждый полигон классифицируется: целиком вне / частично / внутри;
  - печатается смещение bbox разметки относительно bbox сетки (диагностика
    систематического сдвига/чужого CRS) и рекомендуемая сетка, покрывающая
    и данные, и разметку.

Примеры:
    uv run python scripts/grid_coverage_check.py --config configs/night3.yaml
    uv run python scripts/grid_coverage_check.py --sites-dir data/rebuilt \
        --labels-crs EPSG:3857 --sites 036_Тимерево,066_Московиты
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from pyproj import Transformer

_META_PATH = "sample_meta.json"
_LAYERS_DIR = "layers"


def _read_grid(site_dir: Path):
    meta = site_dir / _META_PATH
    if meta.exists():
        g = json.loads(meta.read_text(encoding="utf-8"))["grid"]
        return g["crs"], g["pixel_size"], g["x_min"], g["y_min"], g["width"], g["height"]
    tifs = sorted((site_dir / _LAYERS_DIR).glob("*.tif"))
    if not tifs:
        return None
    from expds.data.raster import read_raster

    lay = read_raster(tifs[0])
    g = lay.grid
    return g.crs, g.pixel_size, g.x_min, g.y_min, g.width, g.height


def _iter_polys(fc: dict):
    """Полигоны (внешние кольца) из FeatureCollection; None-геометрии пропускаем."""
    for f in fc.get("features", []):
        geom = f.get("geometry")
        if not geom:
            continue
        t = geom.get("type")
        if t == "Polygon":
            yield geom["coordinates"][0]
        elif t == "MultiPolygon":
            for poly in geom["coordinates"]:
                yield poly[0]


def audit_site(site_dir: Path, labels_crs: str):
    g = _read_grid(site_dir)
    if g is None:
        return None
    crs, ps, x_min, y_min, w, h = g
    x_max, y_max = x_min + w * ps, y_min + h * ps
    lp = site_dir / "labels.geojson"
    if not lp.exists():
        return {"site": site_dir.name, "error": "нет labels.geojson"}
    fc = json.loads(lp.read_text(encoding="utf-8"))
    tr = Transformer.from_crs(labels_crs, crs, always_xy=True)

    n = out = partial = 0
    lx_min = ly_min = math.inf
    lx_max = ly_max = -math.inf
    for ring in _iter_polys(fc):
        n += 1
        xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
        lx_min, lx_max = min(lx_min, min(xs)), max(lx_max, max(xs))
        ly_min, ly_max = min(ly_min, min(ys)), max(ly_max, max(ys))
        inside = sum(
            1 for x, y in zip(xs, ys, strict=False) if x_min <= x <= x_max and y_min <= y <= y_max
        )
        if inside == 0:
            out += 1
        elif inside < len(xs):
            partial += 1

    # Рекомендуемая сетка: объединение bbox данных и bbox разметки, тот же px.
    u_x_min, u_y_min = min(x_min, lx_min), min(y_min, ly_min)
    u_x_max, u_y_max = max(x_max, lx_max), max(y_max, ly_max)
    u_w = math.ceil((u_x_max - u_x_min) / ps)
    u_h = math.ceil((u_y_max - u_y_min) / ps)
    return {
        "site": site_dir.name,
        "crs": crs,
        "px": ps,
        "grid": f"{w}x{h}",
        "labels": n,
        "outside": out,
        "partial": partial,
        "label_bbox": (lx_min, ly_min, lx_max, ly_max),
        "grid_bbox": (x_min, y_min, x_max, y_max),
        "union_grid": (round(u_x_min, 1), round(u_y_min, 1), u_w, u_h),
        "union_mpx": round(u_w * u_h * ps * ps / 1e6, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="YAML конфига (data.sites_dir, data.labels_crs)")
    ap.add_argument("--sites-dir", type=Path)
    ap.add_argument("--labels-crs", type=str, default="EPSG:3857")
    ap.add_argument("--sites", type=str, help="имена через запятую; иначе все dir в sites-dir")
    args = ap.parse_args(argv)

    sites_dir = args.sites_dir
    if args.config:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        sites_dir = sites_dir or Path(cfg["data"]["sites_dir"])
        args.labels_crs = cfg["data"].get("labels_crs", args.labels_crs)
    if not sites_dir:
        ap.error("укажите --config или --sites-dir")

    names = (
        [s.strip() for s in args.sites.split(",") if s.strip()]
        if args.sites
        else sorted(d.name for d in Path(sites_dir).iterdir() if d.is_dir())
    )

    rows = [r for name in names if (r := audit_site(Path(sites_dir) / name, args.labels_crs))]
    bad = [r for r in rows if r.get("outside") or r.get("error") or r.get("labels") == 0]
    rows = [r for r in rows if r not in bad]

    print(f"{'участок':<44} {'сетка':>11} {'GT':>4} {'вне':>4} {'част':>5}  crs")
    for r in rows:
        print(
            f"{r['site']:<44} {r['grid']:>11} {r['labels']:>4} {r['outside']:>4} "
            f"{r['partial']:>5}  {r['crs']}"
        )
    if bad:
        print(f"\n### Проблемные ({len(bad)}):")
        for r in bad:
            if "error" in r:
                print(f"  {r['site']:<44} {r['error']}")
                continue
            dx = r["label_bbox"][0] - r["grid_bbox"][0]
            dy = r["label_bbox"][1] - r["grid_bbox"][1]
            print(
                f"  {r['site']:<44} {r['grid']:>11} GT {r['labels']:>4} вне {r['outside']:>4} "
                f"част {r['partial']:>3}  {r['crs']}  Δx={dx:.0f}м Δy={dy:.0f}м"
            )
            ux, uy, uw, uh = r["union_grid"]
            print(
                f"      recommended Grid(crs='{r['crs']}', pixel_size={r['px']}, "
                f"x_min={ux}, y_min={uy}, width={uw}, height={uh})  # {r['union_mpx']} Мпкс"
            )
    total_out = sum(r.get("outside", 0) for r in rows) + sum(r.get("outside", 0) for r in bad)
    total_gt = sum(r.get("labels", 0) for r in rows) + sum(r.get("labels", 0) for r in bad)
    print(
        f"\nИТОГО: GT {total_gt}, вне сетки {total_out} "
        f"({100 * total_out / max(total_gt, 1):.1f}%) на {len(rows) + len(bad)} участках"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
