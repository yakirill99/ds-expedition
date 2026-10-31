"""CLI к configs/sites.yaml: какие участки входят в набор.

    uv run python scripts/sites.py list train            # имена, по одному в строке
    uv run python scripts/sites.py list train --paths    # пути data/raw/...
    uv run python scripts/sites.py show                  # сводка + проверка утечек
    uv run python scripts/sites.py table                 # каждый участок

В bash:
    for s in $(uv run python scripts/sites.py list todo_optic --paths); do
        uv run python scripts/build_layers_raster.py "$s"
    done
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from expds.data.site_sets import DEFAULT_MANIFEST, SPLITS, SiteInfo, load_manifest, scan, select


def _obj(sites: list[SiteInfo]) -> int:
    return sum(s.n_objects for s in sites)


def _show(sites: list[SiteInfo], manifest: dict) -> int:
    sets = manifest.get("sets") or {}
    print(f"{'набор':12} {'участков':>8} {'объектов':>9}   lidar optic   geo")
    chosen = {}
    for name, spec in sets.items():
        chosen[name] = select(sites, spec)
        k = Counter(s.kind for s in chosen[name])
        print(
            f"{name:12} {len(chosen[name]):8d} {_obj(chosen[name]):9d}   "
            f"{k['lidar']:5d} {k['optic']:5d} {k['geo']:5d}"
        )

    labeled = [s for s in sites if s.has("labels")]
    total = _obj(labeled) or 1
    print(f"\nсплит по всем размеченным ({len(labeled)} участков, {total} объектов):")
    for sp in SPLITS:
        part = [s for s in labeled if s.split == sp]
        share = _obj(part) / total
        print(f"  {sp:5}  участков {len(part):4d}  объектов {_obj(part):6d} ({share:5.1%})")

    groups: dict[str, list[SiteInfo]] = defaultdict(list)
    for s in sites:
        groups[s.group].append(s)
    multi = {g: v for g, v in groups.items() if len(v) > 1}
    if multi:
        print("\nгруппы из нескольких участков:")
        for g, members in sorted(multi.items()):
            n_obj = _obj(members)
            print(f"  {g:20} {members[0].split:5}  участков {len(members):3d}  объектов {n_obj:5d}")

    bad = 0
    for g, members in groups.items():
        if len({s.split for s in members}) > 1:
            print(f"!! группа {g} в нескольких частях сплита")
            bad += 1
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in chosen and b in chosen:
            inter = {s.name for s in chosen[a]} & {s.name for s in chosen[b]}
            if inter:
                print(f"!! {a} ∩ {b}: {sorted(inter)}")
                bad += 1
    print("\nутечек нет" if not bad else f"\nпроблем: {bad}")
    return 1 if bad else 0


def _table(sites: list[SiteInfo]) -> None:
    print(f"{'участок':38} {'split':5} {'группа':16} {'тип':5} {'объект':>6}  L R")
    for s in sites:
        print(
            f"{s.name:38} {s.split:5} {s.group[:16]:16} {s.kind:5} {s.n_objects:6d}  "
            f"{'+' if s.has_layers else '.'} {'+' if s.has_rgb else '.'}"
        )


def main() -> int:
    """Точка входа."""
    ap = argparse.ArgumentParser(description="Наборы участков из configs/sites.yaml")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--root", type=Path, default=None, help="переопределить root манифеста")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="участки набора")
    p_list.add_argument("set")
    p_list.add_argument("--paths", action="store_true")
    sub.add_parser("show", help="сводка по наборам и проверка утечек")
    sub.add_parser("table", help="все участки с разметкой сплита")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    sites = scan(Path(args.root or manifest.get("root", "data/raw")), manifest)

    if args.cmd == "list":
        sets = manifest.get("sets") or {}
        if args.set not in sets:
            print(f"нет набора {args.set!r}; есть: {', '.join(sets)}", file=sys.stderr)
            return 2
        for s in select(sites, sets[args.set]):
            print(s.path if args.paths else s.name)
        return 0
    if args.cmd == "show":
        return _show(sites, manifest)
    _table(sites)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
