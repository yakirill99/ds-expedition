"""Диагностика разметки (D4.0a, rev 3): классовая сверка labels.geojson с 06_*.

rev 2 сверял только КОЛИЧЕСТВА («labels 163 vs источник 163 — Δ+0 OK»), а
класс-призрак (163 «gorodishcha» на 201_Семибратнее из файлов FindsPoints /
ObjectPoly) такая сверка не ловит: счёт совпадает, класс — нет.

rev 3 сверяет КЛАССЫ и координаты:
  1. per-class: tally(labels.geojson) против tally(источника по именам 06_*);
  2. per-feature: properties.markup_file -> ожидаемый класс по имени файла,
     сравнивается с properties.class (ловит устаревшие labels.geojson);
  3. координаты labels: детект ГРАДУСОВ (CRS84 не сконвертирован при мердже)
     и выход bbox за ожидаемый регион датасета (EPSG:3857);
  4. не-канонические классы (вне семи классов регламента КОЗ №3);
  5. фичи labels без класса / без геометрии; файлы 06_* без класса в имени.

Класс по имени файла определяется той же логикой, что в merge_labels.py
(точный стем после токена модальности, затем подстрочный фолбэк), поэтому
сверка честная.

Вердикты по участку:
    OK                  — сверка сошлась по количеству, классам и координатам;
    ПРОВЕРИТЬ           — только предупреждения (не блокируют обучение);
    ПЕРЕГЕНЕРИРОВАТЬ    — labels.geojson устарел/испорчен, нужен merge_labels.

Скрипт ничего не меняет на диске.

Пример:
    uv run python scripts/diag_labels.py                          # весь data/raw
    uv run python scripts/diag_labels.py data/raw/201_Семибратнее_Городище_1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# --- логика класса из имени файла (копия merge_labels.py @55939cd) ---

CLASS_BY_STEM: dict[str, str] = {
    "курганы_целые": "kurgany_tselye",
    "курганы_поврежденные": "kurgany_povrezhdennye",
    "курганы_повреждённые": "kurgany_povrezhdennye",
    "городища": "gorodishcha",
    "фортификации": "fortifikatsii",
    "архитектуры": "arkhitektury",
    "находки": "finds_points",
    "прочие": "object_poly",
    "объекты": "object_poly",
    "FindsPoints": "finds_points",
    "ObjectPoly": "object_poly",
    "KurganyTselye": "kurgany_tselye",
    "KurganyPovrezhdennye": "kurgany_povrezhdennye",
    "Gorodishcha": "gorodishcha",
    "Fortifikatsii": "fortifikatsii",
    "Arkhitektury": "arkhitektury",
}

MODALITY_TOKENS = ("MagGr", "EtGr", "TerFl", "SpOr", "La", "Li", "Ae", "Or")

#: Канонические классы регламента КОЗ №3 (порядок не важен, важен состав).
CANON7 = frozenset(
    {
        "kurgany_povrezhdennye",
        "kurgany_tselye",
        "gorodishcha",
        "arkhitektury",
        "fortifikatsii",
        "object_poly",
        "finds_points",
    }
)

#: Ожидаемый регион данных в EPSG:3857: Швеция/Аланды (x ~1.8e6) ..
#: Иссык-Куль/Алматы (x ~8.7e6), по широте ~41..61°N.
REGION_BBOX = (1.5e6, 5.0e6, 9.0e6, 8.6e6)  # xmin, ymin, xmax, ymax


def class_of_file(path: Path) -> str | None:
    """Класс по имени файла (логика merge_labels.class_of)."""
    parts = path.stem.split("_")
    stem = None
    for i, part in enumerate(parts):
        if part in MODALITY_TOKENS:
            stem = "_".join(parts[i + 1 :])
            break
    if stem is None:
        stem = "_".join(parts[1:]) if len(parts) > 1 else path.stem
    cls = CLASS_BY_STEM.get(stem)
    if cls is not None:
        return cls
    for key, val in CLASS_BY_STEM.items():
        if key in stem:
            return val
    return None


def has_coords(geom: dict | None) -> bool:
    """Есть ли непустые координаты (как понимает merge_labels)."""
    if not geom:
        return False
    return bool(geom.get("coordinates"))


def coord_bounds(coords: list, box: list[float] | None) -> list[float] | None:
    """Рекурсивно собрать bbox [xmin, ymin, xmax, ymax] по GeoJSON-координатам."""
    if not coords:
        return box
    if isinstance(coords[0], (int, float)):
        x, y = float(coords[0]), float(coords[1])
        if box is None:
            return [x, y, x, y]
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        return box
    for part in coords:
        box = coord_bounds(part, box)
    return box


def load_feats(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"    !! не прочитан {path.name}: {exc}", file=sys.stderr)
        return []
    return [f for f in data.get("features", []) if isinstance(f, dict)]


def markup_dirs(site: Path) -> list[Path]:
    """Папки разметки: как в merge_labels, с фолбэком на 06_*."""
    dirs = sorted(site.glob("0*_*разметка*"))
    return dirs or sorted(site.glob("06_*"))


def inspect_site(site: Path) -> dict:
    """Собрать статистику участка: labels.geojson + файлы 06_*."""
    merged = site / "labels.geojson"
    organizer: list[Path] = []
    for d in markup_dirs(site):
        organizer.extend(sorted(d.glob("*.geojson")))

    # --- источник 06_* ---
    src_tally: Counter = Counter()
    src_feats = src_empty = 0
    noclass_names: list[str] = []
    for p in organizer:
        cls = class_of_file(p)
        if cls is None:
            noclass_names.append(p.name)
        for f in load_feats(p):
            if f.get("geometry") is None:
                continue  # geometry: null — вообще без геометрии
            if not has_coords(f.get("geometry")):
                src_empty += 1  # атрибутивная строка без контура
                continue
            src_feats += 1
            if cls:
                src_tally[cls] += 1

    # --- labels.geojson (обучающая GT) ---
    labels_tally: Counter = Counter()
    noncanon: Counter = Counter()
    markup_mismatch: Counter = Counter()  # (file, got, expected) -> n
    markup_unknown: list[str] = []
    labels_total = no_class = no_geom = 0
    bbox: list[float] | None = None
    for f in load_feats(merged) if merged.is_file() else []:
        geom = f.get("geometry")
        if geom is None or not has_coords(geom):
            no_geom += 1
            continue
        props = f.get("properties") or {}
        cls = props.get("class")
        if not cls:
            no_class += 1
            continue
        labels_total += 1
        labels_tally[str(cls)] += 1
        if str(cls) not in CANON7:
            noncanon[str(cls)] += 1
        bbox = coord_bounds(geom.get("coordinates"), bbox)
        mf = props.get("markup_file")
        if mf:
            expected = class_of_file(Path(str(mf)))
            if expected is None:
                markup_unknown.append(str(mf))
            elif expected != cls:
                markup_mismatch[(str(mf), str(cls), expected)] += 1

    return {
        "labels_exists": merged.is_file(),
        "labels_total": labels_total,
        "labels_tally": labels_tally,
        "labels_no_class": no_class,
        "labels_no_geom": no_geom,
        "labels_noncanon": noncanon,
        "labels_bbox": bbox,
        "markup_mismatch": markup_mismatch,
        "markup_unknown": markup_unknown,
        "src_feats": src_feats,
        "src_empty": src_empty,
        "src_tally": src_tally,
        "noclass_names": noclass_names,
        "organizer_files": len(organizer),
    }


def fmt_counter(c: Counter) -> str:
    return ", ".join(f"{k} ×{v}" for k, v in sorted(c.items())) or "—"


def fmt_bbox(bbox: list[float] | None) -> str:
    if bbox is None:
        return "нет координат"
    if bbox[2] <= 180.0 and bbox[3] <= 90.0:  # градусы
        return f"градусы: x {bbox[0]:.5f}..{bbox[2]:.5f}, y {bbox[1]:.5f}..{bbox[3]:.5f}"
    return (
        f"x {bbox[0] / 1e6:.2f}..{bbox[2] / 1e6:.2f}e6, "
        f"y {bbox[1] / 1e6:.2f}..{bbox[3] / 1e6:.2f}e6"
    )


def site_issues(info: dict) -> tuple[list[str], list[str]]:
    """Разнести проблемы участка на блокирующие (errors) и предупреждения."""
    errs: list[str] = []
    warns: list[str] = []

    if not info["labels_exists"]:
        errs.append("labels.geojson отсутствует")
        return errs, warns

    delta = info["src_feats"] - info["labels_total"]
    if delta != 0:
        errs.append(f"количество: Δ{delta:+d} (потеря при мердже)")

    class_diff: dict[str, int] = {}
    for cls in set(info["labels_tally"]) | set(info["src_tally"]):
        d = info["labels_tally"].get(cls, 0) - info["src_tally"].get(cls, 0)
        if d:
            class_diff[cls] = d
    if class_diff:
        errs.append("классы: " + ", ".join(f"{k} {v:+d}" for k, v in sorted(class_diff.items())))

    mism_total = sum(info["markup_mismatch"].values())
    if mism_total:
        errs.append(f"markup_file: {mism_total} фич с чужим классом")

    if info["labels_noncanon"]:
        errs.append("не-канонические классы: " + fmt_counter(info["labels_noncanon"]))
    if info["labels_no_class"]:
        errs.append(f"фич без класса: {info['labels_no_class']}")

    bbox = info["labels_bbox"]
    if bbox is not None:
        if bbox[2] <= 180.0 and bbox[3] <= 90.0:
            errs.append("КООРДИНАТЫ В ГРАДУСАХ — CRS84 не сконвертирован при мердже")
        elif (
            bbox[0] < REGION_BBOX[0]
            or bbox[1] < REGION_BBOX[1]
            or bbox[2] > REGION_BBOX[2]
            or bbox[3] > REGION_BBOX[3]
        ):
            warns.append("bbox вне ожидаемого региона датасета (EPSG:3857)")
    if info["labels_no_geom"]:
        warns.append(f"фич labels без геометрии: {info['labels_no_geom']}")
    if info["markup_unknown"]:
        warns.append(f"markup_file не распознан: {info['markup_unknown'][:3]}")
    if info["noclass_names"]:
        warns.append(f"файлы 06_* без класса в имени: {len(info['noclass_names'])}")
    return errs, warns


def print_site(site: Path, info: dict) -> tuple[list[str], list[str]]:
    errs, warns = site_issues(info)
    print(f"=== {site.name} ===")
    if info["labels_exists"]:
        extra = ""
        if info["labels_no_class"]:
            extra += f", БЕЗ класса {info['labels_no_class']}"
        if info["labels_no_geom"]:
            extra += f", без геометрии {info['labels_no_geom']}"
        print(
            f"  labels.geojson: {info['labels_total']} фич "
            f"({fmt_counter(info['labels_tally'])}{extra})"
        )
    else:
        print("  labels.geojson: НЕТ")
    tail = f" !! без класса в имени: {info['noclass_names'][:3]}" if info["noclass_names"] else ""
    print(
        f"  источник 06_*: {info['src_feats']} непустых фич ({fmt_counter(info['src_tally'])}); "
        f"пустых координат {info['src_empty']}; файлов {info['organizer_files']}{tail}"
    )
    print(f"  координаты: {fmt_bbox(info['labels_bbox'])}")
    if errs:
        print("  !! БЛОКИРУЮЩИЕ:")
        for e in errs:
            print(f"     - {e}")
    if warns:
        print("  ~ предупреждения:")
        for w in warns:
            print(f"     - {w}")
    if info["markup_mismatch"]:
        print("  примеры markup_file-несовпадений:")
        for (mf, got, exp), n in sorted(info["markup_mismatch"].items())[:5]:
            print(f"     {mf}: labels={got}, ожидалось {exp} (×{n})")
    if not errs and not warns:
        print("  вердикт: OK")
    elif errs:
        print("  вердикт: ПЕРЕГЕНЕРИРОВАТЬ labels.geojson")
    else:
        print("  вердикт: ПРОВЕРИТЬ")
    print()
    return errs, warns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Классовая сверка labels.geojson с 06_* (D4.0a rev 3)")
    ap.add_argument("site", type=Path, nargs="?", help="Один участок (иначе весь --root)")
    ap.add_argument("--root", type=Path, default=Path("data/raw"), help="Корень участков")
    args = ap.parse_args(argv)

    sites = [args.site] if args.site else sorted(p for p in args.root.iterdir() if p.is_dir())
    if not sites:
        print(f"!! участков не найдено в {args.root}", file=sys.stderr)
        return 1

    regen: list[str] = []
    check: list[str] = []
    tot_labels = tot_src = tot_empty = 0
    empty_sites: list[str] = []
    for site in sites:
        info = inspect_site(site)
        if not info["labels_exists"] and info["organizer_files"] == 0:
            continue
        errs, warns = print_site(site, info)
        tot_labels += info["labels_total"]
        tot_src += info["src_feats"]
        tot_empty += info["src_empty"]
        if errs:
            regen.append(site.name)
        elif warns:
            check.append(site.name)
        if info["src_empty"]:
            empty_sites.append(f"{site.name} ({info['src_empty']})")

    print("=== ИТОГ ===")
    print(
        f"labels.geojson (обучающая GT): {tot_labels} | источник 06_*: {tot_src} "
        f"| пустых координат в источнике: {tot_empty}"
    )
    if empty_sites:
        print("пустые координаты (атрибутивные строки, merge их пропускает):")
        for s in empty_sites:
            print(f"  - {s}")
    if regen:
        print(f"ПЕРЕГЕНЕРИРОВАТЬ ({len(regen)}): ")
        for s in regen:
            print(f"  - {s}")
        print(
            "команда (out-name обязателен: дефолт _labels_merged.geojson не читается):\n"
            "  uv run python scripts/merge_labels.py --site data/raw/*/ \\\n"
            "      --out-name labels.geojson   # сначала прогони с --dry-run"
        )
    if check:
        print(f"ПРОВЕРИТЬ ВРУЧНУЮ ({len(check)}): " + "; ".join(check))
    if not regen and not check:
        print("GT чиста: количества, классы и координаты сходятся — регенерация не требуется")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
