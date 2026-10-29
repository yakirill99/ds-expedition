"""Инвентаризация растровых участков (D4.1, rev 2): данные + GT + CRS.

Rev 2 — что исправлено по сравнению с rev 1:
  - ДВОЙНОЙ СЧЁТ: rev 1 суммировал labels.geojson И организаторские 06_*
    (а labels — это и есть мердж 06_*), завышая всё вдвое. Теперь GT
    считается ТОЛЬКО по labels.geojson; 06_* — отдельной строкой для сверки.
  - ФОЛБЭК КЛАССА: rev 1 угадывал класс по подстроке имени файла с порядком
    паттернов «городищ» раньше «findspoints» — из-за этого на 201
    («Семибратнее_Городище_1») finds/object превратились в 163 ложных
    gorodishcha. Теперь та же логика, что в merge_labels.py: split по токену
    модальности + точный стем, фолбэк — по словарю классов.
  - Класс в properties организаторских файлов отсутствует (только fid/id) —
    это норма, он берётся из имени файла.

По каждому участку печатается:
  - модальности, UTM.json, layers;
  - каждый растр: CRS, размер, полосы, пиксель, bbox, центр на земле;
  - GT: классы из labels.geojson + сверка с источником 06_*;
  - вердикт: LAS-участок / кандидат D4 / геофизика D4 / пусто.

Пример:
    uv run python scripts/diag_raster.py                        # весь data/raw
    uv run python scripts/diag_raster.py data/raw/013_Нора_Виннон
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from pyproj import Transformer

# Номер каталога -> модальность (регламент КОЗ №3, слайд «Формат файлов»).
MODALITY_BY_NUM = {
    "01": "La",
    "02": "Li",
    "03": "Ae",
    "04": "SpOr",
    "05": "Or",
    "06": "разметка",
    "07": "MagGr",
    "08": "EtGr",
    "09": "TerFl",
}
CODE_RE = re.compile(r"(?:^|_)(La|Li|Ae|SpOr|Or|MagGr|EtGr|TerFl)(?:_|$)")

# Стем имени файла -> класс (копия merge_labels.py — единая логика).
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

# Веса метрики (compute_metrics.py организаторов).
CLASS_WEIGHTS = {
    "kurgany_povrezhdennye": 30.0,
    "kurgany_tselye": 20.0,
    "gorodishcha": 15.0,
    "arkhitektury": 10.0,
    "fortifikatsii": 5.0,
    "object_poly": 10.0,
    "finds_points": 10.0,
}

LAS_MODS = {"La"}
D4_RASTER_MODS = {"Li", "Ae", "SpOr", "Or"}
D4_GEOPHYS_MODS = {"MagGr", "EtGr", "TerFl"}


def classify_dir(name: str) -> str:
    """Модальность по номеру каталога, фолбэк — по коду в имени."""
    num = name.split("_", 1)[0]
    if num in MODALITY_BY_NUM:
        return MODALITY_BY_NUM[num]
    m = CODE_RE.search(name)
    return m.group(1) if m else "прочее"


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


def load_feats(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"    !! не прочитан {path.name}: {exc}", file=sys.stderr)
        return []
    return [f for f in data.get("features", []) if isinstance(f, dict)]


def gt_stats(site: Path) -> dict:
    """GT участка: labels.geojson (обучающая) + источник 06_* (для сверки)."""
    merged = site / "labels.geojson"
    tally: Counter = Counter()
    labels_total = 0
    labels_exists = merged.is_file()
    if labels_exists:
        for f in load_feats(merged):
            if f.get("geometry") is None or not has_coords(f.get("geometry")):
                continue
            cls = (f.get("properties") or {}).get("class")
            if cls:
                tally[str(cls)] += 1
                labels_total += 1

    src_feats = src_empty = 0
    src_tally: Counter = Counter()
    organizer: list[Path] = []
    for d in sorted(site.glob("06_*")):
        organizer.extend(sorted(d.glob("*.geojson")))
    for p in organizer:
        cls = class_of_file(p)
        for f in load_feats(p):
            if f.get("geometry") is None:
                continue
            if not has_coords(f.get("geometry")):
                src_empty += 1
                continue
            src_feats += 1
            if cls:
                src_tally[cls] += 1

    return {
        "labels_exists": labels_exists,
        "tally": tally,
        "total": labels_total,
        "src_feats": src_feats,
        "src_empty": src_empty,
        "src_tally": src_tally,
    }


def raster_info(path: Path) -> dict:
    import rasterio

    try:
        with rasterio.open(path) as src:
            b = src.bounds
            epsg = src.crs.to_epsg() if src.crs else None
            return {
                "path": path,
                "epsg": epsg,
                "w": src.width,
                "h": src.height,
                "bands": src.count,
                "dtype": src.dtypes[0] if src.dtypes else "?",
                "res": (float(src.res[0]), float(src.res[1])),
                "bounds": (float(b.left), float(b.bottom), float(b.right), float(b.top)),
                "center": ((b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0),
            }
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "error": str(exc)}


def to_ll(epsg: int, x: float, y: float) -> tuple[float, float]:
    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return tr.transform(x, y)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def scan_site(site: Path) -> dict:
    """Собрать всё об одном участке (без проверки содержимого пикселей)."""
    mods: dict[str, list[Path]] = {}
    las: list[Path] = []
    for p in site.rglob("*"):
        if not p.is_file() or "layers" in p.parts[len(site.parts) :]:
            continue
        if p.suffix.lower() in (".las", ".laz"):
            las.append(p)
            continue
        if p.suffix.lower() in (".tif", ".tiff"):
            rel = p.relative_to(site)
            mod = classify_dir(rel.parts[0]) if len(rel.parts) > 1 else "прочее"
            mods.setdefault(mod, []).append(p)

    if las:
        status = "LAS-участок"
    elif mods.keys() & D4_RASTER_MODS:
        status = "кандидат D4 (растровый)"
    elif mods.keys() & D4_GEOPHYS_MODS:
        status = "кандидат D4 (геофизика)"
    else:
        status = "пусто (SKIP)"
    return {
        "site": site,
        "status": status,
        "mods": mods,
        "las": las,
        "gt": gt_stats(site),
        "utm_json": (site / "UTM.json").is_file(),
        "layers_built": (site / "layers").is_dir(),
    }


def print_site(info: dict) -> None:
    site: Path = info["site"]
    mods = info["mods"]
    print(f"=== {site.name} ===")
    mod_str = ", ".join(f"{m}({len(v)})" for m, v in sorted(mods.items())) or "—"
    utm = "да" if info["utm_json"] else "НЕТ"
    built = "да" if info["layers_built"] else "нет"
    print(
        f"  статус: {info['status']} | модальности: {mod_str} | UTM.json: {utm} | layers: {built}"
    )

    centers: list[tuple[float, float]] = []
    for mod, files in sorted(mods.items()):
        for f in files:
            ri = raster_info(f)
            if "error" in ri:
                print(f"  [{mod}] {f.name}: !! НЕ ЧИТАЕТСЯ — {ri['error']}")
                continue
            if ri["epsg"] is None:
                print(
                    f"  [{mod}] {f.name}: EPSG:None {ri['w']}x{ri['h']}, "
                    f"{ri['bands']} пол., {ri['dtype']}, res {ri['res'][0]:.2f} — "
                    f"!! CRS НЕ ЗАПИСАН (кадр — из UTM.json)"
                )
                continue
            lon, lat = to_ll(ri["epsg"], *ri["center"])
            centers.append((lon, lat))
            print(
                f"  [{mod}] {f.name}: EPSG:{ri['epsg']} {ri['w']}x{ri['h']}, "
                f"{ri['bands']} пол., {ri['dtype']}, res {ri['res'][0]:.2f} м/px, "
                f"bbox X {ri['bounds'][0]:.0f}..{ri['bounds'][2]:.0f}, "
                f"Y {ri['bounds'][1]:.0f}..{ri['bounds'][3]:.0f} "
                f"(центр lon {lon:.4f}, lat {lat:.4f})"
            )
    if len(centers) > 1:
        spread = max(
            haversine_km(a[0], a[1], b[0], b[1])
            for i, a in enumerate(centers)
            for b in centers[i + 1 :]
        )
        if spread > 5.0:
            print(f"  !! центры растров расходятся на {spread:.1f} км — геопривязка несогласована")

    gt = info["gt"]
    if gt["labels_exists"]:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(gt["tally"].items())) or "пусто"
        print(f"  GT labels.geojson: {parts} (объектов: {gt['total']})")
    else:
        print("  GT labels.geojson: НЕТ")
    if gt["src_feats"] or gt["src_empty"]:
        sparts = ", ".join(f"{k} {v}" for k, v in sorted(gt["src_tally"].items())) or "—"
        print(
            f"  источник 06_*: {gt['src_feats']} непустых ({sparts}), "
            f"пустых координат {gt['src_empty']}"
        )
        if gt["labels_exists"]:
            delta = gt["src_feats"] - gt["total"]
            mark = "OK" if delta == 0 else f"!! ПОТЕРЯНО {delta} при мердже"
            print(f"  сверка GT: Δ{delta:+d} {mark}")
    print()


def print_summary(infos: list[dict]) -> None:
    n_las = sum(1 for i in infos if i["status"] == "LAS-участок")
    n_rast = sum(1 for i in infos if "растровый" in i["status"])
    n_geo = sum(1 for i in infos if "геофизика" in i["status"])
    n_empty = sum(1 for i in infos if i["status"].startswith("пусто"))
    print("=== СВОДКА ===")
    print(
        f"участков: {len(infos)} | LAS: {n_las} | D4 растровые: {n_rast} | "
        f"D4 геофизика: {n_geo} | пустых: {n_empty}"
    )

    total_lost = 0
    for title, pred in (
        ("D4-участки (без LAS)", lambda i: i["status"] != "LAS-участок"),
        ("LAS-участки", lambda i: i["status"] == "LAS-участок"),
    ):
        group = [i for i in infos if pred(i) and i["gt"]["labels_exists"]]
        total: Counter = Counter()
        src = empty = 0
        for i in group:
            total.update(i["gt"]["tally"])
            src += i["gt"]["src_feats"]
            empty += i["gt"]["src_empty"]
        weight = sum(v * CLASS_WEIGHTS.get(k, 0.0) for k, v in total.items())
        print(
            f"{title}: {len(group)} шт. с GT, объектов {sum(total.values())}, "
            f"вес в метрике ~{weight:.0f}%·шт, источник 06_*: {src} (пустых {empty})"
        )
        for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
            w = CLASS_WEIGHTS.get(k)
            share = f" (вес {w:.0f}%)" if w else ""
            print(f"    {k}: {v}{share}")
        total_lost += src - sum(total.values())
    print(
        f"\nПотенциально потеряно при мердже (суммарно): {total_lost} фич — детали: diag_labels.py"
    )
    print("Дальше: классовая сверка GT — diag_labels.py, каналы — dataset_real.yaml")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Инвентаризация растровых участков (D4.1 rev 2)")
    ap.add_argument("site", type=Path, nargs="?", help="Один участок (иначе весь --root)")
    ap.add_argument("--root", type=Path, default=Path("data/raw"), help="Корень участков")
    args = ap.parse_args(argv)

    sites = [args.site] if args.site else sorted(p for p in args.root.iterdir() if p.is_dir())
    if not sites:
        print(f"!! участков не найдено в {args.root}", file=sys.stderr)
        return 1

    infos = [scan_site(s) for s in sites]
    for info in infos:
        print_site(info)
    print_summary(infos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
