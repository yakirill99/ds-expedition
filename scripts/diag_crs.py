"""Диагностика CRS-согласованности участка: las vs разметка vs UTM.json.

Зачем: import_labels.py на 006 показал, что разметка (объявлена EPSG:3857)
в пересчёте в EPSG:25833 даёт X≈2.37 млн — за пределами любой зоны UTM —
и с облаком las (объявлен EPSG:25833, X≈488-489 тыс.) не пересекается.

Скрипт печатает всё, чтобы решить, где врёт CRS:
  - CRS из заголовка las (включая ПО/систему, которой писали файл), bounds
    и место las на земле под разными трактовками координат;
  - сырое BBox разметки, её место (как EPSG:3857 -> lon/lat), реальный размер;
  - содержимое UTM.json из корня участка и его BBox, если там координаты;
  - расстояния между «местом разметки» и «местом las» по каждой трактовке.

Пример:
    uv run python scripts/diag_crs.py data/raw/006_Осечки_1.1км
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from pyproj import Transformer

# Кандидаты на «настоящую» проекцию las: 25833 (заявлено), 32636-32638 (UTM 36N-38N).
CANDIDATES = (25833, 32636, 32637, 32638)


def collect_points(node: object) -> list[tuple[float, float]]:
    """Все пары чисел (x, y) в произвольной вложенной JSON-структуре."""
    pts: list[tuple[float, float]] = []

    def walk(n: object) -> None:
        if isinstance(n, (list, tuple)):
            if len(n) >= 2 and isinstance(n[0], (int, float)) and isinstance(n[1], (int, float)):
                pts.append((float(n[0]), float(n[1])))
            else:
                for child in n:
                    walk(child)
        elif isinstance(n, dict):
            for child in n.values():
                walk(child)

    walk(node)
    return pts


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def find_one(site: Path, patterns: tuple[str, ...], what: str) -> Path | None:
    for pat in patterns:
        hits = sorted(site.glob(pat))
        if hits:
            if len(hits) > 1:
                print(f"!! {what}: несколько совпадений, беру {hits[0].name}", file=sys.stderr)
            return hits[0]
    return None


def to_ll(epsg: int, x: float, y: float) -> tuple[float, float]:
    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return tr.transform(x, y)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Диагностика CRS участка (las / разметка / UTM.json)")
    ap.add_argument("site", type=Path, help="Каталог участка, напр. data/raw/006_Осечки_1.1км")
    args = ap.parse_args(argv)
    site = args.site

    las = find_one(site, ("01_*/*.las", "01_*/*.laz"), "LAS")
    labels = find_one(site, ("06_*/*.geojson", "06_*/*.json"), "разметка")

    print(f"=== Участок: {site.name} ===")

    # --- LAS ---
    if las is None:
        print("!! LAS не найден", file=sys.stderr)
        return 1
    import laspy

    with laspy.open(las) as reader:
        header = reader.header
        x0, x1, y0, y1 = header.x_min, header.x_max, header.y_min, header.y_max
        software = getattr(header, "generating_software", "?")
        system_id = getattr(header, "system_identifier", "?")
        try:
            crs = header.parse_crs()
            if crs is not None:
                crs_desc = f"EPSG:{crs.to_epsg()} ({crs.name})"
            else:
                crs_desc = "нет в заголовке"
        except Exception as exc:  # noqa: BLE001
            crs_desc = f"не разобрался: {exc}"
    print(f"LAS: {las}")
    print(f"  ПО: {software!r}, система: {system_id!r}")
    print(f"  CRS в заголовке: {crs_desc}")
    print(
        f"  bounds: X {x0:.1f}..{x1:.1f}, Y {y0:.1f}..{y1:.1f} "
        f"(span {x1 - x0:.1f} x {y1 - y0:.1f} м)"
    )
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    print("  центр las, трактовка координат:")
    for epsg in CANDIDATES:
        lon, lat = to_ll(epsg, cx, cy)
        print(f"    как EPSG:{epsg}: lon {lon:.4f}, lat {lat:.4f}")

    # --- Разметка ---
    glon: float | None = None
    glat: float | None = None
    if labels is None:
        print("!! GeoJSON разметки не найден", file=sys.stderr)
    else:
        data = json.loads(labels.read_text(encoding="utf-8"))
        declared = None
        if isinstance(data, dict) and isinstance(data.get("crs"), dict):
            declared = (data["crs"].get("properties") or {}).get("name")
        pts = collect_points(data.get("features") if isinstance(data, dict) else data)
        if not pts:
            print(f"Разметка: {labels} — координат не нашёл")
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            print(f"Разметка: {labels} ({len(pts)} вершин)")
            print(f"  объявленный CRS: {declared or 'не указан'}")
            print(f"  сырое BBox: X {min(xs):.1f}..{max(xs):.1f}, Y {min(ys):.1f}..{max(ys):.1f}")
            glon, glat = to_ll(3857, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
            scale = math.cos(math.radians(glat))
            wx = (max(xs) - min(xs)) * scale
            wy = (max(ys) - min(ys)) * scale
            print(
                f"  как EPSG:3857: центр lon {glon:.4f}, lat {glat:.4f}; "
                f"реальный размер ~{wx:.0f} x {wy:.0f} м"
            )

    # --- UTM.json ---
    utm = site / "UTM.json"
    if utm.is_file():
        raw = utm.read_text(encoding="utf-8")
        print(f"UTM.json: {utm} ({len(raw)} байт)")
        print(f"  начало: {raw[:500].replace(chr(10), ' ')}")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if parsed is not None:
            pts = collect_points(parsed)
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                print(
                    f"  BBox UTM.json: X {min(xs):.1f}..{max(xs):.1f}, "
                    f"Y {min(ys):.1f}..{max(ys):.1f} ({len(pts)} пар координат)"
                )
                ux, uy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                near_las = abs(ux - cx) < (x1 - x0) and abs(uy - cy) < (y1 - y0)
                print(f"  попадает в bounds las: {'ДА' if near_las else 'НЕТ'}")
    else:
        print("UTM.json: нет в корне участка")

    # --- Вердикт ---
    if glon is not None:
        print("Расстояние: (центр разметки как 3857) -> (центр las как ...):")
        for epsg in CANDIDATES:
            llon, llat = to_ll(epsg, cx, cy)
            d = haversine_km(glon, glat, llon, llat)
            verdict = "СОВПАДАЮТ ✓" if d < 5.0 else f"{d:.0f} км"
            print(f"  las как EPSG:{epsg}: {verdict}")
        print(
            "\nДальше: пришли вывод целиком — решим, где врёт CRS "
            "(заголовок las, объявление разметки или UTM.json)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
