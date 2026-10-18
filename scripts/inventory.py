#!/usr/bin/env python3
"""Блок 0 — инвентаризация исходных данных.

Обходит --root рекурсивно, для каждого .tif/.las/.geojson/.zip собирает
метаданные, пишет:
  - data/inventory.json        (машинный)
  - docs/inventory.md      (таблица для людей)

Запуск:
    uv run python scripts/inventory.py --root data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_SUFFIXES = {".tif", ".tiff", ".las", ".laz", ".geojson", ".json", ".zip"}
HASH_LIMIT_BYTES = 2 * 1024**3  # хешируем только файлы < 2 ГБ


@dataclass
class Record:
    path: str
    rel_path: str
    suffix: str
    modality: str
    size_bytes: int
    mtime: str
    sha256: str | None = None
    raster: dict[str, Any] | None = None
    error: str | None = None


def classify(path: Path) -> str:
    """Грубая модальность по имени файла/расширению."""
    s = path.name.lower()
    if path.suffix.lower() in {".las", ".laz"}:
        return "lidar"
    if path.suffix.lower() == ".zip":
        return "archive"
    if path.suffix.lower() in {".geojson", ".json"}:
        return "labels"
    if path.suffix.lower() in {".tif", ".tiff"}:
        if "hillshade" in s or "hs" in s:
            return "relief"
        if "rgb" in s or "ortho" in s or "imag" in s:
            return "optic"
        if "mag" in s:
            return "geo"
        if "gpr" in s or "radar" in s:
            return "geo"
        if "dtm" in s or "dem" in s or "dsm" in s:
            return "relief"
        return "raster"
    return "other"


def sha256_of(path: Path) -> str | None:
    if path.stat().st_size >= HASH_LIMIT_BYTES:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raster_meta(path: Path) -> dict[str, Any] | None:
    try:
        import rasterio
    except ImportError:
        return {"error": "rasterio не установлен"}

    try:
        with rasterio.open(path) as ds:
            return {
                "crs": str(ds.crs) if ds.crs else None,
                "pixel_size": [abs(ds.transform.a), abs(ds.transform.e)],
                "shape": [ds.height, ds.width],
                "count": ds.count,
                "dtype": ds.dtypes[0] if ds.dtypes else None,
                "nodata": ds.nodata,
                "bounds": list(ds.bounds),
            }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def walk(root: Path) -> list[Record]:
    records: list[Record] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TARGET_SUFFIXES:
            continue
        st = path.stat()
        rec = Record(
            path=str(path.resolve()),
            rel_path=str(path.relative_to(root.parent if root.name else root)),
            suffix=path.suffix.lower(),
            modality=classify(path),
            size_bytes=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        )
        try:
            rec.sha256 = sha256_of(path)
            if path.suffix.lower() in {".tif", ".tiff"}:
                rec.raster = raster_meta(path)
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"
        records.append(rec)
    return records


def find_gaps(records: list[Record]) -> list[str]:
    """Слои с несовпадающим CRS и разные размеры пикселя — тихая ошибка №0."""
    notes: list[str] = []
    rasters = [r for r in records if r.raster and "error" not in (r.raster or {})]
    crs_set = {r.raster["crs"] for r in rasters}  # type: ignore[index]
    if len(crs_set) > 1:
        notes.append(f"РАЗНЫЕ CRS: {sorted(str(c) for c in crs_set)}")
    px_set: set[tuple] = set()
    for r in rasters:
        px = r.raster["pixel_size"]  # type: ignore[index]
        px_set.add((round(px[0], 6), round(px[1], 6)))
    if len(px_set) > 1:
        notes.append(f"РАЗНЫЙ РАЗМЕР ПИКСЕЛЯ: {sorted(px_set)}")
    if not notes:
        notes.append("Несовпадений CRS и размера пикселя не обнаружено.")
    return notes


def write_json(records: list[Record], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(records),
        "total_bytes": sum(r.size_bytes for r in records),
        "records": [asdict(r) for r in records],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(records: list[Record], gaps: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Инвентаризация исходных данных")
    lines.append("")
    lines.append(f"Сгенерировано: {datetime.now(timezone.utc).isoformat()}")
    lines.append(
        f"Файлов: {len(records)} · Суммарный размер: "
        f"{sum(r.size_bytes for r in records) / 1024**2:.1f} МБ"
    )
    lines.append("")
    lines.append("## Файлы по модальностям")
    lines.append("")
    lines.append("| Модальность | Путь | Размер, МБ | CRS | Пиксель | Shape | dtype | nodata |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda x: (x.modality, x.rel_path)):
        size_mb = f"{r.size_bytes / 1024**2:.2f}"
        if r.raster and "error" not in r.raster:
            crs = r.raster["crs"]
            px = r.raster["pixel_size"]
            px_s = f"{px[0]:.4g}×{px[1]:.4g}"
            shape = "×".join(str(v) for v in r.raster["shape"])
            dtype = r.raster["dtype"]
            nodata = r.raster["nodata"]
        else:
            crs, px_s, shape, dtype, nodata = "—", "—", "—", "—", "—"
        lines.append(
            f"| {r.modality} | `{r.rel_path}` | {size_mb} | {crs} | "
            f"{px_s} | {shape} | {dtype} | {nodata} |"
        )
    lines.append("")
    lines.append("## Дыры")
    lines.append("")
    for note in gaps:
        lines.append(f"- {note}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Инвентаризация исходных данных КОЗ №3")
    ap.add_argument("--root", type=Path, required=True, help="корень для обхода")
    ap.add_argument("--out-json", type=Path, default=Path("data/inventory.json"))
    ap.add_argument("--out-md", type=Path, default=Path("docs/inventory.md"))
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[inventory] корень не найден: {args.root}", file=sys.stderr)
        return 2

    t0 = time.time()
    records = walk(args.root)
    gaps = find_gaps(records)
    write_json(records, args.out_json)
    write_report(records, gaps, args.out_md)

    by_mod: dict[str, int] = {}
    for r in records:
        by_mod[r.modality] = by_mod.get(r.modality, 0) + 1

    print(f"[inventory] обойдено {len(records)} файлов за {time.time() - t0:.1f} с")
    for mod, n in sorted(by_mod.items()):
        print(f"  {mod:<10} {n}")
    print(f"[inventory] JSON -> {args.out_json}")
    print(f"[inventory] MD   -> {args.out_md}")
    for note in gaps:
        print(f"[inventory] ДЫРА: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
