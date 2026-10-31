"""Обзор датасета на Яндекс.Диске: что докачано, что целое, что брать в обучение.

Работает по /mnt/d (9p) — поэтому НЕ читает пиксели: только scandir + stat.
Глубокий режим (--deep) открывает GeoTIFF через rasterio; на /mnt/d это
десятки секунд на файл, запускать только по отобранным участкам.

Терпимость к недокачанному:
  - файлы Яндекса (.~*, *.~*, *.tmp, *.part, *.sync) -> участок «качается»;
  - нулевой размер при ненулевом ожидании       -> «битый»;
  - нет labels.geojson                          -> «нет разметки», не ошибка;
  - любое исключение на участке ловится, обзор продолжается.

Пример:
    uv run python scripts/survey_dataset.py --out /tmp/survey.json
    uv run python scripts/survey_dataset.py --deep --only 185,082
    uv run python scripts/survey_dataset.py --split configs/split_80.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

WIN_ROOT = (
    "/mnt/d/Yandex_Disk/Yandex.Disk-yakirill1999/Учеба/Хакатоны/"
    "2026-09-ds-expedition/04-data/train-koz-3"
)

MODALITY = {
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
NUM_RE = re.compile(r"^(\d{3})_")
TEMP_RE = re.compile(r"(\.~|~$|\.tmp$|\.part$|\.crdownload$|\.sync$)", re.IGNORECASE)
RASTER_RE = re.compile(r"\.(tif|tiff)$", re.IGNORECASE)
MAX_RASTER_BYTES = 3 * 1024**3  # защита от «файл размером 40 ГБ»

# Группы одного ландшафта — не разрываются между train и val.
KNOWN_GROUPS = ("Гнёздово", "Осечки", "Эсбю", "Кисловодская")


@dataclass
class SiteReport:
    """Что нашли в одной папке участка."""

    name: str
    number: str = ""
    modalities: dict[str, int] = field(default_factory=dict)
    n_files: int = 0
    n_rasters: int = 0
    total_bytes: int = 0
    empty_files: int = 0
    temp_files: int = 0
    has_labels: bool = False
    labels_error: str = ""
    n_objects: int = 0
    src_objects: int = 0  # объекты в 06_*_разметка/ — до импорта
    classes: dict[str, int] = field(default_factory=dict)
    has_layers: bool = False
    deep: dict[str, object] = field(default_factory=dict)
    error: str = ""

    @property
    def status(self) -> str:
        """Однословный вердикт."""
        if self.error:
            return "ОШИБКА"
        if self.temp_files or self.empty_files:
            return "качается"
        if self.n_rasters == 0 and not self.has_layers:
            return "нет исходников"
        if not self.has_labels:
            return "ok (импорт)" if self.src_objects else "нет разметки"
        if self.labels_error:
            return "разметка битая"
        if self.n_objects == 0:
            return "разметка пустая"
        return "ok"

    @property
    def kind(self) -> str:
        """Лидар / оптика / геофизика — по папкам."""
        m = set(self.modalities)
        if self.has_layers or {"01", "02"} & m:
            return "lidar"
        if m & {"03", "04", "05"}:
            return "optic"
        if m & {"07", "08", "09"}:
            return "geo"
        return "?"

    @property
    def group(self) -> str:
        """Группа участков, которые нельзя разрывать между train/val."""
        rest = NUM_RE.sub("", self.name)
        head = rest.split("_")[0]
        return head if head in KNOWN_GROUPS else self.name

    @property
    def usable(self) -> bool:
        """Годится в сплит: разметка есть хоть в источнике, хоть импортированная."""
        return self.status.startswith("ok")

    def mb(self) -> float:
        """Объём, МБ."""
        return self.total_bytes / 1024**2


def _scan_site(path: Path, deep: bool, deadline: float) -> SiteReport:
    """Один участок. Никогда не бросает — всё в report.error."""
    rep = SiteReport(name=path.name)
    m = NUM_RE.match(path.name)
    rep.number = m.group(1) if m else ""

    try:
        for entry in os.scandir(path):
            if TEMP_RE.search(entry.name):
                rep.temp_files += 1
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            code = entry.name[:2]
            if code not in MODALITY:
                continue
            rep.modalities[code] = rep.modalities.get(code, 0) + 1

            for sub in os.scandir(entry.path):
                if not sub.is_file(follow_symlinks=False):
                    continue
                if TEMP_RE.search(sub.name):
                    rep.temp_files += 1
                    continue
                try:
                    size = sub.stat(follow_symlinks=False).st_size
                except OSError:
                    rep.temp_files += 1
                    continue
                rep.n_files += 1
                rep.total_bytes += size
                if size == 0:
                    rep.empty_files += 1
                if RASTER_RE.search(sub.name):
                    rep.n_rasters += 1
                    if deep and size < MAX_RASTER_BYTES and time.monotonic() < deadline:
                        _deep_raster(Path(sub.path), rep)
    except OSError as exc:
        rep.error = f"scandir: {exc}"
        return rep

    rep.has_layers = (path / "layers").is_dir()
    for sub in path.glob("06_*"):
        if not sub.is_dir():
            continue
        for gj in sorted(sub.glob("*.geojson")):
            if TEMP_RE.search(gj.name):
                continue
            try:
                feats = json.loads(gj.read_text(encoding="utf-8")).get("features") or []
            except (OSError, ValueError):
                continue
            rep.src_objects += len(feats)

    labels = path / "labels.geojson"
    rep.has_labels = labels.is_file()
    if rep.has_labels:
        try:
            data = json.loads(labels.read_text(encoding="utf-8"))
            feats = data.get("features") or []
            rep.n_objects = len(feats)
            for f in feats:
                cls = (f.get("properties") or {}).get("class")
                if cls:
                    rep.classes[cls] = rep.classes.get(cls, 0) + 1
        except (OSError, ValueError) as exc:
            rep.labels_error = f"{type(exc).__name__}: {exc}"
    return rep


def _deep_raster(path: Path, rep: SiteReport) -> None:
    """Метаданные GeoTIFF. Медленно на /mnt/d — только по запросу."""
    try:
        import rasterio

        with rasterio.open(path) as ds:
            rep.deep[path.name] = {
                "crs": str(ds.crs),
                "size": f"{ds.width}x{ds.height}",
                "bands": ds.count,
                "res": round(float(abs(ds.res[0])), 3),
                "dtype": ds.dtypes[0],
            }
    except Exception as exc:  # noqa: BLE001 — битый файл не должен валить обзор
        rep.deep[path.name] = {"error": f"{type(exc).__name__}: {exc}"}


def _print_progress(rep: SiteReport, i: int, total: int) -> None:
    """Строка по мере обхода — видно, что процесс жив."""
    mods = ",".join(MODALITY.get(c, c) for c in sorted(rep.modalities)) or "—"
    print(
        f"[{i:3d}/{total}] {rep.name:36} {rep.status:16} {rep.kind:6} {rep.mb():7.1f} МБ  {mods}",
        flush=True,
    )


def _print_table(reports: list[SiteReport]) -> None:
    """Сводная таблица в конце."""
    head = (
        f"{'участок':36} {'статус':16} {'тип':6} "
        f"{'файлов':>6} {'МБ':>8} {'объектов':>8} {'в 06_':>8}"
    )
    print("\n" + head)
    print("-" * len(head))
    for r in sorted(reports, key=lambda x: x.name):
        print(
            f"{r.name:36} {r.status:16} {r.kind:6} {r.n_files:6d} {r.mb():8.1f} "
            f"{r.n_objects:8d} {r.src_objects:8d}"
        )

    total_mb = sum(r.mb() for r in reports)
    ok = [r for r in reports if r.usable]
    bad = [r for r in reports if not r.usable]
    print("-" * len(head))
    print(
        f"{'ВСЕГО':36} {'':16} {'':6} {sum(r.n_files for r in reports):6d} "
        f"{total_mb:8.1f} {sum(r.n_objects for r in reports):8d} "
        f"{sum(r.src_objects for r in reports):8d}"
    )
    print(f"\nучастков {len(reports)} | ok {len(ok)} | требуют внимания {len(bad)}")
    if bad:
        print("\nтребуют внимания:")
        for r in bad:
            print(f"  {r.name:36} {r.status:16} {r.labels_error or r.error or ''}")


def _split(reports: list[SiteReport], train_frac: float, seed: int) -> dict[str, list[str]]:
    """Стратифицированный сплит с сохранением групп.

    Делим по (тип участка, группа): группа целиком уходит в train или val,
    доли выдерживаются по возможности — при малых числах проценты плавают,
    это нормально и лучше, чем разорвать Гнёздово.
    """
    import random

    rng = random.Random(seed)
    usable = [r for r in reports if r.usable]
    buckets: dict[tuple[str, str], list[str]] = {}
    for r in usable:
        buckets.setdefault((r.kind, r.group), []).append(r.name)

    train: list[str] = []
    val: list[str] = []
    for key in sorted(buckets):
        members = sorted(buckets[key])
        rng.shuffle(members)
        if len(members) == 1:
            (train if rng.random() < train_frac else val).extend(members)
            continue
        cut = max(1, round(len(members) * train_frac))
        cut = min(cut, len(members) - 1)  # хотя бы один в val
        train.extend(members[:cut])
        val.extend(members[cut:])
    return {"train_sites": sorted(train), "val_sites": sorted(val)}


def main() -> int:
    """Точка входа."""
    ap = argparse.ArgumentParser(description="Обзор датасета и отбор 80% в обучение")
    ap.add_argument("--root", type=Path, default=Path(WIN_ROOT))
    ap.add_argument("--out", type=Path, default=Path("data/survey.json"))
    ap.add_argument("--deep", action="store_true", help="rasterio-метаданные (медленно на /mnt/d)")
    ap.add_argument("--only", default="", help="номера участков через запятую: 185,082")
    ap.add_argument("--limit", type=int, default=0, help="обойти не больше N участков")
    ap.add_argument("--sleep", type=float, default=0.0, help="пауза между участками, с")
    ap.add_argument("--deadline", type=float, default=1800.0, help="стоп deep-режима, с")
    ap.add_argument("--split", type=Path, default=None, help="куда записать YAML-сплит")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        print(f"нет каталога: {root}", file=sys.stderr)
        return 2

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    sites = sorted(p for p in root.iterdir() if p.is_dir() and not TEMP_RE.search(p.name))
    unknown = [p.name for p in sites if not NUM_RE.match(p.name)]
    if unknown:
        print(
            f"{root.resolve()}: {len(unknown)} каталогов без номера участка "
            f"(например {unknown[0]!r}) — похоже, путь не тот",
            file=sys.stderr,
        )
        return 2
    if only:
        sites = [p for p in sites if (NUM_RE.match(p.name) or [None, ""])[1] in only]
    if args.limit:
        sites = sites[: args.limit]

    print(f"корень: {root}\nучастков: {len(sites)}  deep={args.deep}\n", flush=True)
    deadline = time.monotonic() + args.deadline

    reports: list[SiteReport] = []
    for i, path in enumerate(sites, 1):
        rep = _scan_site(path, args.deep, deadline)
        reports.append(rep)
        _print_progress(rep, i, len(sites))
        if args.sleep:
            time.sleep(args.sleep)

    _print_table(reports)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "status": r.status,
                    "kind": r.kind,
                    "group": r.group,
                    "modalities": r.modalities,
                    "files": r.n_files,
                    "mb": round(r.mb(), 1),
                    "objects": r.n_objects,
                    "classes": r.classes,
                    "deep": r.deep,
                    "error": r.error or r.labels_error,
                }
                for r in reports
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nотчёт: {args.out}")

    if args.split:
        split = _split(reports, args.train_frac, args.seed)
        lines = [
            "# Сгенерировано scripts/survey_dataset.py",
            f"# train_frac={args.train_frac} seed={args.seed}",
            "train_sites:",
            *[f"  - {n}" for n in split["train_sites"]],
            "val_sites:",
            *[f"  - {n}" for n in split["val_sites"]],
        ]
        args.split.parent.mkdir(parents=True, exist_ok=True)
        args.split.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"сплит: train {len(split['train_sites'])}, val {len(split['val_sites'])} "
            f"-> {args.split}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
