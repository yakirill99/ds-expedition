#!/usr/bin/env python3
"""Локатор сырых данных КОЗ-3: где лежат участки и сколько их реально на диске.

Данные приходят через Yandex.Disk на диск /mnt/d (WSL drvfs/9P). Полный обход
такого дерева (rglob + stat каждого файла) порождает десятки тысяч мелких
операций ввода-вывода и на фоне синхронизации вешает диск. Поэтому скрипт:

- делает только плоские os.scandir глубиной не более 2 от корня данных;
- НЕ читает содержимое файлов и НЕ считает хэши (никогда);
- по умолчанию работает в quick-режиме: ни одного stat(), только имена файлов;
- размеры снимает только в --full и только по одному stat() на файл;
- печатает прогресс, останавливается по --timeout и корректно реагирует
  на Ctrl+C (печатает частичный отчёт вместо падения);
- первый корень, где найдены участки, завершает обход (--all-candidates
  отменяет этот ранний выход).

Участок (сайт) — директория вида "001_Гнёздово_Нивленское" внутри корня
участков (train-koz-3, test). Служебные директории (layers, extra, archive,
current, .git) не открываются; zip-двойники в корне участков не считаются.

Примеры:
    uv run python scripts/locate_data.py --expect-min 192
    uv run python scripts/locate_data.py --full --json locate_report.json
    uv run python scripts/locate_data.py --root "/mnt/d/.../04-data" --timeout 120
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

SITE_RE = re.compile(r"^(\d{3})_")

# Служебные директории, которые не открываем вообще (кэш слоёв, выгрузки).
SKIP_DIR_NAMES = {
    "layers",
    "extra",
    "archive",
    "current",
    ".git",
    ".yandexdisk",
    "__pycache__",
}

LAS_SUFFIXES = (".las", ".laz")
RASTER_SUFFIXES = (".tif", ".tiff")

PROGRESS_EVERY = 25


def default_candidates() -> list[Path]:
    """Известные варианты расположения данных (первый удачный — рабочий)."""
    yandex = Path(
        "/mnt/d/Yandex_Disk/Yandex.Disk-yakirill1999/Учеба/Хакатоны/2026-09-ds-expedition"
    )
    repo_data = Path.home() / "ds-expedition" / "data"
    return [
        yandex / "04-data",
        yandex / "04-data" / "train-koz-3",
        yandex / "data",
        repo_data,
        repo_data / "raw",
        Path("data"),
    ]


class Deadline:
    """Мягкий дедлайн по времени: expired -> скан останавливается без ошибки."""

    def __init__(self, seconds: float | None) -> None:
        self._t_end = None if seconds is None else time.monotonic() + seconds

    @property
    def expired(self) -> bool:
        return self._t_end is not None and time.monotonic() >= self._t_end


@dataclass
class SiteInfo:
    """Один участок: количество las/tif по именам файлов (без чтения)."""

    site_id: str
    name: str
    path: str
    las_count: int = 0
    raster_count: int = 0
    las_bytes: int | None = None
    raster_bytes: int | None = None

    @property
    def ok(self) -> bool:
        return self.las_count >= 1 and self.raster_count >= 1

    def to_dict(self) -> dict:
        out: dict = {
            "site_id": self.site_id,
            "name": self.name,
            "path": self.path,
            "las_count": self.las_count,
            "raster_count": self.raster_count,
            "ok": self.ok,
        }
        if self.las_bytes is not None:
            out["las_bytes"] = self.las_bytes
        if self.raster_bytes is not None:
            out["raster_bytes"] = self.raster_bytes
        return out


@dataclass
class GroupResult:
    """Корень участков (например train-koz-3 или test) со списком участков."""

    name: str
    path: str
    sites: list[SiteInfo] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "sites": [s.to_dict() for s in self.sites],
            "sites_total": len(self.sites),
            "sites_ok": sum(1 for s in self.sites if s.ok),
            "incomplete": [s.name for s in self.sites if not s.ok],
            "truncated": self.truncated,
        }


@dataclass
class CandidateResult:
    """Итог проверки одного корня-кандидата."""

    path: str
    exists: bool
    groups: list[GroupResult] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "exists": self.exists,
            "truncated": self.truncated,
            "groups": [g.to_dict() for g in self.groups],
        }


def list_subdirs(path: Path) -> list[Path]:
    """Поддиректории одного уровня, кроме служебных; ошибки -> пустой список."""
    try:
        with os.scandir(path) as it:
            return [
                Path(entry.path)
                for entry in it
                if entry.name not in SKIP_DIR_NAMES and entry.is_dir()
            ]
    except OSError:
        return []


def collect_sites(root: Path) -> list[Path]:
    """Директории-участки непосредственно внутри root (без обхода вглубь)."""
    try:
        with os.scandir(root) as it:
            entries = [
                Path(entry.path)
                for entry in it
                if entry.name not in SKIP_DIR_NAMES and entry.is_dir()
            ]
    except OSError:
        return []
    return [p for p in entries if SITE_RE.match(p.name)]


def scan_site(site_path: Path, with_sizes: bool = False) -> SiteInfo:
    """Один участок: плоские списки в корне участка и его подпапках.

    Классификация по суффиксу имени; в quick-режиме stat() не вызывается
    вообще (важно для медленного drvfs-пути /mnt/d во время синхронизации).
    """
    match = SITE_RE.match(site_path.name)
    info = SiteInfo(
        site_id=match.group(1) if match else "",
        name=site_path.name,
        path=str(site_path),
    )
    las_bytes = 0
    raster_bytes = 0
    for directory in [site_path, *list_subdirs(site_path)]:
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    low = entry.name.lower()
                    if low.endswith(LAS_SUFFIXES):
                        kind = "las"
                    elif low.endswith(RASTER_SUFFIXES):
                        kind = "raster"
                    else:
                        continue
                    size = 0
                    if with_sizes:
                        try:
                            size = entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            size = 0
                    if kind == "las":
                        info.las_count += 1
                        las_bytes += size
                    else:
                        info.raster_count += 1
                        raster_bytes += size
        except OSError:
            continue
    if with_sizes:
        info.las_bytes = las_bytes
        info.raster_bytes = raster_bytes
    return info


def scan_sites_root(
    root: Path,
    *,
    with_sizes: bool,
    deadline: Deadline,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[SiteInfo], bool]:
    """Сканирует список участков в root; возвращает (частичный_список, стоп)."""
    result: list[SiteInfo] = []
    try:
        for index, site_path in enumerate(collect_sites(root), start=1):
            if deadline.expired:
                return result, True
            result.append(scan_site(site_path, with_sizes=with_sizes))
            if progress is not None and index % PROGRESS_EVERY == 0:
                progress(index)
    except KeyboardInterrupt:
        return result, True
    return result, False


def scan_data_root(
    root: Path,
    *,
    with_sizes: bool,
    deadline: Deadline,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[GroupResult], bool]:
    """Корень данных: либо сам корень участков, либо содержит его на уровень ниже.

    Поддерживаются оба варианта: .../train-koz-3 (участки внутри) и
    .../04-data (внутри train-koz-3 и test). Группы без участков не выводятся.
    """
    truncated_any = False
    groups: list[GroupResult] = []
    direct_sites, truncated = scan_sites_root(
        root, with_sizes=with_sizes, deadline=deadline, progress=progress
    )
    truncated_any = truncated_any or truncated
    if direct_sites:
        groups.append(
            GroupResult(name=root.name, path=str(root), sites=direct_sites, truncated=truncated)
        )
        return groups, truncated_any
    for sub in list_subdirs(root):
        if deadline.expired:
            truncated_any = True
            break
        sub_sites, truncated = scan_sites_root(
            sub, with_sizes=with_sizes, deadline=deadline, progress=progress
        )
        truncated_any = truncated_any or truncated
        if sub_sites or truncated:
            groups.append(
                GroupResult(name=sub.name, path=str(sub), sites=sub_sites, truncated=truncated)
            )
    groups.sort(key=lambda g: (-len(g.sites), g.name))
    return groups, truncated_any


def scan_candidate(
    root: Path,
    *,
    with_sizes: bool,
    deadline: Deadline,
    progress: Callable[[int], None] | None = None,
) -> CandidateResult:
    """Проверяет одного кандидата: несуществующий путь — не ошибка."""
    try:
        exists = root.exists()
    except OSError:
        exists = False
    result = CandidateResult(path=str(root), exists=exists)
    if not exists:
        return result
    groups, truncated = scan_data_root(
        root, with_sizes=with_sizes, deadline=deadline, progress=progress
    )
    result.groups = groups
    result.truncated = truncated
    return result


def compute_totals(results: list[CandidateResult]) -> dict[str, int]:
    """Суммарные счётчики по всем группам всех кандидатов."""
    sites = [s for cand in results for g in cand.groups for s in g.sites]
    return {
        "sites": len(sites),
        "ok": sum(1 for s in sites if s.ok),
        "las_files": sum(s.las_count for s in sites),
        "raster_files": sum(s.raster_count for s in sites),
    }


def fmt_bytes(num: int) -> str:
    value = float(num)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024:
            return f"{int(value)} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ТБ"


def print_report(
    results: list[CandidateResult],
    *,
    quick: bool,
    expect_min: int,
    candidates_total: int,
    interrupted: bool,
    out: TextIO,
) -> None:
    mode = "quick (без stat — безопасно при синхронизации)" if quick else "full (с размерами)"
    print("=== Локатор данных КОЗ-3 ===", file=out)
    print(f"Режим: {mode}; кандидатов: {candidates_total}", file=out)
    for index, cand in enumerate(results, start=1):
        if not cand.exists:
            tag = "нет пути"
        elif cand.truncated:
            tag = "ЧАСТИЧНО (прервано по таймауту/Ctrl+C)"
        else:
            tag = "ок"
        print(f"[{index}/{len(results)}] {cand.path} — {tag}", file=out)
        for group in cand.groups:
            ok = sum(1 for s in group.sites if s.ok)
            las = sum(s.las_count for s in group.sites)
            raster = sum(s.raster_count for s in group.sites)
            line = (
                f"    {group.name}: сайтов {len(group.sites)} | полных {ok}"
                f" | las {las} | tif {raster}"
            )
            if group.sites and group.sites[0].las_bytes is not None:
                las_b = sum(s.las_bytes or 0 for s in group.sites)
                ras_b = sum(s.raster_bytes or 0 for s in group.sites)
                line += f" | {fmt_bytes(las_b)} las, {fmt_bytes(ras_b)} tif"
            print(line, file=out)
            incomplete = [s for s in group.sites if not s.ok]
            if incomplete:
                preview = ", ".join(s.name for s in incomplete[:10])
                more = f" …+{len(incomplete) - 10}" if len(incomplete) > 10 else ""
                print(f"      неполные ({len(incomplete)}): {preview}{more}", file=out)
    totals = compute_totals(results)
    print(
        f"ИТОГО: сайтов {totals['sites']}, полных {totals['ok']},"
        f" las {totals['las_files']}, tif {totals['raster_files']}",
        file=out,
    )
    if expect_min > 0:
        verdict = "OK" if totals["ok"] >= expect_min else "НЕ ДОСТИГНУТ (возможно, ещё качается)"
        print(f"Минимум полных сайтов {expect_min}: {verdict}", file=out)
    if interrupted:
        print("ВНИМАНИЕ: отчёт частичный — скан был прерван.", file=out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Локатор сырых данных КОЗ-3 (безопасный к медленному диску)"
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        dest="roots",
        help="корень данных (можно повторять); по умолчанию — известные пути",
    )
    parser.add_argument(
        "--expect-min",
        type=int,
        default=0,
        help="минимум полных сайтов для сводки (например 192)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="снять размеры файлов (по одному stat на файл; медленнее)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="мягкий лимит времени скана в секундах (0 — без лимита)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="сохранить отчёт в JSON по этому пути",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="код возврата 2, если полных сайтов меньше --expect-min",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="сканировать все пути, а не останавливаться на первом удачном",
    )
    parser.add_argument("--quiet", action="store_true", help="без строк прогресса")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quick = not args.full
    timeout = args.timeout if args.timeout and args.timeout > 0 else None
    deadline = Deadline(timeout)
    roots = [Path(p) for p in args.roots] or default_candidates()
    results: list[CandidateResult] = []
    interrupted = False

    def progress(done: int) -> None:
        if not args.quiet:
            print(f"    ... просмотрено сайтов: {done}", flush=True)

    try:
        for root in roots:
            if deadline.expired:
                interrupted = True
                break
            candidate = scan_candidate(
                root, with_sizes=not quick, deadline=deadline, progress=progress
            )
            results.append(candidate)
            found = sum(len(g.sites) for g in candidate.groups)
            if found and not args.all_candidates:
                break
    except KeyboardInterrupt:
        interrupted = True
    interrupted = interrupted or any(c.truncated for c in results)

    print_report(
        results,
        quick=quick,
        expect_min=args.expect_min,
        candidates_total=len(roots),
        interrupted=interrupted,
        out=sys.stdout,
    )

    if args.json_path:
        payload = {
            "mode": "quick" if quick else "full",
            "interrupted": interrupted,
            "totals": compute_totals(results),
            "candidates": [c.to_dict() for c in results],
        }
        Path(args.json_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON-отчёт: {args.json_path}")

    if interrupted:
        return 130
    totals = compute_totals(results)
    if args.strict and args.expect_min > 0 and totals["ok"] < args.expect_min:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
