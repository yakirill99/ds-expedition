#!/usr/bin/env python3
"""Точка входа для Платформы: python solution.py <input_dir> <output.geojson>.

Контракт — Приложение 2 Технического регламента КОЗ №3:
  - два позиционных аргумента, флагов нет;
  - вход: NNN_НАЗВАНИЕ_РЕГИОНА/ с подпапками 01_.._La_точки/ (LAS);
  - выход: GeoJSON FeatureCollection в EPSG:3857.

Вход на Платформе смонтирован только на чтение (п. 7.5.1: запись разрешена
лишь в выделенный каталог результатов). build_site собирает layers/ in-place
рядом с LAS, поэтому работаем на записываемой копии во временной папке.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _ensure_expds() -> None:
    try:
        import expds  # noqa: F401
    except ImportError:
        for cand in (_HERE, _HERE.parent / "src"):
            if (cand / "expds").is_dir():
                sys.path.insert(0, str(cand))
                return
        raise


def _mirror(src: Path, dst: Path) -> None:
    """Структурная копия дерева входа.

    Каталоги создаются заново через mkdir: shutil.copytree переносит и биты
    прав, поэтому каталог с a-w остался бы незаписываемым и build_site не
    смог бы создать внутри layers/. Файлы — хардлинком, чтобы не копировать
    сотни мегабайт LAS (при разных ФС os.link падает с EXDEV, тогда копия).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if entry.is_dir():
            _mirror(entry, target)
        elif entry.is_file():
            try:
                os.link(entry, target)
            except OSError:
                # /data/input и /data/output — разные тома, os.link даёт EXDEV.
                # Симлинк дешевле копирования, а LAS только читается.
                try:
                    os.symlink(entry, target)
                except OSError:
                    shutil.copyfile(entry, target)


def _writable_copy(input_dir: Path, dest: Path) -> Path:
    """Записываемая копия входа; исходный каталог не трогаем."""
    _mirror(input_dir, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python solution.py <input_dir> <output_geojson>", file=sys.stderr)
        return 2
    input_dir, output_file = Path(argv[0]), Path(argv[1])
    if not input_dir.is_dir():
        print(f"нет входной директории: {input_dir}", file=sys.stderr)
        return 2

    _ensure_expds()
    from expds.predict import main as predict_main

    # Рабочая папка — рядом с выходом: это единственная записываемая
    # локация по п. 7.5.1. /tmp в контейнере может быть мал или noexec.
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".koz3_", dir=out_path.parent))
    try:
        data = _writable_copy(input_dir, work / "input")
        print(f"рабочая копия: {data}", flush=True)
        return predict_main(
            [
                "--data",
                str(data),
                "--out",
                str(output_file),
                "--config",
                str(_HERE / "config.yaml"),
                "--weights",
                str(_HERE / "weights.pt"),
                "--device",
                "auto",
            ]
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
