#!/usr/bin/env python3
"""Точка входа для Платформы: python solution.py <input_dir> <output.geojson>.

Контракт — Приложение 2 Технического регламента КОЗ №3:
  - два позиционных аргумента, флагов нет;
  - вход: директория с территориями NNN_НАЗВАНИЕ_РЕГИОН/;
  - внутри — только LAS (суффикс La), без хиллшейдов и разметки;
  - выход: один GeoJSON в EPSG:3857.
"""

from __future__ import annotations

import sys
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(
            "usage: python solution.py <input_dir> <output_geojson>",
            file=sys.stderr,
        )
        return 2
    input_dir, output_file = Path(argv[0]), Path(argv[1])
    if not input_dir.is_dir():
        print(f"нет входной директории: {input_dir}", file=sys.stderr)
        return 2

    _ensure_expds()
    from expds.predict import main as predict_main

    return predict_main(
        [
            "--data",
            str(input_dir),
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


if __name__ == "__main__":
    sys.exit(main())
