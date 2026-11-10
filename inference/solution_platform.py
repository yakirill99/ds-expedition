#!/usr/bin/env python3
"""Точка входа для Платформы: python solution.py <input_dir> <output.geojson>.

Контракт — Приложение 2 Технического регламента КОЗ №3:
  - путь к директории с валидационной выборкой и путь к выходному GeoJSON;
  - выход: GeoJSON FeatureCollection в EPSG:3857.

Платформа запускает этот файл из распакованного архива, а код (expds),
config.yaml и weights.pt лежат в образе по /app. Поэтому всё ищется
с фолбэком: сначала рядом со скриптом, затем в /app.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_APP = Path("/app")
_DEFAULT_INPUT = Path("/data/input")
_DEFAULT_OUTPUT = Path("/data/output/result.geojson")


def _pkg_dir(module: object) -> Path:
    """Каталог пакета.

    expds собран без __init__.py — это namespace-пакет: у него
    module.__file__ is None, а путь лежит в module.__path__.
    """
    f = getattr(module, "__file__", None)
    if f:
        return Path(f).resolve().parent
    paths = list(getattr(module, "__path__", []) or [])
    if not paths:
        raise ImportError(f"{module!r}: нет ни __file__, ни __path__")
    return Path(paths[0]).resolve()


def _ensure_expds() -> Path:
    """Кладёт каталог с пакетом expds в sys.path. Возвращает путь к пакету."""
    try:
        import expds

        return _pkg_dir(expds)
    except ImportError:
        pass
    for cand in (_HERE, _HERE.parent / "src", _APP):
        if (cand / "expds").is_dir():
            sys.path.insert(0, str(cand))
            import expds

            return _pkg_dir(expds)
    raise ImportError(f"пакет expds не найден (искали в {_HERE}, {_APP})")


def _asset(name: str, pkg_dir: Path) -> Path:
    """Ищет config.yaml / weights.pt рядом со скриптом, затем в образе."""
    for cand in (_HERE, pkg_dir.parent, _APP, Path.cwd()):
        path = cand / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{name} не найден (искали в {_HERE}, {pkg_dir.parent}, {_APP})")


def _mirror(src: Path, dst: Path) -> None:
    """Структурная копия дерева входа.

    Каталоги создаются заново: shutil.copytree переносит биты прав, и
    каталог с a-w остался бы незаписываемым — build_site не смог бы создать
    внутри layers/. Хардлинк дешевле копирования LAS; между разными томами
    os.link даёт EXDEV, тогда симлинк (LAS только читается), и лишь в
    последнюю очередь — копия.
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
                try:
                    os.symlink(entry, target)
                except OSError:
                    shutil.copyfile(entry, target)


def _writable_copy(input_dir: Path, dest: Path) -> Path:
    """Записываемая копия входа; исходный каталог не трогаем."""
    _mirror(input_dir, dest)
    return dest


def _write_empty(out: Path, crs: str = "EPSG:3857") -> None:
    """Пустой FeatureCollection в контракте выхода (крайний случай)."""
    epsg = crs.split(":")[-1] if ":" in crs else crs
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}},
        "features": [],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Платформа передаёт два пути. Если entry_point в metadata.json содержит
    # их же, придёт четыре — берём последние два (переданные платформой).
    if len(argv) >= 2:
        input_dir, output_file = Path(argv[-2]), Path(argv[-1])
    elif not argv:
        input_dir, output_file = _DEFAULT_INPUT, _DEFAULT_OUTPUT
        print(f"пути не переданы, использую {input_dir} -> {output_file}", flush=True)
    else:
        print("usage: python solution.py <input_dir> <output_geojson>", file=sys.stderr)
        return 2

    if not input_dir.is_dir():
        print(f"нет входной директории: {input_dir}", file=sys.stderr)
        return 2

    pkg_dir = _ensure_expds()
    config = _asset("config.yaml", pkg_dir)
    weights = _asset("weights.pt", pkg_dir)
    print(f"expds: {pkg_dir} | config: {config} | weights: {weights}", flush=True)

    from expds.predict import main as predict_main

    # Рабочая папка — рядом с выходом: единственная записываемая локация.
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
                str(config),
                "--weights",
                str(weights),
                "--device",
                "auto",
            ]
        )
    except Exception:
        # Непойманное исключение = код != 0 и нет файла: платформа тратит
        # попытку впустую. Пустой FeatureCollection даёт 0 баллов по региону,
        # но запуск остаётся валидным; traceback уходит в лог платформы.
        print("пайплайн упал — пишу пустой FeatureCollection", file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            _write_empty(out_path)
        except Exception:
            traceback.print_exc()
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
