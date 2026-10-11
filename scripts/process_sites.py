"""Пакетная обработка участков: находит все папки с UTM.json в указанной
директории и прогоняет build_layers для каждого лидарного участка.

Использование:
    uv run python scripts/process_sites.py --data-dir /path/to/data
    uv run python scripts/process_sites.py --data-dir /path/to/data --only 006 017
    uv run python scripts/process_sites.py --data-dir /path/to/data --skip 101
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from expds.data.raster import write_raster  # noqa: E402
from expds.features.config import load_config, setup_logging  # noqa: E402
from expds.features.grid import DataSpec, stack_layers  # noqa: E402
from expds.utils.profiler import Profiler  # noqa: E402

logger = logging.getLogger(__name__)


def find_sites(data_dir: Path) -> list[Path]:
    """Находит все папки-участки: содержат UTM.json или подпапки с _La_, _SpOr_.

    Возвращает список папок, которые выглядят как участки.
    """
    sites = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_dir():
            continue
        # Участок может быть на верхнем уровне (с UTM.json) или внутри
        # (одна вложенная папка с тем же именем).
        if (path / "UTM.json").exists():
            sites.append(path)
        else:
            # Ищем вложенную папку.
            for sub in path.iterdir():
                if sub.is_dir() and (sub / "UTM.json").exists():
                    sites.append(sub)
                    break
    return sites


def find_las(site: Path) -> Path | None:
    """Находит LAS/LAZ в папке участка."""
    candidates = list(site.rglob("*.las")) + list(site.rglob("*.laz"))
    return candidates[0] if candidates else None


def read_crs(site: Path) -> str:
    """Читает UTM.json."""
    utm = site / "UTM.json"
    data = json.loads(utm.read_text(encoding="utf-8"))
    return data["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")


def read_las_crs(las_path: Path) -> str | None:
    """Читает CRS из LAS header, если есть."""
    import laspy

    las = laspy.open(las_path)
    crs = las.header.parse_crs()
    las.close()
    return crs.to_string() if crs else None


def transliterate(name: str) -> str:
    """Простая транслитерация в латиницу."""
    table = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
        "А": "A",
        "Б": "B",
        "В": "V",
        "Г": "G",
        "Д": "D",
        "Е": "E",
        "Ё": "E",
        "Ж": "Zh",
        "З": "Z",
        "И": "I",
        "Й": "Y",
        "К": "K",
        "Л": "L",
        "М": "M",
        "Н": "N",
        "О": "O",
        "П": "P",
        "Р": "R",
        "С": "S",
        "Т": "T",
        "У": "U",
        "Ф": "F",
        "Х": "Kh",
        "Ц": "Ts",
        "Ч": "Ch",
        "Ш": "Sh",
        "Щ": "Shch",
        "Ъ": "",
        "Ы": "Y",
        "Ь": "",
        "Э": "E",
        "Ю": "Yu",
        "Я": "Ya",
    }
    result = "".join(table.get(ch, ch) for ch in name)
    return "".join(
        ch for ch in result.replace(" ", "_").replace("-", "_") if ch.isalnum() or ch == "_"
    )


def patch_config(
    base_config: dict,
    site: Path,
    pixel_size: float = 1.0,
) -> dict:
    """Подставляет значения участка в базовый конфиг.

    Меняет только dataset.*, crs.*, lidar.source_z_unit, logging.log_file,
    grid.pixel_size. Остальное — из base_config (configs/default.yaml).

    Args:
        base_config: словарь из configs/default.yaml.
        site: папка участка.
        pixel_size: размер пикселя.

    Returns:
        Копия base_config с подставленными значениями.
    """
    import copy

    import laspy

    las_path = find_las(site)
    if las_path is None:
        raise FileNotFoundError(f"LAS/LAZ не найден в {site}")

    site_crs = read_crs(site)
    las_crs = read_las_crs(las_path)
    if las_crs is None:
        logger.info("LAS без CRS. Использую CRS из UTM.json: %s", site_crs)
    elif las_crs != site_crs:
        logger.warning("LAS CRS=%s, UTM.json=%s. Использую LAS CRS.", las_crs, site_crs)
        site_crs = las_crs

    las = laspy.read(las_path)
    z_span = float(las.z.max() - las.z.min())
    z_unit = "feet" if z_span > 100 else "meters"

    name = transliterate(site.parent.name)

    config = copy.deepcopy(base_config)
    config["dataset"]["name"] = name
    config["dataset"]["raw_dir"] = str(las_path.parent)
    config["dataset"]["cache_dir"] = f"data/cache/{name}"
    config["dataset"]["lidar_file"] = las_path.name
    config["lidar"]["source_z_unit"] = z_unit
    config["crs"]["source"] = site_crs
    config["crs"]["internal"] = site_crs
    config["logging"]["log_file"] = f"logs/{name}.log"
    config["grid"]["pixel_size"] = pixel_size
    return config


def write_temp_config(config_dict: dict, path: Path) -> None:
    """Записывает словарь конфига в YAML."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)


def process_site(
    site: Path,
    base_config: dict,
    pixel_size: float = 1.0,
) -> tuple[bool, str, float]:
    """Обрабатывает один участок. Возвращает (успех, сообщение, время)."""
    from scripts.build_layers import _run_pipeline

    start = time.perf_counter()
    try:
        config_dict = patch_config(base_config, site, pixel_size=pixel_size)
        name = config_dict["dataset"]["name"]

        # Пишем временный конфиг.
        tmp_config = Path(f"/tmp/dataset_{name}.yaml")
        write_temp_config(config_dict, tmp_config)

        # Загружаем конфиг.
        config = load_config(path=tmp_config)
        setup_logging(config=config.logging)

        profiler = Profiler()
        result = _run_pipeline(
            config=config,
            use_cache=False,
            limit_points=None,
            profiler=profiler,
        )

        # Сохраняем слои.
        output_dir = config.dataset.cache_dir / "layers"
        for name_layer, layer in result.layers.items():
            write_raster(layer=layer, path=output_dir / f"{name_layer}.tif")

        # Сохраняем Sample.
        import numpy as np

        sample_dir = config.dataset.cache_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(result.layers.keys())
        spec = DataSpec(grid=result.layers["dtm"].grid, channel_names=tuple(names))
        sample = stack_layers(layers=[result.layers[n] for n in names], spec=spec)
        np.savez_compressed(
            sample_dir / "sample.npz",
            tensor=sample.tensor,
            channel_mask=sample.channel_mask,
        )
        meta = {
            "channel_names": list(sample.channel_names),
            "channel_mask": sample.channel_mask.tolist(),
            "grid": {
                "crs": result.layers["dtm"].grid.crs,
                "pixel_size": result.layers["dtm"].grid.pixel_size,
                "x_min": result.layers["dtm"].grid.x_min,
                "y_min": result.layers["dtm"].grid.y_min,
                "width": result.layers["dtm"].grid.width,
                "height": result.layers["dtm"].grid.height,
            },
            "nodata": config.grid.nodata,
            "n_channels": sample.n_channels,
            "n_available_channels": sample.n_available_channels,
        }
        with (sample_dir / "sample_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        duration = time.perf_counter() - start
        return True, f"OK: {len(result.layers)} слоёв", duration
    except Exception as exc:
        duration = time.perf_counter() - start
        return False, f"ОШИБКА: {type(exc).__name__}: {exc}", duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Пакетная обработка участков")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", help="Только участки с этими префиксами")
    parser.add_argument("--skip", nargs="*", help="Пропустить участки с этими префиксами")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    import yaml

    # Загружаем базовый конфиг.
    base_config_path = Path("configs/default.yaml")
    with base_config_path.open(encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    sites = find_sites(args.data_dir)
    print(f"Найдено участков: {len(sites)}")
    for s in sites:
        print(f"  {s.name}")
    print()

    # Фильтрация.
    if args.only:
        sites = [s for s in sites if any(o in s.name for o in args.only)]
    if args.skip:
        sites = [s for s in sites if not any(k in s.name for k in args.skip)]

    print(f"К обработке: {len(sites)}")
    print()

    results = []
    for i, site in enumerate(sites, 1):
        print(f"=== [{i}/{len(sites)}] {site.name} ===")
        success, message, duration = process_site(
            site, base_config=base_config, pixel_size=args.pixel_size
        )
        status = "OK" if success else "FAIL"
        print(f"  {status} | {message} | {duration:.1f} сек")
        print()
        results.append((site.name, success, message, duration))

    # Отчёт.
    print("=" * 60)
    print("ИТОГ")
    print("=" * 60)
    n_ok = sum(1 for _, s, _, _ in results if s)
    print(f"Успешно: {n_ok} / {len(results)}")
    for name, success, message, duration in results:
        marker = "✓" if success else "✗"
        print(f"  {marker} {name}: {duration:.1f} сек — {message}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
