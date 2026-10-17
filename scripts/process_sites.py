"""Пакетная обработка участков: находит все папки с UTM.json или LAS/GeoTIFF
и прогоняет build_site для каждого.

Использование:
    uv run python scripts/process_sites.py --data-dir /path/to/data
    uv run python scripts/process_sites.py --data-dir /path/to/data --only 006 017
    uv run python scripts/process_sites.py --data-dir /path/to/data --skip 101
    uv run python scripts/process_sites.py --data-dir /path/to/data \
        --output-crs EPSG:3857
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

from expds.features.build_site import build_site  # noqa: E402
from expds.features.config import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def find_sites(data_dir: Path) -> list[Path]:
    """Находит все папки-участки.

    Участок — папка, внутри которой есть UTM.json ИЛИ LAS/LAZ ИЛИ
    GeoTIFF в подпапке с кодом модальности.

    Args:
        data_dir: корневая папка с данными.

    Returns:
        Список папок-участков.
    """
    sites = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_dir():
            continue
        # Вариант 1: участок на верхнем уровне.
        if _is_site_dir(path):
            sites.append(path)
            continue
        # Вариант 2: одна вложенная папка с тем же именем.
        for sub in path.iterdir():
            if sub.is_dir() and _is_site_dir(sub):
                sites.append(sub)
                break
    return sites


def _is_site_dir(path: Path) -> bool:
    """Проверяет, выглядит ли папка как участок."""
    if (path / "UTM.json").exists():
        return True
    if list(path.rglob("*.las")) or list(path.rglob("*.laz")):
        return True
    # GeoTIFF в подпапке с кодом модальности.
    for sub in path.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        modality_codes = ("la_", "li_", "or_", "spor", "ae_", "maggr", "etgr", "terfl")
        if any(code in name for code in modality_codes):
            if list(sub.rglob("*.tif")) or list(sub.rglob("*.tiff")):
                return True
    return False


def read_crs(site: Path) -> str:
    """Читает UTM.json."""
    utm = site / "UTM.json"
    data = json.loads(utm.read_text(encoding="utf-8"))
    return data["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")


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
    grid.pixel_size. Остальное — из base_config.

    Args:
        base_config: словарь из configs/default.yaml.
        site: папка участка.
        pixel_size: размер пикселя.

    Returns:
        Копия base_config с подставленными значениями.
    """
    import copy

    import laspy

    # CRS: UTM.json > LAS header > base_config.
    site_crs: str | None = None
    if (site / "UTM.json").exists():
        site_crs = read_crs(site)

    # Лидар.
    las_candidates = list(site.rglob("*.las")) + list(site.rglob("*.laz"))
    las_path = las_candidates[0] if las_candidates else None

    z_unit = "meters"
    if las_path is not None:
        las = laspy.read(las_path)
        z_span = float(las.z.max() - las.z.min())
        z_unit = "feet" if z_span > 100 else "meters"
        las_crs = las.header.parse_crs()
        if las_crs is not None:
            las_crs_str = las_crs.to_string()
            if site_crs is not None and las_crs_str != site_crs:
                logger.warning(
                    "LAS CRS=%s, UTM.json=%s. Использую LAS CRS.",
                    las_crs_str,
                    site_crs,
                )
            site_crs = las_crs_str

    if site_crs is None:
        site_crs = base_config["crs"]["source"] or "EPSG:32637"
        logger.warning("CRS не найден для %s, использую %s", site.name, site_crs)

    name = transliterate(site.parent.name if site.parent.name else site.name)

    config = copy.deepcopy(base_config)
    config["dataset"]["name"] = name
    config["dataset"]["raw_dir"] = str(site)
    config["dataset"]["cache_dir"] = f"data/cache/{name}"
    config["dataset"]["lidar_file"] = las_path.name if las_path else "none.laz"
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
    output_crs: str | None = None,
) -> tuple[bool, str, float]:
    """Обрабатывает один участок. Возвращает (успех, сообщение, время)."""
    start = time.perf_counter()
    try:
        config_dict = patch_config(base_config, site, pixel_size=pixel_size)
        name = config_dict["dataset"]["name"]

        tmp_config = Path(f"/tmp/dataset_{name}.yaml")
        write_temp_config(config_dict, tmp_config)

        config = load_config(path=tmp_config)
        setup_logging(config=config.logging)

        result = build_site(
            raw_dir=site,
            out_dir=Path(config.dataset.cache_dir),
            config=config,
            max_points=None,
            output_crs=output_crs,
        )

        duration = time.perf_counter() - start
        n_layers = len(result.layers)
        return True, f"OK: {n_layers} слоёв", duration
    except Exception as exc:
        duration = time.perf_counter() - start
        return False, f"ОШИБКА: {type(exc).__name__}: {exc}", duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Пакетная обработка участков")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", help="Только участки с этими префиксами")
    parser.add_argument("--skip", nargs="*", help="Пропустить участки с этими префиксами")
    parser.add_argument(
        "--output-crs",
        type=str,
        default=None,
        help="Перепроецировать слои в указанную CRS (например, EPSG:3857)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    import yaml

    base_config_path = Path("configs/default.yaml")
    with base_config_path.open(encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    sites = find_sites(args.data_dir)
    print(f"Найдено участков: {len(sites)}")
    for s in sites:
        print(f"  {s.name}")
    print()

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
            site,
            base_config=base_config,
            pixel_size=args.pixel_size,
            output_crs=args.output_crs,
        )
        status = "OK" if success else "FAIL"
        print(f"  {status} | {message} | {duration:.1f} сек")
        print()
        results.append((site.name, success, message, duration))

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
