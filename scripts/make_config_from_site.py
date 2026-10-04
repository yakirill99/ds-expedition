"""Генерирует configs/dataset_<name>.yaml по структуре папки участка.

Поддерживает только лидарные участки (папка La_точки с .las/.laz).
Для растровых — пока не поддерживается.

Использование:
    uv run python scripts/make_config_from_site.py \
        --site /path/to/013_Нора_Виннон/013_Нора_Виннон \
        --out configs/dataset_013.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_las(site: Path) -> Path:
    """Находит LAS/LAZ в папке участка."""
    candidates = list(site.rglob("*.las")) + list(site.rglob("*.laz"))
    if not candidates:
        raise FileNotFoundError(f"LAS/LAZ не найден в {site}")
    if len(candidates) > 1:
        print(f"Найдено {len(candidates)} LAS, берём первый: {candidates[0].name}")
    return candidates[0]


def read_crs(site: Path) -> str:
    """Читает UTM.json в папке участка."""
    utm = site / "UTM.json"
    if not utm.exists():
        raise FileNotFoundError(f"UTM.json не найден в {site}")
    data = json.loads(utm.read_text(encoding="utf-8"))
    return data["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")


def read_las_crs(las_path: Path) -> str | None:
    """Читает CRS из LAS header, если есть."""
    import laspy

    las = laspy.open(las_path)
    crs = las.header.parse_crs()
    las.close()
    if crs is None:
        return None
    return crs.to_string()


def detect_z_unit(las_path: Path) -> str:
    """Определяет единицы Z (feet/meters) по диапазону."""
    import laspy

    las = laspy.read(las_path)
    span = float(las.z.max() - las.z.min())
    # Эвристика: > 100 при малом участке = футы.
    if span > 100:
        print(f"Диапазон Z = {span:.1f}, похоже на футы. Ставлю feet.")
        return "feet"
    return "meters"


def transliterate(name: str) -> str:
    """Простая транслитерация русского имени в латиницу.

    Для путей, логов, имён датасетов. Не для людей — для файловой системы.

    Args:
        name: исходное имя (может содержать кириллицу).

    Returns:
        Латиница с подчёркиваниями.
    """
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
    # Заменяем пробелы и дефисы на подчёркивания.
    result = result.replace(" ", "_").replace("-", "_")
    # Убираем всё, кроме букв, цифр, подчёркиваний.
    result = "".join(ch for ch in result if ch.isalnum() or ch == "_")
    return result


def make_config(
    site: Path,
    out: Path,
    name: str | None = None,
    pixel_size: float = 1.0,
) -> None:
    """Генерирует YAML-конфиг."""
    las_path = find_las(site)
    site_crs = read_crs(site)
    las_crs = read_las_crs(las_path)
    if las_crs and las_crs != site_crs:
        print(f"ВНИМАНИЕ: LAS CRS = {las_crs}, UTM.json = {site_crs}. Использую LAS CRS.")
        site_crs = las_crs
    z_unit = detect_z_unit(las_path)

    if name is None:
        name = transliterate(site.parent.name)

    las_dir = las_path.parent
    config_text = f"""# ============================================================
# Конфиг датасета {name} (реальный лидар)
# ============================================================
# Сгенерирован scripts/make_config_from_site.py.
# Источник: {las_path}
# ============================================================


dataset:
  name: {name}
  raw_dir: {las_dir}
  cache_dir: data/cache/{name}
  lidar_file: {las_path.name}


lidar:
  source_z_unit: {z_unit}
  target_z_unit: meters
  source_xy_unit: meters
  chunk_size: 1000000
  max_points: null


crs:
  source: {site_crs}
  internal: {site_crs}
  output: EPSG:3857


grid:
  pixel_size: {pixel_size}
  nodata: -9999.0


logging:
  level: INFO
  format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
  datefmt: "%Y-%m-%d %H:%M:%S"
  log_file: logs/{name}.log


ground:
  cell_size: 1.0
  preset: soft
  window_sizes: [3, 5, 7, 9, 11]
  height_threshold: 0.5
  slope_threshold: 0.2
  max_iterations: 5
  min_change_ratio: 0.001


dtm:
  statistic: mean
  fill_method: idw
  idw_k: 8
  idw_power: 2.0
  max_hole_pixels: 1000
  artifact_window: 5
  artifact_threshold: 2.0


relief:
  hillshade_azimuths: [45, 135, 225, 315]
  hillshade_altitude: 45.0
  hillshade_z_factor: 1.0
  slrm_sigmas: [2.0, 8.0, 32.0]
  openness_radius: 10
  openness_n_directions: 16
  curvature_sigmas: [2.0, 8.0]
  tpi_radii: [5, 15]
  tri_radii: [3, 9]


cloud_features:
  enabled:
    - point_density
    - mean_intensity
    - std_intensity
    - non_first_return_ratio
    - mean_return_number
    - mean_z
    - z_std
  nodata: -9999.0


align:
  default_method: bilinear
  max_shift: 5
  confidence_threshold: 0.3
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(config_text, encoding="utf-8")
    print(f"Записан: {out}")
    print(f"  LAS: {las_path}")
    print(f"  CRS: {site_crs}")
    print(f"  Z unit: {z_unit}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Генерирует конфиг для лидарного участка")
    parser.add_argument(
        "--site", type=Path, required=True, help="Папка участка (верхний уровень с UTM.json)"
    )
    parser.add_argument("--out", type=Path, required=True, help="Путь к выходному YAML")
    parser.add_argument(
        "--name", type=str, default=None, help="Имя датасета (по умолчанию имя папки)"
    )
    parser.add_argument("--pixel-size", type=float, default=1.0, help="Размер пикселя в метрах")
    args = parser.parse_args()

    make_config(site=args.site, out=args.out, name=args.name, pixel_size=args.pixel_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
