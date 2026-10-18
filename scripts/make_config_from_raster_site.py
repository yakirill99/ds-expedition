"""Генерирует configs/dataset_<name>.yaml по папке растрового участка.

Аналог scripts/make_config_from_site.py, но для участков без LAS:
SpOr, Or, Ae, MagGr, EtGr, TerFl. CRS берётся из UTM.json, иначе
требуется явное указание --crs. Если у растра есть CRS в GeoTIFF —
предпочитаем его (UTM.json может врать).

Кроме генерации YAML, выводит инвентаризацию: какие модальности нашли,
сколько файлов, какие каналы в первом файле каждой модальности,
есть ли CRS, nodata.

Использование:
    uv run python scripts/make_config_from_raster_site.py \
        --site data/raw/048_Калмыкия_1 \
        --out configs/dataset_048.yaml

    # Если нет UTM.json и в GeoTIFF нет CRS:
    uv run python scripts/make_config_from_raster_site.py \
        --site data/raw/048_Калмыкия_1 \
        --out configs/dataset_048.yaml \
        --crs EPSG:32638
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# Константы: те же, что в build_site.py, чтобы детект модальности был единым.
_MODALITY_CODES = {
    "spor": "sp_or",
    "sp_or": "sp_or",
    "_or_": "or",
    "or_": "or",
    "_ae_": "ae",
    "ae_": "ae",
    "maggr": "maggr",
    "mag_gr": "maggr",
    "etgr": "etgr",
    "et_gr": "etgr",
    "terfl": "terfl",
    "ter_fl": "terfl",
}

# Константы: паттерны GeoTIFF.
_TIF_PATTERNS = ("*.tif", "*.tiff")


@dataclass(frozen=True)
class RasterInfo:
    """Информация по одному растровому файлу."""

    path: Path
    modality: str
    count: int
    crs: str | None
    nodata: float | None
    width: int
    height: int
    dtype: str


def _detect_modality(dir_name: str) -> str | None:
    """Определяет модальность по имени подпапки (как в build_site)."""
    name_lower = dir_name.lower()
    for code, modality in _MODALITY_CODES.items():
        if code in name_lower:
            return modality
    return None


def find_rasters(site: Path) -> dict[str, list[Path]]:
    """Находит GeoTIFF по модальностям.

    Args:
        site: папка участка.

    Returns:
        Словарь modality -> список путей (отсортированный).
    """
    result: dict[str, list[Path]] = {}
    for subdir in site.iterdir():
        if not subdir.is_dir():
            continue
        modality = _detect_modality(subdir.name)
        if modality is None:
            continue
        tifs: list[Path] = []
        for pattern in _TIF_PATTERNS:
            tifs.extend(sorted(subdir.rglob(pattern)))
        if tifs:
            result.setdefault(modality, []).extend(tifs)
    return result


def read_raster_info(path: Path, modality: str) -> RasterInfo:
    """Читает метаданные GeoTIFF без загрузки пикселей в память.

    Args:
        path: путь к GeoTIFF.
        modality: код модальности.

    Returns:
        RasterInfo.
    """
    import rasterio

    with rasterio.open(path) as src:
        crs_str = src.crs.to_string() if src.crs is not None else None
        return RasterInfo(
            path=path,
            modality=modality,
            count=src.count,
            crs=crs_str,
            nodata=float(src.nodata) if src.nodata is not None else None,
            width=src.width,
            height=src.height,
            dtype=str(src.dtypes[0]),
        )


def read_utm_crs(site: Path) -> str | None:
    """Читает CRS из UTM.json, если файл есть."""
    utm = site / "UTM.json"
    if not utm.exists():
        return None
    try:
        data = json.loads(utm.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    crs = data.get("crs")
    if not crs:
        return None
    return str(crs).replace("urn:ogc:def:crs:EPSG::", "EPSG:")


def detect_channels(info: RasterInfo) -> dict[str, str]:
    """Пробует угадать назначение каналов для спектральных индексов.

    Для 3-канального снимка (RGB) — red=0, green=1, blue=2, nir=None.
    Для 4-канального — предполагаем RGBA: red=0, green=1, blue=2,
    nir=None (4-й = alpha, а не NIR).

    Настоящий NIR обычно в multispectral-снимках (5+ каналов), но у нас
    таких данных нет. Поэтому nir_channel=None — NDVI/SAVI не строятся,
    пока не появится multispectral.

    Returns:
        Словарь с ключами red_channel/green_channel/blue_channel/
        nir_channel (значения int | None).
    """
    if info.count >= 5:
        # Предполагаем multispectral: R, G, B, NIR, ...
        return {
            "red_channel": "0",
            "green_channel": "1",
            "blue_channel": "2",
            "nir_channel": "3",
        }
    if info.count >= 3:
        # RGB(A). NIR нет.
        return {
            "red_channel": "0",
            "green_channel": "1",
            "blue_channel": "2",
            "nir_channel": "null",
        }
    # Grayscale: индексы бессмысленны.
    return {
        "red_channel": "null",
        "green_channel": "null",
        "blue_channel": "null",
        "nir_channel": "null",
    }


def transliterate(name: str) -> str:
    """Простая транслитерация (совпадает с make_config_from_site.py)."""
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
    result = result.replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in result if ch.isalnum() or ch == "_")


def make_config(
    site: Path,
    out: Path,
    name: str | None = None,
    pixel_size: float = 1.0,
    crs_override: str | None = None,
) -> None:
    """Генерирует YAML-конфиг для растрового участка."""
    rasters = find_rasters(site)
    if not rasters:
        raise FileNotFoundError(f"В {site} не найдено ни одной растровой модальности")

    print(f"Найдено модальностей: {len(rasters)}")
    infos: list[RasterInfo] = []
    for modality, paths in sorted(rasters.items()):
        info = read_raster_info(paths[0], modality)
        infos.append(info)
        print(
            f"  {modality}: {len(paths)} файлов, первый — {paths[0].name} "
            f"({info.count} кан., {info.width}x{info.height}, CRS={info.crs}, "
            f"dtype={info.dtype})"
        )
        if len(paths) > 1:
            print(
                f"    ВНИМАНИЕ: {len(paths)} файлов в модальности, build_site берёт только первый"
            )

    # CRS: override > CRS первого растра > UTM.json.
    crs: str | None = crs_override
    if crs is None:
        crs = next((i.crs for i in infos if i.crs is not None), None)
        if crs is None:
            crs = read_utm_crs(site)
    if crs is None:
        raise ValueError(
            f"Не удалось определить CRS участка {site}. "
            f"Укажи --crs (например, EPSG:32638) или положи UTM.json."
        )

    if name is None:
        name = transliterate(site.name if site.name else "raster_site")

    # Каналы для спектральных индексов — по первому растру с 3+ каналами.
    channels = {
        "red_channel": "null",
        "green_channel": "null",
        "blue_channel": "null",
        "nir_channel": "null",
    }
    for info in infos:
        if info.count >= 3:
            channels = detect_channels(info)
            break

    config_text = f"""# ============================================================
# Конфиг растрового участка {name}
# ============================================================
# Сгенерирован scripts/make_config_from_raster_site.py.
# Источник: {site}
# Модальности: {sorted(rasters.keys())}
# ============================================================


dataset:
  name: {name}
  raw_dir: {site}
  cache_dir: data/cache/{name}
  lidar_file: none.laz


lidar:
  source_z_unit: meters
  target_z_unit: meters
  source_xy_unit: meters
  chunk_size: 1000000
  max_points: null


crs:
  source: {crs}
  internal: {crs}
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
    - height_range
    - skewness_z
    - kurtosis_z
  nodata: -9999.0


align:
  default_method: bilinear
  max_shift: 5
  confidence_threshold: 0.3


spectral:
  enabled: [ndvi, sr, ari, gndvi, ndwi, savi]
  red_channel: {channels["red_channel"]}
  nir_channel: {channels["nir_channel"]}
  green_channel: {channels["green_channel"]}
  blue_channel: {channels["blue_channel"]}
  channel_std_min: 0.01
  nodata: -9999.0


modalities:
  enabled: [or, sp_or, ae, maggr, etgr, terfl]
  priority: [or, sp_or, ae, maggr, etgr, terfl]


output_crs:
  enabled: false
  epsg: EPSG:3857
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(config_text, encoding="utf-8")
    print(f"\nЗаписан: {out}")
    print(f"  CRS: {crs}")
    print(f"  Модальности: {sorted(rasters.keys())}")
    print(
        f"  nir_channel: {channels['nir_channel']} "
        f"(NDVI/SAVI {'строятся' if channels['nir_channel'] != 'null' else 'пропущены'})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Генерирует конфиг для растрового участка (SpOr/Or/Ae/...)"
    )
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument(
        "--crs",
        type=str,
        default=None,
        help="Переопределить CRS (например, EPSG:32638), если нет UTM.json и в TIF",
    )
    args = parser.parse_args()

    make_config(
        site=args.site,
        out=args.out,
        name=args.name,
        pixel_size=args.pixel_size,
        crs_override=args.crs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
