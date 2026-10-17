"""Универсальный адаптер build_site: сырьё участка -> layers/ + sample.npz.

Заменяет разрозненные scripts/build_layers.py (только лидар) и
scripts/prepare_raster_site.py (только растры). Одна функция, один контракт:

    build_site(raw_dir, out_dir, cfg) -> SiteResult

Что делает:
1. Инвентаризация папки участка: находит LAS/LAZ, GeoTIFF-модальности
   (or, sp_or, ae, maggr, etgr, terfl), UTM.json.
2. Определяет CRS (LAS header > UTM.json > конфиг).
3. Строит целевую сетку: по bounds лидара, если он есть; иначе — по
   первому растру (с учётом modalities.priority).
4. Лидар (если есть): читает -> фильтр земли -> DTM -> рельеф ->
   cloud features.
5. Растры (если есть): читает GeoTIFF -> ресэмплит в общую сетку ->
   спектральные индексы (если каналы есть) -> слои с префиксом
   модальности (or_, sp_or_, maggr_).
6. Собирает Sample с valid_mask (попиксельная валидность).
7. Сохраняет layers/*.tif и samples/sample.npz + sample_meta.json.

Геофизика (MagGr, EtGr, TerFl) обрабатывается как обычные GeoTIFF:
каждый канал -> слой. Без спектральных индексов (они для оптики).

Лидар — единственная модальность, которая строит производные рельефа
и cloud features. Остальные — «как есть» каналы.

Границы приличия:
- Если лидара нет и растров нет — ValueError.
- Если лидар есть, но фильтр земли отбросил всё — ValueError.
- Если у растра нет CRS и нет UTM.json — ValueError.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expds.data.las import read_header, read_points
from expds.data.raster import read_raster_meta, write_raster
from expds.features.cloud_features import build_cloud_features
from expds.features.config import AppConfig
from expds.features.dtm import build_dtm
from expds.features.grid import DataSpec, Grid, Layer, stack_layers
from expds.features.ground import filter_ground
from expds.features.relief import build_relief_layers
from expds.features.spectral import build_spectral_indices

logger = logging.getLogger(__name__)

# Константы: имена модальностей.
MODALITY_LIDAR = "li"
MODALITY_OR = "or"
MODALITY_SP_OR = "sp_or"
MODALITY_AE = "ae"
MODALITY_MAGGR = "maggr"
MODALITY_ETGR = "etgr"
MODALITY_TERFL = "terfl"

# Константы: префиксы для слоёв растровых модальностей.
MODALITY_PREFIX = {
    MODALITY_OR: "or",
    MODALITY_SP_OR: "sp_or",
    MODALITY_AE: "ae",
    MODALITY_MAGGR: "maggr",
    MODALITY_ETGR: "etgr",
    MODALITY_TERFL: "terfl",
}

# Константы: имена служебных файлов.
_UTM_JSON = "UTM.json"
_SAMPLE_NPZ = "sample.npz"
_SAMPLE_META = "sample_meta.json"
_LAYERS_DIR = "layers"
_SAMPLES_DIR = "samples"

# Константы: паттерны файлов.
_LAS_PATTERNS = ("*.las", "*.laz")
_TIF_PATTERNS = ("*.tif", "*.tiff")

# Константы: nodata для служебных масок.
_MASK_NODATA = -1.0


@dataclass(frozen=True)
class SiteInventory:
    """Результат инвентаризации папки участка.

    lidar_path — путь к LAS/LAZ или None.
    rasters — словарь modality -> список путей к GeoTIFF.
    utm_json — путь к UTM.json или None.
    crs — CRS участка (из LAS header, UTM.json или конфига).
    """

    site_dir: Path
    lidar_path: Path | None
    rasters: dict[str, list[Path]]
    utm_json: Path | None
    crs: str


@dataclass
class SiteResult:
    """Результат build_site.

    layers — словарь имя -> Layer (все слои в общей сетке).
    sample — Sample для модели (tensor + channel_mask + valid_mask).
    grid — общая сетка.
    inventory — что нашли в папке.
    n_points_total, n_points_ground — статистика по лидару (0 если нет).
    """

    layers: dict[str, Layer]
    sample: object  # Sample, но чтобы не тянуть импорт в dataclass
    grid: Grid
    inventory: SiteInventory
    n_points_total: int = 0
    n_points_ground: int = 0


# ============================================================
# Инвентаризация
# ============================================================


def _read_utm_crs(site_dir: Path) -> str | None:
    """Читает CRS из UTM.json, если файл есть.

    Args:
        site_dir: папка участка.

    Returns:
        Строка CRS (EPSG:XXXX) или None.
    """
    utm_path = site_dir / _UTM_JSON
    if not utm_path.exists():
        return None
    try:
        data = json.loads(utm_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Не удалось прочитать %s: %s", utm_path, exc)
        return None
    crs = data.get("crs")
    if not crs:
        return None
    return str(crs).replace("urn:ogc:def:crs:EPSG::", "EPSG:")


def _detect_modality_from_dir(dir_name: str) -> str | None:
    """Определяет модальность по имени подпапки.

    Имена папок в данных хакатона содержат код модальности:
    013_La_..., 048_SpOr_..., 101_Ae_..., 101_Li_... и т.п.

    Args:
        dir_name: имя подпапки.

    Returns:
        Код модальности или None.
    """
    name_lower = dir_name.lower()
    # Порядок важен: sp_or проверяем раньше or.
    if "spor" in name_lower or "sp_or" in name_lower:
        return MODALITY_SP_OR
    if "_or_" in name_lower or name_lower.startswith("or_"):
        return MODALITY_OR
    if "_ae_" in name_lower or name_lower.startswith("ae_"):
        return MODALITY_AE
    if "maggr" in name_lower or "mag_gr" in name_lower:
        return MODALITY_MAGGR
    if "etgr" in name_lower or "et_gr" in name_lower:
        return MODALITY_ETGR
    if "terfl" in name_lower or "ter_fl" in name_lower:
        return MODALITY_TERFL
    if "_la_" in name_lower or name_lower.startswith("la_"):
        return MODALITY_LIDAR
    if "_li_" in name_lower or name_lower.startswith("li_"):
        # Li-карты — визуализации, не сырьё. Не путать с лидаром.
        return "li_maps"
    return None


def inventory_site(
    site_dir: Path,
    config: AppConfig,
) -> SiteInventory:
    """Инвентаризация: что есть в папке участка.

    Ищет LAS/LAZ (рекурсивно), GeoTIFF по модальностям (по именам папок),
    UTM.json. Определяет CRS: LAS header > UTM.json > config.crs.source.

    Args:
        site_dir: папка участка.
        config: конфиг приложения.

    Returns:
        SiteInventory.

    Raises:
        FileNotFoundError: если папки нет.
        ValueError: если не нашли ни лидара, ни растров.
    """
    if not site_dir.exists():
        raise FileNotFoundError(f"Папка участка не найдена: {site_dir}")

    # Лидар.
    lidar_candidates: list[Path] = []
    for pattern in _LAS_PATTERNS:
        lidar_candidates.extend(site_dir.rglob(pattern))
    lidar_path = lidar_candidates[0] if lidar_candidates else None
    if len(lidar_candidates) > 1:
        logger.warning(
            "Найдено %d LAS/LAZ, берём первый: %s",
            len(lidar_candidates),
            lidar_path.name,
        )

    # Растры по модальностям.
    rasters: dict[str, list[Path]] = {}
    for subdir in site_dir.iterdir():
        if not subdir.is_dir():
            continue
        modality = _detect_modality_from_dir(subdir.name)
        if modality is None or modality in (MODALITY_LIDAR, "li_maps"):
            # Li-карты — визуализации, не используем как сырьё.
            continue
        tifs: list[Path] = []
        for pattern in _TIF_PATTERNS:
            tifs.extend(sorted(subdir.rglob(pattern)))
        if tifs:
            rasters.setdefault(modality, []).extend(tifs)

    if lidar_path is None and not rasters:
        raise ValueError(
            f"В папке {site_dir} не найдено ни LAS/LAZ, ни GeoTIFF. Проверь структуру участка."
        )

    # CRS: LAS header > UTM.json > config.
    crs: str | None = None
    if lidar_path is not None:
        header = read_header(path=lidar_path)
        crs = header.crs
        if crs is not None:
            logger.info("CRS из LAS header: %s", crs)

    utm_json = site_dir / _UTM_JSON
    if not utm_json.exists():
        utm_json = None

    if crs is None:
        crs = _read_utm_crs(site_dir=site_dir)
        if crs is not None:
            logger.info("CRS из UTM.json: %s", crs)

    if crs is None:
        crs = config.crs.source
        logger.warning("CRS не найден в данных, используем config.crs.source: %s", crs)

    inventory = SiteInventory(
        site_dir=site_dir,
        lidar_path=lidar_path,
        rasters=rasters,
        utm_json=utm_json,
        crs=crs,
    )

    logger.info(
        "Инвентаризация %s: lidar=%s, растры=%s, CRS=%s",
        site_dir.name,
        lidar_path.name if lidar_path else "нет",
        {k: len(v) for k, v in rasters.items()},
        crs,
    )
    return inventory


# ============================================================
# Лидар
# ============================================================


def _read_all_lidar_points(
    path: Path,
    config: AppConfig,
    target_crs: str,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Читает все точки LAS в массивы.

    Args:
        path: путь к LAS/LAZ.
        config: конфиг приложения.
        target_crs: CRS для внутренних расчётов.
        max_points: ограничение числа точек.

    Returns:
        Кортеж (x, y, z, intensity, return_number).
    """
    chunks = []
    for block in read_points(
        path=path,
        chunk_size=config.lidar.chunk_size,
        source_crs=config.crs.source,
        target_crs=target_crs,
        source_z_unit=config.lidar.source_z_unit,
        max_points=max_points,
    ):
        chunks.append(block)

    if not chunks:
        raise ValueError(f"LAS не дал ни одного чанка: {path}")

    all_points = np.vstack(chunks)
    x = all_points[:, 0].astype(np.float64)
    y = all_points[:, 1].astype(np.float64)
    z = all_points[:, 2].astype(np.float64)
    intensity = all_points[:, 3].astype(np.float32)
    return_number = all_points[:, 4].astype(np.uint8)
    logger.info(
        "Прочитано точек: %d | X=(%.1f, %.1f) | Y=(%.1f, %.1f) | Z=(%.2f, %.2f)",
        len(x),
        x.min(),
        x.max(),
        y.min(),
        y.max(),
        z.min(),
        z.max(),
    )
    return x, y, z, intensity, return_number


def _build_lidar_layers(
    inventory: SiteInventory,
    config: AppConfig,
    grid: Grid,
    max_points: int | None,
) -> tuple[dict[str, Layer], int, int]:
    """Строит слои из лидара.

    Args:
        inventory: инвентаризация участка (lidar_path обязателен).
        config: конфиг.
        grid: целевая сетка.
        max_points: ограничение числа точек.

    Returns:
        Кортеж (layers, n_points_total, n_points_ground).
    """
    assert inventory.lidar_path is not None

    x, y, z, intensity, return_number = _read_all_lidar_points(
        path=inventory.lidar_path,
        config=config,
        target_crs=grid.crs,
        max_points=max_points,
    )
    n_total = len(x)

    ground_result = filter_ground(
        x=x,
        y=y,
        z=z,
        config=config.ground,
        crs=grid.crs,
    )
    ground_mask = ground_result.ground_mask
    n_ground = int(ground_mask.sum())
    if n_ground == 0:
        raise ValueError(
            f"Фильтр земли отбросил все точки ({n_total}). Проверь пресет ground и данные."
        )
    logger.info("Земля: %d из %d (%.1f%%)", n_ground, n_total, 100.0 * n_ground / n_total)

    x_g = x[ground_mask]
    y_g = y[ground_mask]
    z_g = z[ground_mask]

    dtm_result = build_dtm(
        x=x_g,
        y=y_g,
        z=z_g,
        grid=grid,
        config=config.dtm,
        source=str(inventory.lidar_path),
    )
    dtm_layer = dtm_result.dtm

    layers: dict[str, Layer] = {
        "dtm": dtm_layer,
        # hole_mask и count идут в layers/: hole_mask как канал для модели
        # (закрывает TODO), count — для отладки.
        "hole_mask": dtm_result.hole_mask,
        "count": dtm_result.count,
    }

    relief_arrays = build_relief_layers(
        dtm=dtm_layer.data,
        pixel_size=grid.pixel_size,
        hillshade_azimuths=config.relief.hillshade_azimuths,
        hillshade_altitude=config.relief.hillshade_altitude,
        hillshade_z_factor=config.relief.hillshade_z_factor,
        slrm_sigmas=config.relief.slrm_sigmas,
        openness_radius=config.relief.openness_radius,
        openness_n_directions=config.relief.openness_n_directions,
        curvature_sigmas=config.relief.curvature_sigmas,
        tpi_radii=config.relief.tpi_radii,
        tri_radii=config.relief.tri_radii,
        nodata=dtm_layer.nodata,
    )
    for name, arr in relief_arrays.items():
        layers[name] = Layer(
            name=name,
            data=arr,
            grid=grid,
            nodata=config.grid.nodata,
            source=str(inventory.lidar_path),
            license="unknown",
            unit="unitless",
        )

    cloud_result = build_cloud_features(
        x=x,
        y=y,
        z=z,
        intensity=intensity,
        return_number=return_number,
        grid=grid,
        enabled=config.cloud_features.enabled,
        nodata=config.cloud_features.nodata,
    )
    layers.update(cloud_result.layers)

    logger.info("Лидар: %d слоёв", len(layers))
    return layers, n_total, n_ground


# ============================================================
# Растры (оптика, геофизика)
# ============================================================


def _grid_from_raster(path: Path, pixel_size: float, target_crs: str) -> Grid:
    """Строит целевую сетку по bounds растра, с перепроецированием в target_crs.

    Если CRS растра не совпадает с target_crs — bounds перепроецируются
    через pyproj. Это нужно, когда лидара нет, а растр в другой CRS.

    Args:
        path: путь к GeoTIFF.
        pixel_size: размер пикселя в target_crs.
        target_crs: целевая CRS.

    Returns:
        Grid.

    Raises:
        ValueError: если у растра нет CRS.
    """
    meta = read_raster_meta(path=path)
    if meta.crs is None:
        raise ValueError(f"Растр {path.name} без CRS. Нужен UTM.json или CRS в файле.")

    if meta.crs == target_crs:
        bounds = meta.bounds
    else:
        from pyproj import Transformer

        transformer = Transformer.from_crs(meta.crs, target_crs, always_xy=True)
        x_min, y_min, x_max, y_max = meta.bounds
        corners_x = [x_min, x_min, x_max, x_max]
        corners_y = [y_min, y_max, y_min, y_max]
        xs_t, ys_t = transformer.transform(corners_x, corners_y)
        bounds = (
            float(min(xs_t)),
            float(min(ys_t)),
            float(max(xs_t)),
            float(max(ys_t)),
        )
        logger.info("Bounds растра перепроецированы: %s -> %s", meta.crs, target_crs)

    return Grid.from_bounds(crs=target_crs, pixel_size=pixel_size, bounds=bounds)


def _read_raster_channels(
    path: Path,
    grid: Grid,
    nodata: float,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Читает все каналы растра и приводит к целевой сетке.

    Args:
        path: путь к GeoTIFF.
        grid: целевая сетка.
        nodata: значение nodata в результате.

    Returns:
        Кортеж (channels, valid_mask):
        channels — словарь 0-based index -> 2D массив;
        valid_mask — булева маска валидных пикселей (True = валидно).
    """
    import rasterio

    channels: dict[int, np.ndarray] = {}
    valid_mask = np.zeros(grid.shape, dtype=bool)

    with rasterio.open(path) as src:
        n_bands = src.count
        src_crs = src.crs.to_string() if src.crs is not None else None

    for band_idx in range(n_bands):
        # read_raster читает только 1-й канал. Для многоканальных читаем
        # каждый канал отдельно через _read_single_band_to_grid.
        band_data = _read_single_band_to_grid(
            path=path,
            band_idx=band_idx,
            grid=grid,
            nodata=nodata,
            src_crs=src_crs,
        )
        channels[band_idx] = band_data
        valid_mask |= band_data != nodata

    logger.info(
        "Растр %s: %d каналов, валидных пикселей %.1f%%",
        path.name,
        len(channels),
        100.0 * float(valid_mask.mean()),
    )
    return channels, valid_mask


def _read_single_band_to_grid(
    path: Path,
    band_idx: int,
    grid: Grid,
    nodata: float,
    src_crs: str | None,
) -> np.ndarray:
    """Читает один канал растра и ресэмплит в целевую сетку.

    Использует rasterio.warp.reproject. Если CRS растра не совпадает
    с grid.crs — перепроецирует.

    Args:
        path: путь к GeoTIFF.
        band_idx: 0-based индекс канала.
        grid: целевая сетка.
        nodata: значение nodata в результате.
        src_crs: CRS растра (для проверки).

    Returns:
        2D float32 массив в grid.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from expds.features.grid import grid_to_affine

    with rasterio.open(path) as src:
        source = src.read(band_idx + 1).astype(np.float32)
        src_transform = src.transform
        src_crs_obj = src.crs
        src_nodata = src.nodata

    dst_transform = grid_to_affine(grid)
    dst_crs_obj = CRS.from_user_input(grid.crs)

    destination = np.full(grid.shape, nodata, dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs_obj,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs_obj,
        dst_nodata=nodata,
        resampling=Resampling.bilinear,
    )
    # GeoTIFF north-up -> Grid (row 0 = юг).
    return np.ascontiguousarray(np.flipud(destination))


def _build_raster_layers(
    inventory: SiteInventory,
    config: AppConfig,
    grid: Grid,
) -> dict[str, Layer]:
    """Строит слои из растровых модальностей.

    Для каждой модальности:
    - читает все каналы -> ресэмплит в общую сетку;
    - если это оптика (or, sp_or, ae) и есть спектральные индексы —
      строит NDVI/SR/ARI;
    - именует слои с префиксом модальности: or_ch0, sp_or_ndvi, maggr_ch0.

    Args:
        inventory: инвентаризация участка.
        config: конфиг.
        grid: целевая сетка.

    Returns:
        Словарь имя -> Layer.
    """
    layers: dict[str, Layer] = {}

    for modality, paths in inventory.rasters.items():
        prefix = MODALITY_PREFIX.get(modality, modality)
        # Для каждой модальности берём первый файл (главный).
        # Дополнительные TIF с другим разрешением пока не используем.
        path = paths[0]
        if len(paths) > 1:
            logger.info(
                "Модальность %s: %d файлов, берём первый: %s",
                modality,
                len(paths),
                path.name,
            )

        channels, valid_mask = _read_raster_channels(
            path=path,
            grid=grid,
            nodata=config.grid.nodata,
        )

        # Базовые каналы: prefix_ch0, prefix_ch1, ...
        for idx, data in channels.items():
            name = f"{prefix}_ch{idx}"
            layers[name] = Layer(
                name=name,
                data=data,
                grid=grid,
                nodata=config.grid.nodata,
                source=str(path),
                license="unknown",
                unit="uint8",
            )

        # Спектральные индексы — только для оптики.
        if modality in (MODALITY_OR, MODALITY_SP_OR, MODALITY_AE):
            spectral_result = build_spectral_indices(
                channels=channels,
                valid_mask=valid_mask,
                grid=grid,
                config=config.spectral,
                source=str(path),
            )
            for index_name, index_layer in spectral_result.layers.items():
                renamed = Layer(
                    name=f"{prefix}_{index_name}",
                    data=index_layer.data,
                    grid=grid,
                    nodata=config.grid.nodata,
                    source=str(path),
                    license="unknown",
                    unit=index_layer.unit,
                )
                layers[renamed.name] = renamed

    logger.info("Растры: %d слоёв", len(layers))
    return layers


# ============================================================
# Сборка и сохранение
# ============================================================


def _order_channel_names(names: list[str]) -> list[str]:
    """Упорядочивает каналы: dtm, hole_mask, рельеф, cloud, растры.

    Внутри групп — по алфавиту. hole_mask идёт сразу после dtm: это
    маска валидности, модель должна видеть её рядом с высотами.

    Args:
        names: список имён слоёв.

    Returns:
        Упорядоченный список.
    """
    ordered: list[str] = []

    if "dtm" in names:
        ordered.append("dtm")

    relief_prefixes = (
        "slope",
        "aspect",
        "hillshade_",
        "slrm_",
        "positive_openness",
        "negative_openness",
        "sky_view_factor",
        "curvature_",
        "tpi_",
        "tri_",
    )
    relief = sorted(n for n in names if any(n.startswith(p) for p in relief_prefixes))
    ordered.extend(relief)

    cloud_prefixes = (
        "point_density",
        "mean_intensity",
        "std_intensity",
        "non_first_return_ratio",
        "mean_return_number",
        "mean_z",
        "z_std",
    )
    cloud = sorted(n for n in names if any(n.startswith(p) for p in cloud_prefixes))
    ordered.extend(cloud)

    remaining = sorted(set(names) - set(ordered))
    ordered.extend(remaining)
    return ordered


def _save_site(
    layers: dict[str, Layer],
    grid: Grid,
    out_dir: Path,
    nodata: float,
) -> object:
    """Сохраняет layers/*.tif и samples/sample.npz + sample_meta.json.

    Args:
        layers: словарь имя -> Layer.
        grid: общая сетка.
        out_dir: папка участка в кэше.
        nodata: значение nodata.

    Returns:
        Sample (для дальнейшего использования).
    """
    layers_dir = out_dir / _LAYERS_DIR
    samples_dir = out_dir / _SAMPLES_DIR
    layers_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    for name, layer in layers.items():
        write_raster(layer=layer, path=layers_dir / f"{name}.tif")

    names = _order_channel_names(list(layers.keys()))
    spec = DataSpec(grid=grid, channel_names=tuple(names), nodata=nodata)
    sample = stack_layers(layers=[layers[n] for n in names], spec=spec)

    np.savez_compressed(
        samples_dir / _SAMPLE_NPZ,
        tensor=sample.tensor,
        channel_mask=sample.channel_mask,
        valid_mask=(
            sample.valid_mask if sample.valid_mask is not None else np.ones(grid.shape, dtype=bool)
        ),
    )

    meta = {
        "channel_names": list(sample.channel_names),
        "channel_mask": sample.channel_mask.tolist(),
        "grid": {
            "crs": grid.crs,
            "pixel_size": grid.pixel_size,
            "x_min": grid.x_min,
            "y_min": grid.y_min,
            "width": grid.width,
            "height": grid.height,
        },
        "nodata": nodata,
        "n_channels": sample.n_channels,
        "n_available_channels": sample.n_available_channels,
        "has_valid_mask": sample.valid_mask is not None,
        "valid_fraction": (
            float(sample.valid_mask.mean()) if sample.valid_mask is not None else 1.0
        ),
    }
    with (samples_dir / _SAMPLE_META).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "Сохранено: %d слоёв, sample C=%d (доступно %d), valid=%.1f%%",
        len(layers),
        sample.n_channels,
        sample.n_available_channels,
        100.0 * meta["valid_fraction"],
    )
    return sample


def build_site(
    raw_dir: Path,
    out_dir: Path,
    config: AppConfig,
    max_points: int | None = None,
    output_crs: str | None = None,
) -> SiteResult:
    """Строит слои и Sample для одного участка.

    Универсальный адаптер: лидар + растры -> layers/ + samples/.

    Args:
        raw_dir: папка участка с сырьём.
        out_dir: папка для результата (layers/, samples/).
        config: конфиг приложения.
        max_points: ограничение числа точек лидара (для отладки).
        output_crs: если задан — перепроецировать все слои в эту CRS
            перед сохранением. По умолчанию None (internal CRS).

    Returns:
        SiteResult.

    Raises:
        ValueError: если в папке нет ни лидара, ни растров;
            если не удалось построить ни одного слоя.
    """
    inventory = inventory_site(site_dir=raw_dir, config=config)

    # Целевая сетка: приоритет модальностей из конфига.
    # Лидар обычно в internal CRS, и его плотность выше — он задаёт сетку.
    grid: Grid | None = None

    # 1. Лидар задаёт сетку, если есть.
    if inventory.lidar_path is not None:
        header = read_header(path=inventory.lidar_path)
        # Границы в target_crs: читаем одну точку? Нет — bounds из header
        # в source CRS, перепроецируем.
        from pyproj import Transformer

        src_crs = header.crs or config.crs.source
        transformer = Transformer.from_crs(src_crs, config.crs.internal, always_xy=True)
        x_min = header.bounds_min[0]
        y_min = header.bounds_min[1]
        x_max = header.bounds_max[0]
        y_max = header.bounds_max[1]
        corners_x = [x_min, x_min, x_max, x_max]
        corners_y = [y_min, y_max, y_min, y_max]
        xs_t, ys_t = transformer.transform(corners_x, corners_y)
        bounds = (
            float(min(xs_t)),
            float(min(ys_t)),
            float(max(xs_t)),
            float(max(ys_t)),
        )
        grid = Grid.from_bounds(
            crs=config.crs.internal,
            pixel_size=config.grid.pixel_size,
            bounds=bounds,
        )
        logger.info(
            "Сетка по лидару: %dx%d, pixel=%.2f, CRS=%s",
            grid.width,
            grid.height,
            grid.pixel_size,
            grid.crs,
        )

    # 2. Иначе — по первому растру (с учётом priority).
    if grid is None:
        for modality in config.modalities.priority:
            if modality in inventory.rasters:
                first_path = inventory.rasters[modality][0]
                grid = _grid_from_raster(
                    path=first_path,
                    pixel_size=config.grid.pixel_size,
                    target_crs=config.crs.internal,
                )
                logger.info(
                    "Сетка по растру %s: %dx%d, pixel=%.2f, CRS=%s",
                    first_path.name,
                    grid.width,
                    grid.height,
                    grid.pixel_size,
                    grid.crs,
                )
                break
        if grid is None:
            raise ValueError(
                f"Не удалось построить сетку: нет ни лидара, ни растров "
                f"в приоритетных модальностях {config.modalities.priority}"
            )

    layers: dict[str, Layer] = {}
    n_total = 0
    n_ground = 0

    if inventory.lidar_path is not None and MODALITY_LIDAR in config.modalities.enabled:
        lidar_layers, n_total, n_ground = _build_lidar_layers(
            inventory=inventory,
            config=config,
            grid=grid,
            max_points=max_points,
        )
        layers.update(lidar_layers)

    raster_layers = _build_raster_layers(
        inventory=inventory,
        config=config,
        grid=grid,
    )
    layers.update(raster_layers)

    if not layers:
        raise ValueError(
            f"Не построено ни одного слоя для {raw_dir}. "
            f"Проверь modalities.enabled и содержимое папки."
        )

    # Опциональное перепроецирование перед сохранением.
    if output_crs is not None:
        from expds.features.align import reproject_layers_to_crs

        layers = reproject_layers_to_crs(
            layers=layers,
            target_crs=output_crs,
            target_pixel_size=grid.pixel_size,
        )
        grid = next(iter(layers.values())).grid
        logger.info("Слои перепроецированы в %s", output_crs)

    sample = _save_site(
        layers=layers,
        grid=grid,
        out_dir=out_dir,
        nodata=config.grid.nodata,
    )

    return SiteResult(
        layers=layers,
        sample=sample,
        grid=grid,
        inventory=inventory,
        n_points_total=n_total,
        n_points_ground=n_ground,
    )
