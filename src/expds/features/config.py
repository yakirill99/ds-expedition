"""Загрузка и валидация конфигурации проекта."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Константы — имена ключей конфига.
_KEY_DATASET = "dataset"
_KEY_LIDAR = "lidar"
_KEY_LOGGING = "logging"

_KEY_NAME = "name"
_KEY_RAW_DIR = "raw_dir"
_KEY_CACHE_DIR = "cache_dir"
_KEY_LIDAR_FILE = "lidar_file"

_KEY_SOURCE_Z_UNIT = "source_z_unit"
_KEY_TARGET_Z_UNIT = "target_z_unit"
_KEY_CHUNK_SIZE = "chunk_size"
_KEY_MAX_POINTS = "max_points"

_KEY_LEVEL = "level"
_KEY_FORMAT = "format"
_KEY_DATEFMT = "datefmt"
_KEY_LOG_FILE = "log_file"

_KEY_CRS = "crs"
_KEY_CRS_SOURCE = "source"
_KEY_CRS_INTERNAL = "internal"
_KEY_CRS_OUTPUT = "output"

_KEY_GRID = "grid"
_KEY_GRID_PIXEL_SIZE = "pixel_size"
_KEY_GRID_NODATA = "nodata"

_KEY_GROUND = "ground"
_KEY_GROUND_CELL_SIZE = "cell_size"
_KEY_GROUND_PRESET = "preset"
_KEY_GROUND_WINDOW_SIZES = "window_sizes"
_KEY_GROUND_HEIGHT_THRESHOLD = "height_threshold"
_KEY_GROUND_SLOPE_THRESHOLD = "slope_threshold"
_KEY_GROUND_MAX_ITERATIONS = "max_iterations"
_KEY_GROUND_MIN_CHANGE_RATIO = "min_change_ratio"

_KEY_DTM = "dtm"
_KEY_DTM_STATISTIC = "statistic"
_KEY_DTM_FILL_METHOD = "fill_method"
_KEY_DTM_IDW_K = "idw_k"
_KEY_DTM_IDW_POWER = "idw_power"
_KEY_DTM_MAX_HOLE_PIXELS = "max_hole_pixels"
_KEY_DTM_ARTIFACT_WINDOW = "artifact_window"
_KEY_DTM_ARTIFACT_THRESHOLD = "artifact_threshold"

_KEY_RELIEF = "relief"
_KEY_RELIEF_HILLSHADE_AZIMUTHS = "hillshade_azimuths"
_KEY_RELIEF_HILLSHADE_ALTITUDE = "hillshade_altitude"
_KEY_RELIEF_HILLSHADE_Z_FACTOR = "hillshade_z_factor"
_KEY_RELIEF_SLRM_SIGMAS = "slrm_sigmas"
_KEY_RELIEF_OPENNESS_RADIUS = "openness_radius"
_KEY_RELIEF_OPENNESS_N_DIRECTIONS = "openness_n_directions"
_KEY_RELIEF_CURVATURE_SIGMAS = "curvature_sigmas"
_KEY_RELIEF_TPI_RADII = "tpi_radii"
_KEY_RELIEF_TRI_RADII = "tri_radii"

_KEY_CLOUD_FEATURES = "cloud_features"
_KEY_CLOUD_FEATURES_ENABLED = "enabled"
_KEY_CLOUD_FEATURES_NODATA = "nodata"

_KEY_ALIGN = "align"
_KEY_ALIGN_DEFAULT_METHOD = "default_method"
_KEY_ALIGN_MAX_SHIFT = "max_shift"
_KEY_ALIGN_CONFIDENCE_THRESHOLD = "confidence_threshold"

_KEY_SPECTRAL = "spectral"
_KEY_SPECTRAL_RED_CHANNEL = "red_channel"
_KEY_SPECTRAL_NIR_CHANNEL = "nir_channel"
_KEY_SPECTRAL_GREEN_CHANNEL = "green_channel"
_KEY_SPECTRAL_BLUE_CHANNEL = "blue_channel"
_KEY_SPECTRAL_ENABLED = "enabled"
_KEY_SPECTRAL_NODATA = "nodata"

_KEY_MODALITIES = "modalities"
_KEY_MODALITIES_ENABLED = "enabled"
_KEY_MODALITIES_PRIORITY = "priority"

_KEY_OUTPUT_CRS = "output_crs"
_KEY_OUTPUT_CRS_ENABLED = "enabled"
_KEY_OUTPUT_CRS_EPSG = "epsg"


@dataclass(frozen=True)
class AppConfig:
    """Корневой конфиг приложения."""

    dataset: DatasetConfig
    lidar: LidarConfig
    crs: CrsConfig
    grid: GridConfig
    ground: GroundConfig
    dtm: DtmConfig
    relief: ReliefConfig
    cloud_features: CloudFeaturesConfig
    align: AlignConfig
    spectral: SpectralConfig
    modalities: ModalitiesConfig
    output_crs: OutputCrsConfig
    logging: LoggingConfig


@dataclass(frozen=True)
class OutputCrsConfig:
    """Параметры экспорта в выходной CRS.

    По регламенту хакатона координаты векторов (result.geojson) — всегда
    EPSG:3857. Растровые слои для модели остаются в internal CRS (метры),
    чтобы модель работала в метрических единицах. Экспорт в output_crs
    применяется только на границе векторизации/сабмита.

    enabled=False по умолчанию: перепроецировать все слои в 3857 дорого
    и бессмысленно для модели. Включается флагом --output-crs.
    """

    enabled: bool
    epsg: str


@dataclass(frozen=True)
class ModalitiesConfig:
    """Управление модальностями участка.

    enabled — какие модальности обрабатывать (li, or, sp_or, ae, maggr,
    etgr, terfl). priority — порядок приоритета при конфликте сеток
    (первая в списке задаёт целевую сетку).
    """

    enabled: tuple[str, ...]
    priority: tuple[str, ...]


@dataclass(frozen=True)
class SpectralConfig:
    """Параметры спектральных индексов (NDVI, SR, ARI).

    Индексы каналов — 0-based, как в rasterio (band 1 -> index 0).
    Если nir_channel is None — NDVI/SR не строятся (нет NIR).
    Если red_channel is None — ARI не строится.
    """

    enabled: tuple[str, ...]
    red_channel: int | None
    nir_channel: int | None
    green_channel: int | None
    blue_channel: int | None
    nodata: float


@dataclass(frozen=True)
class AlignConfig:
    """Параметры выравнивания слоёв.

    confidence_threshold — порог NCC: пик выше этого значения считается
    надёжным сдвигом. NCC в диапазоне [-1, 1].
    """

    default_method: str
    max_shift: int
    confidence_threshold: float


@dataclass(frozen=True)
class CloudFeaturesConfig:
    """Параметры растровых признаков из облака точек."""

    enabled: tuple[str, ...]
    nodata: float


@dataclass(frozen=True)
class ReliefConfig:
    """Параметры производных рельефа."""

    hillshade_azimuths: tuple[float, ...]
    hillshade_altitude: float
    hillshade_z_factor: float
    slrm_sigmas: tuple[float, ...]
    openness_radius: int
    openness_n_directions: int
    curvature_sigmas: tuple[float, ...]
    tpi_radii: tuple[int, ...]
    tri_radii: tuple[int, ...]


@dataclass(frozen=True)
class DtmConfig:
    """Параметры построения DTM."""

    statistic: str
    fill_method: str
    idw_k: int
    idw_power: float
    max_hole_pixels: int
    artifact_window: int
    artifact_threshold: float


@dataclass(frozen=True)
class GroundConfig:
    """Параметры фильтра земли."""

    cell_size: float
    preset: str
    window_sizes: tuple[int, ...]
    height_threshold: float
    slope_threshold: float
    max_iterations: int
    min_change_ratio: float


@dataclass(frozen=True)
class LoggingConfig:
    """Параметры логирования."""

    level: str
    format: str
    datefmt: str
    log_file: Path


@dataclass(frozen=True)
class CrsConfig:
    """Целевые CRS проекта.

    source — в каком CRS лежат исходные данные.
    internal — CRS для внутренних расчётов (метры, без искажений).
    output — CRS для выходных GeoJSON (EPSG:3857 по регламенту).
    """

    source: str
    internal: str
    output: str


@dataclass(frozen=True)
class GridConfig:
    """Параметры общей растровой сетки."""

    pixel_size: float
    nodata: float


@dataclass(frozen=True)
class DatasetConfig:
    """Описание датасета: имя, папки, файлы."""

    name: str
    raw_dir: Path
    cache_dir: Path
    lidar_file: Path

    @property
    def lidar_path(self) -> Path:
        """Полный путь к лидарному файлу."""
        return self.raw_dir / self.lidar_file


@dataclass(frozen=True)
class LidarConfig:
    """Параметры обработки лидара."""

    source_z_unit: str
    target_z_unit: str
    chunk_size: int
    max_points: int | None


def _require_mapping(data: Any, key: str) -> dict[str, Any]:
    """Проверяет, что по ключу лежит словарь, и возвращает его."""
    if key not in data:
        raise KeyError(f"В конфиге отсутствует секция '{key}'")
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"Секция '{key}' должна быть словарём, получено {type(value)}")
    return value


def _optional_mapping(data: Any, key: str, default: dict[str, Any]) -> dict[str, Any]:
    """Возвращает словарь по ключу или default, если секции нет.

    Нужно для обратной совместимости: старые конфиги без новых секций
    (spectral, modalities, output_crs) должны загружаться.
    """
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"Секция '{key}' должна быть словарём, получено {type(value)}")
    return value


def _parse_spectral(raw: dict[str, Any]) -> SpectralConfig:
    """Разбирает секцию spectral с дефолтами."""
    return SpectralConfig(
        enabled=tuple(str(f) for f in raw.get(_KEY_SPECTRAL_ENABLED, [])),
        red_channel=raw.get(_KEY_SPECTRAL_RED_CHANNEL),
        nir_channel=raw.get(_KEY_SPECTRAL_NIR_CHANNEL),
        green_channel=raw.get(_KEY_SPECTRAL_GREEN_CHANNEL),
        blue_channel=raw.get(_KEY_SPECTRAL_BLUE_CHANNEL),
        nodata=float(raw.get(_KEY_SPECTRAL_NODATA, -9999.0)),
    )


def _parse_modalities(raw: dict[str, Any]) -> ModalitiesConfig:
    """Разбирает секцию modalities с дефолтами."""
    return ModalitiesConfig(
        enabled=tuple(str(m) for m in raw.get(_KEY_MODALITIES_ENABLED, ["li"])),
        priority=tuple(
            str(m) for m in raw.get(_KEY_MODALITIES_PRIORITY, ["li", "or", "sp_or", "ae"])
        ),
    )


def _parse_output_crs(raw: dict[str, Any]) -> OutputCrsConfig:
    """Разбирает секцию output_crs с дефолтами."""
    return OutputCrsConfig(
        enabled=bool(raw.get(_KEY_OUTPUT_CRS_ENABLED, False)),
        epsg=str(raw.get(_KEY_OUTPUT_CRS_EPSG, "EPSG:3857")),
    )


def load_config(path: Path) -> AppConfig:
    """Читает YAML-конфиг и превращает его в типизированный объект.

    Секции spectral, modalities, output_crs опциональны: если их нет,
    используются дефолты (обратная совместимость со старыми конфигами).

    Args:
        path: путь к YAML-файлу конфигурации.

    Returns:
        AppConfig с заполненными секциями.

    Raises:
        FileNotFoundError: если файл не найден.
        KeyError: если обязательная секция отсутствует.
        TypeError: если структура секции неверна.
    """
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise TypeError("Корень конфига должен быть словарём")

    ds_raw = _require_mapping(raw, _KEY_DATASET)
    lidar_raw = _require_mapping(raw, _KEY_LIDAR)
    crs_raw = _require_mapping(raw, _KEY_CRS)
    grid_raw = _require_mapping(raw, _KEY_GRID)
    ground_raw = _require_mapping(raw, _KEY_GROUND)
    dtm_raw = _require_mapping(raw, _KEY_DTM)
    relief_raw = _require_mapping(raw, _KEY_RELIEF)
    cloud_raw = _require_mapping(raw, _KEY_CLOUD_FEATURES)
    align_raw = _require_mapping(raw, _KEY_ALIGN)
    log_raw = _require_mapping(raw, _KEY_LOGGING)

    spectral_raw = _optional_mapping(raw, _KEY_SPECTRAL, {})
    modalities_raw = _optional_mapping(raw, _KEY_MODALITIES, {})
    output_crs_raw = _optional_mapping(raw, _KEY_OUTPUT_CRS, {})

    dataset = DatasetConfig(
        name=str(ds_raw[_KEY_NAME]),
        raw_dir=Path(ds_raw[_KEY_RAW_DIR]),
        cache_dir=Path(ds_raw[_KEY_CACHE_DIR]),
        lidar_file=Path(ds_raw[_KEY_LIDAR_FILE]),
    )
    lidar = LidarConfig(
        source_z_unit=str(lidar_raw[_KEY_SOURCE_Z_UNIT]),
        target_z_unit=str(lidar_raw[_KEY_TARGET_Z_UNIT]),
        chunk_size=int(lidar_raw[_KEY_CHUNK_SIZE]),
        max_points=lidar_raw[_KEY_MAX_POINTS],
    )
    crs = CrsConfig(
        source=str(crs_raw[_KEY_CRS_SOURCE]),
        internal=str(crs_raw[_KEY_CRS_INTERNAL]),
        output=str(crs_raw[_KEY_CRS_OUTPUT]),
    )
    grid = GridConfig(
        pixel_size=float(grid_raw[_KEY_GRID_PIXEL_SIZE]),
        nodata=float(grid_raw[_KEY_GRID_NODATA]),
    )
    ground = GroundConfig(
        cell_size=float(ground_raw[_KEY_GROUND_CELL_SIZE]),
        preset=str(ground_raw[_KEY_GROUND_PRESET]),
        window_sizes=tuple(int(w) for w in ground_raw[_KEY_GROUND_WINDOW_SIZES]),
        height_threshold=float(ground_raw[_KEY_GROUND_HEIGHT_THRESHOLD]),
        slope_threshold=float(ground_raw[_KEY_GROUND_SLOPE_THRESHOLD]),
        max_iterations=int(ground_raw[_KEY_GROUND_MAX_ITERATIONS]),
        min_change_ratio=float(ground_raw[_KEY_GROUND_MIN_CHANGE_RATIO]),
    )
    dtm = DtmConfig(
        statistic=str(dtm_raw[_KEY_DTM_STATISTIC]),
        fill_method=str(dtm_raw[_KEY_DTM_FILL_METHOD]),
        idw_k=int(dtm_raw[_KEY_DTM_IDW_K]),
        idw_power=float(dtm_raw[_KEY_DTM_IDW_POWER]),
        max_hole_pixels=int(dtm_raw[_KEY_DTM_MAX_HOLE_PIXELS]),
        artifact_window=int(dtm_raw[_KEY_DTM_ARTIFACT_WINDOW]),
        artifact_threshold=float(dtm_raw[_KEY_DTM_ARTIFACT_THRESHOLD]),
    )
    relief = ReliefConfig(
        hillshade_azimuths=tuple(float(a) for a in relief_raw[_KEY_RELIEF_HILLSHADE_AZIMUTHS]),
        hillshade_altitude=float(relief_raw[_KEY_RELIEF_HILLSHADE_ALTITUDE]),
        hillshade_z_factor=float(relief_raw[_KEY_RELIEF_HILLSHADE_Z_FACTOR]),
        slrm_sigmas=tuple(float(s) for s in relief_raw[_KEY_RELIEF_SLRM_SIGMAS]),
        openness_radius=int(relief_raw[_KEY_RELIEF_OPENNESS_RADIUS]),
        openness_n_directions=int(relief_raw[_KEY_RELIEF_OPENNESS_N_DIRECTIONS]),
        curvature_sigmas=tuple(float(s) for s in relief_raw[_KEY_RELIEF_CURVATURE_SIGMAS]),
        tpi_radii=tuple(int(r) for r in relief_raw[_KEY_RELIEF_TPI_RADII]),
        tri_radii=tuple(int(r) for r in relief_raw[_KEY_RELIEF_TRI_RADII]),
    )
    cloud_features = CloudFeaturesConfig(
        enabled=tuple(str(f) for f in cloud_raw[_KEY_CLOUD_FEATURES_ENABLED]),
        nodata=float(cloud_raw[_KEY_CLOUD_FEATURES_NODATA]),
    )
    align = AlignConfig(
        default_method=str(align_raw[_KEY_ALIGN_DEFAULT_METHOD]),
        max_shift=int(align_raw[_KEY_ALIGN_MAX_SHIFT]),
        confidence_threshold=float(align_raw[_KEY_ALIGN_CONFIDENCE_THRESHOLD]),
    )
    logging_cfg = LoggingConfig(
        level=str(log_raw[_KEY_LEVEL]),
        format=str(log_raw[_KEY_FORMAT]),
        datefmt=str(log_raw[_KEY_DATEFMT]),
        log_file=Path(log_raw[_KEY_LOG_FILE]),
    )

    return AppConfig(
        dataset=dataset,
        lidar=lidar,
        crs=crs,
        grid=grid,
        ground=ground,
        dtm=dtm,
        relief=relief,
        cloud_features=cloud_features,
        align=align,
        spectral=_parse_spectral(spectral_raw),
        modalities=_parse_modalities(modalities_raw),
        output_crs=_parse_output_crs(output_crs_raw),
        logging=logging_cfg,
    )


def setup_logging(config: LoggingConfig) -> None:
    """Настраивает корневой логгер по конфигу.

    Args:
        config: секция логирования из AppConfig.
    """
    config.log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(config.log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        datefmt=config.datefmt,
        handlers=handlers,
    )
    logger.debug("Логирование настроено: level=%s, file=%s", config.level, config.log_file)
