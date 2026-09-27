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
    logging: LoggingConfig


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
    output — CRS для выходных GeoJSON.
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


def load_config(path: Path) -> AppConfig:
    """Читает YAML-конфиг и превращает его в типизированный объект.

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
