"""build_site: сырой участок -> формат костяка (layers/*.tif + samples/).

Функция build_site(raw_dir, out_dir, cfg) — без YAML-путей внутри: все
параметры приходят в AppConfig (expds.features.config.load_config reads
sections dataset/lidar/crs/grid/ground/dtm/relief/cloud_features/align/logging).

Что умеет:
- лидар (LAS/LAZ внутри raw_dir, поиск рекурсивный): полный пайплайн
  бывшего scripts/build_layers.py — чтение -> фильтр земли -> сетка ->
  DTM -> производные рельефа -> признаки облака -> GeoTIFF + Sample;
- уже собранный участок (out_dir/layers/ с GeoTIFF): пропуск (idempotent),
  force=True пересобирает;
- растровый участок без лидара: RawSiteUnsupportedError — семантика каналов
  не согласована с организаторами (docs/issues/03-data-geo.md, статус
  04.10.2026: 8 из 9 участков растровые, из них лидар только в 013).

Вызов из инференса: expds.predict.run -> resolve_sites (автосборка сырых
лидарных участков, структурированная ошибка для растровых).

Кэш слоёв (expds.data.cache) не используется: участок 013 строится за
секунды, повторный вызов отсекается проверкой out_dir/layers.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expds.data.build_raster import RGB_PREFIXES, build_site_rasters
from expds.data.las import read_points
from expds.data.raster import write_raster
from expds.features.cloud_features import build_cloud_features
from expds.features.config import AppConfig
from expds.features.dtm import build_dtm
from expds.features.grid import DataSpec, Grid, Layer, stack_layers
from expds.features.ground import filter_ground
from expds.features.relief import build_relief_layers
from expds.utils.profiler import Profiler

logger = logging.getLogger(__name__)

_LAYERS_DIR = "layers"
_SAMPLES_DIR = "samples"
_LAS_SUFFIXES = {".las", ".laz"}

# Фиксированный порядок каналов для Sample: сначала dtm, потом рельеф,
# потом cloud. Внутри каждой группы — строковая сортировка (CONTRACTS.md §3).
_CHANNEL_ORDER_DTM = "dtm"
_CHANNEL_ORDER_RELIEF_PREFIXES = (
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
_CHANNEL_ORDER_CLOUD_PREFIXES = (
    "point_density",
    "mean_intensity",
    "std_intensity",
    "non_first_return_ratio",
    "mean_return_number",
    "mean_z",
    "z_std",
)


class RawSiteUnsupportedError(RuntimeError):
    """Сырой участок, который build_site не умеет собрать (растровый/пустой)."""


@dataclass(frozen=True)
class BuildSiteResult:
    """Результат сборки участка.

    built=False означает, что layers/ уже были и пересборки не было.
    """

    site_dir: Path
    layer_names: tuple[str, ...]
    n_points_total: int
    n_points_ground: int
    built: bool


def find_lidar_files(raw_dir: Path) -> list[Path]:
    """Все LAS/LAZ внутри raw_dir (рекурсивно), по алфавиту."""
    p = Path(raw_dir)
    if not p.is_dir():
        return []
    return sorted(q for q in p.rglob("*") if q.suffix.lower() in _LAS_SUFFIXES and q.is_file())


def site_kind(path: Path) -> str:
    """Классификация папки участка: built | lidar | raster | empty.

    Смотрит только на прямых детях (не вглубь): папка участков с сырыми
    детьми не должна классифицироваться как лидарная. built  — есть layers/
    хотя бы с одним GeoTIFF; lidar — LAS/LAZ прямо в корне; raster — GeoTIFF
    в корне без лидара; empty — ничего узнаваемого (или путь не папка).
    """
    p = Path(path)
    if not p.is_dir():
        return "empty"
    layers = p / _LAYERS_DIR
    if layers.is_dir() and next(layers.glob("*.tif"), None) is not None:
        return "built"
    if any(q.suffix.lower() in _LAS_SUFFIXES for q in p.glob("*") if q.is_file()):
        return "lidar"
    if next(p.glob("*.tif"), None) is not None or next(p.glob("*.tiff"), None) is not None:
        return "raster"
    return "empty"


def _order_channel_names(names: list[str]) -> list[str]:
    """Упорядочивает каналы: dtm, рельеф, cloud; внутри группы по алфавиту."""
    ordered: list[str] = []

    if _CHANNEL_ORDER_DTM in names:
        ordered.append(_CHANNEL_ORDER_DTM)

    relief = sorted(
        n
        for n in names
        if n != _CHANNEL_ORDER_DTM and any(n.startswith(p) for p in _CHANNEL_ORDER_RELIEF_PREFIXES)
    )
    ordered.extend(relief)

    cloud = sorted(n for n in names if any(n.startswith(p) for p in _CHANNEL_ORDER_CLOUD_PREFIXES))
    ordered.extend(cloud)

    remaining = sorted(set(names) - set(ordered))
    ordered.extend(remaining)

    return ordered


def _save_sample(
    layers: dict[str, Layer],
    grid: Grid,
    output_dir: Path,
    nodata: float,
) -> None:
    """Собирает Sample из слоёв и сохраняет как .npz + .json (канон, с channel_mask)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    names = _order_channel_names(list(layers.keys()))
    spec = DataSpec(
        grid=grid,
        channel_names=tuple(names),
        nodata=nodata,
    )
    sample = stack_layers(layers=[layers[n] for n in names], spec=spec)

    npz_path = output_dir / "sample.npz"
    np.savez_compressed(
        npz_path,
        tensor=sample.tensor,
        channel_mask=sample.channel_mask,
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
    }
    meta_path = output_dir / "sample_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2))

    logger.info(
        "Sample сохранён: %s | C=%d (доступно %d), H=%d, W=%d",
        npz_path,
        sample.n_channels,
        sample.n_available_channels,
        grid.height,
        grid.width,
    )


def _read_all_points(
    las_files: Sequence[Path],
    chunk_size: int,
    source_crs: str,
    target_crs: str,
    source_z_unit: str,
    max_points: int | None,
    profiler: Profiler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Читает все LAS в один массив. Returns: (x, y, z, intensity, return_number)."""
    chunks: list[np.ndarray] = []
    with profiler.stage("чтение LAS"):
        for las_path in las_files:
            logger.info("Источник: %s", las_path)
            for block in read_points(
                path=las_path,
                chunk_size=chunk_size,
                source_crs=source_crs,
                target_crs=target_crs,
                source_z_unit=source_z_unit,
                max_points=max_points,
            ):
                chunks.append(block)

    if not chunks:
        raise RawSiteUnsupportedError(f"LAS не дал ни одной точки: {[str(p) for p in las_files]}")

    all_points = np.vstack(chunks)
    if max_points is not None and len(all_points) > max_points:
        all_points = all_points[:max_points]

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


def _unsupported_message(raw_dir: Path, kind: str) -> str:
    """Человекочитаемое объяснение, почему участок не собирается."""
    if kind == "raster":
        return (
            f"{Path(raw_dir).name}: нечего собрать — нет ни LAS/LAZ, ни 03_/04_/05_ "
            "(только геофизика 07_..09_ или пустая папка). Геофизика ждёт D4.4."
        )
    return (
        f"{raw_dir}: нет ни собранных слоёв (layers/ с GeoTIFF), ни LAS/LAZ, ни растров — "
        "нечего собирать."
    )


def build_site(
    raw_dir: Path,
    out_dir: Path,
    cfg: AppConfig,
    profiler: Profiler | None = None,
    limit_points: int | None = None,
    force: bool = False,
) -> BuildSiteResult:
    """Сырой участок -> out_dir/layers/*.tif + out_dir/samples/.

    Args:
        raw_dir: папка с сырьём (LAS/LAZ ищутся рекурсивно).
        out_dir: куда писать layers/ и samples/ (для инференса out_dir == site_dir).
        cfg: AppConfig роли 1 (build_site сам YAML не читает).
        profiler: необязательный Profiler (стадии совпадают с build_layers).
        limit_points: ограничить число точек (отладка).
        force: пересобрать, даже если out_dir/layers/ уже не пуст.

    Returns:
        BuildSiteResult (built=False — layers/ уже были, пропуск).

    Raises:
        RawSiteUnsupportedError: растровый/пустой участок или LAS без точек.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    prof = profiler if profiler is not None else Profiler()

    layers_dir = out_dir / _LAYERS_DIR
    layers_ready = layers_dir.is_dir() and next(layers_dir.glob("*.tif"), None) is not None
    if not force and layers_ready:
        names = tuple(sorted(p.stem for p in layers_dir.glob("*.tif")))
        logger.info("%s: layers/ уже есть (%d слоёв), пропуск (force=False)", out_dir, len(names))
        return BuildSiteResult(out_dir, names, 0, 0, built=False)

    las_files = find_lidar_files(raw_dir)
    if not las_files:
        if layers_ready:
            raise RawSiteUnsupportedError(
                f"{out_dir}: layers/ уже есть, но сырьё LAS/LAZ в {raw_dir} не найдено — "
                "пересборка (force) невозможна"
            )
        raise RawSiteUnsupportedError(_unsupported_message(raw_dir, site_kind(raw_dir)))

    # 1. Точки.
    x, y, z, intensity, return_number = _read_all_points(
        las_files=las_files,
        chunk_size=cfg.lidar.chunk_size,
        source_crs=cfg.crs.source,
        target_crs=cfg.crs.internal,
        source_z_unit=cfg.lidar.source_z_unit,
        max_points=limit_points,
        profiler=prof,
    )
    n_total = len(x)

    # 2. Фильтр земли.
    with prof.stage("фильтр земли"):
        ground_result = filter_ground(x=x, y=y, z=z, config=cfg.ground, crs=cfg.crs.internal)
        ground_mask = ground_result.ground_mask
        n_ground = int(ground_mask.sum())
        logger.info("Земля: %d из %d (%.1f%%)", n_ground, n_total, 100.0 * n_ground / n_total)

    # 3. Целевая сетка по границам точек.
    with prof.stage("построение целевой сетки"):
        bounds = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
        grid = Grid.from_bounds(crs=cfg.crs.internal, pixel_size=cfg.grid.pixel_size, bounds=bounds)

    # 4. DTM из точек земли.
    x_g, y_g, z_g = x[ground_mask], y[ground_mask], z[ground_mask]
    with prof.stage("DTM"):
        dtm_result = build_dtm(
            x=x_g,
            y=y_g,
            z=z_g,
            grid=grid,
            config=cfg.dtm,
            source=str(las_files[0]),
        )
        dtm_layer = dtm_result.dtm

    # 5. Производные рельефа.
    with prof.stage("производные рельефа"):
        relief_arrays = build_relief_layers(
            dtm=dtm_layer.data,
            pixel_size=grid.pixel_size,
            hillshade_azimuths=cfg.relief.hillshade_azimuths,
            hillshade_altitude=cfg.relief.hillshade_altitude,
            hillshade_z_factor=cfg.relief.hillshade_z_factor,
            slrm_sigmas=cfg.relief.slrm_sigmas,
            openness_radius=cfg.relief.openness_radius,
            openness_n_directions=cfg.relief.openness_n_directions,
            curvature_sigmas=cfg.relief.curvature_sigmas,
            tpi_radii=cfg.relief.tpi_radii,
            tri_radii=cfg.relief.tri_radii,
        )
        relief_layers = {
            name: Layer(
                name=name,
                data=arr,
                grid=grid,
                nodata=cfg.grid.nodata,
                source=str(las_files[0]),
                license="unknown",
                unit="unitless",
            )
            for name, arr in relief_arrays.items()
        }

    # 6. Признаки из облака (по всем точкам, не только земле).
    with prof.stage("признаки из облака"):
        cloud_result = build_cloud_features(
            x=x,
            y=y,
            z=z,
            intensity=intensity,
            return_number=return_number,
            grid=grid,
            enabled=cfg.cloud_features.enabled,
            nodata=cfg.cloud_features.nodata,
        )

    # 7. Сборка и запись.
    layers: dict[str, Layer] = {"dtm": dtm_layer}
    layers.update(relief_layers)
    layers.update(cloud_result.layers)

    with prof.stage("сохранение слоёв"):
        layers_dir.mkdir(parents=True, exist_ok=True)
        for name, layer in layers.items():
            write_raster(layer=layer, path=layers_dir / f"{name}.tif")

    with prof.stage("сохранение Sample"):
        _save_sample(
            layers=layers,
            grid=layers["dtm"].grid,
            output_dir=out_dir / _SAMPLES_DIR,
            nodata=cfg.grid.nodata,
        )

    logger.info(
        "Итог: %d слоёв, точек всего %d, земли %d -> %s",
        len(layers),
        n_total,
        n_ground,
        layers_dir,
    )
    return BuildSiteResult(
        site_dir=out_dir,
        layer_names=tuple(sorted(layers)),
        n_points_total=n_total,
        n_points_ground=n_ground,
        built=True,
    )


_MODALITY_RE = re.compile(r"^\d{2}_")


def _has_rgb_sources(site: Path) -> bool:
    """Есть ли подпапки 03_/04_/05_ (аэрофото, спутник, ортофото)."""
    return any(p.is_dir() and p.name[:3] in RGB_PREFIXES for p in site.iterdir())


def _has_modality_dirs(site: Path) -> bool:
    """Есть ли подпапки-модальности организаторов (01_ .. 09_).

    Участок датасета — NNN_Имя, источник внутри него — NN_Имя (две цифры).
    По этому признаку регион платформы (снимок в подпапке 04_) отличается
    от папки, в которой лежат участки.
    """
    return any(p.is_dir() and _MODALITY_RE.match(p.name) for p in site.iterdir())


def _has_layers(site: Path) -> bool:
    layers = site / _LAYERS_DIR
    return layers.is_dir() and next(layers.glob("*.tif"), None) is not None


def _build_rgb(site: Path, labels_crs: str, max_pixels: int, prof: Profiler) -> None:
    """rgb_* из 03_/04_/05_, если они есть. Уже собранные не трогает."""
    if not _has_rgb_sources(site):
        return
    with prof.stage(f"{site.name}: сборка rgb_* (build_site_rasters)"):
        res = build_site_rasters(site, labels_crs, max_pixels=max_pixels, log=logger.info)
    if not (site / _LAYERS_DIR / "rgb_r.tif").is_file():
        logger.warning("%s: rgb_* не собраны: %s", site.name, res.get("reason"))


def resolve_sites(
    data: Path,
    cfg: AppConfig | None,
    profiler: Profiler | None = None,
    labels_crs: str = "EPSG:3857",
    max_pixels: int = 12_000_000,
) -> list[Path]:
    """Готовые участки + автосборка сырых (лидарных и растровых).

    Правила:
    - data участок с layers/ -> [data];
    - data участок с сырьём (LAS в корне или подпапки 01_..09_) -> сборка
      in-place: лидар через build_site, rgb_* через build_site_rasters;
    - data папка участков -> обход, каждый участок по тем же правилам;
    - участок, который собрать нельзя (только геофизика 07_..09_, снимок
      больше max_pixels, покрытие ниже порога), -> warning и пропуск.
      Возвращается [] — вызывающий пишет пустой FeatureCollection.
      На платформе неуспешный запуск стоит одной из трёх попыток в сутки;
    - cfg=None разрешает только готовые и растровые участки.

    Пишет in-place, как и раньше build_site: каталог данных должен быть
    доступен на запись.
    """
    data = Path(data)
    prof = profiler if profiler is not None else Profiler()

    if site_kind(data) == "built":
        # У регионов La+SpOr / La+Ae рельеф собран, а rgb_* может не быть.
        _build_rgb(data, labels_crs, max_pixels, prof)
        return [data]

    if not data.is_dir():
        raise FileNotFoundError(f"{data}: не папка участка")

    # Признак «это участок»: своё сырьё (LAS в корне либо подпапки-модальности)
    # или отсутствие детей-участков. Иначе — папка участков.
    own_sources = site_kind(data) == "lidar" or _has_modality_dirs(data)
    child_sites = [
        p
        for p in data.iterdir()
        if p.is_dir() and (site_kind(p) in {"built", "lidar", "raster"} or _has_modality_dirs(p))
    ]
    if own_sources or not child_sites:
        return _resolve_one(data, cfg, prof, labels_crs, max_pixels)

    ready: list[Path] = []
    skipped: list[Path] = []
    need_cfg: list[Path] = []
    for entry in sorted(child_sites):
        try:
            got = _resolve_one(entry, cfg, prof, labels_crs, max_pixels)
        except RawSiteUnsupportedError:
            if cfg is None:
                need_cfg.append(entry)
                continue
            raise
        (ready if got else skipped).append(entry)

    if skipped:
        logger.warning(
            "Пропущены участки, которые не собрать: %s", ", ".join(p.name for p in skipped)
        )
    if need_cfg:
        raise RawSiteUnsupportedError(
            "Сырые лидарные участки без конфига роли 1 (нужны секции dataset/lidar/ground/"
            "dtm/relief/cloud_features): " + ", ".join(str(p) for p in need_cfg)
        )
    if not ready and not skipped:
        raise FileNotFoundError(f"{data}: нет ни одного участка")
    return ready


def _resolve_one(
    site: Path,
    cfg: AppConfig | None,
    prof: Profiler,
    labels_crs: str,
    max_pixels: int,
) -> list[Path]:
    """Собирает один участок in-place. [] — собрать нечего."""
    if find_lidar_files(site):
        if cfg is None:
            raise RawSiteUnsupportedError(
                f"{site}: сырой лидарный участок, но конфиг роли 1 недоступен — "
                "автосборка слоёв выключена"
            )
        with prof.stage(f"{site.name}: сборка слоёв (build_site)"):
            build_site(site, site, cfg, prof)

    _build_rgb(site, labels_crs, max_pixels, prof)

    if _has_layers(site):
        return [site]
    logger.warning("%s: %s — пропуск", site.name, _unsupported_message(site, site_kind(site)))
    return []
