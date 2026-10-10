"""CLI: построение всех растровых слоёв из LAS.

Полный пайплайн:
1. Читает LAS чанками, склеивает в массивы.
2. Фильтр земли: отделяет землю от растительности и зданий.
3. DTM из точек земли.
4. Производные рельефа (slope, hillshade, SLRM, openness, TPI, TRI, ...).
5. Растровые признаки из облака (плотность, интенсивность, ...).
6. Сохраняет все слои через LayerCache.

Все параметры — из YAML-конфига.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expds.data.las import read_points
from expds.data.raster import write_raster
from expds.features.cloud_features import build_cloud_features
from expds.features.config import AppConfig, load_config, setup_logging
from expds.features.dtm import build_dtm
from expds.features.grid import DataSpec, Grid, Layer, stack_layers
from expds.features.ground import filter_ground
from expds.features.relief import build_relief_layers
from expds.utils.profiler import Profiler

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("configs/dataset_raleigh.yaml")

# Фиксированный порядок каналов для Sample: сначала dtm, потом рельеф,
# потом cloud features. Внутри каждой группы — по алфавиту.
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


def _order_channel_names(names: list[str]) -> list[str]:
    """Упорядочивает каналы: dtm, рельеф, cloud.

    Внутри каждой группы — по алфавиту. Это фиксированный порядок,
    на который роль 2 может рассчитывать.

    Args:
        names: список имён слоёв.

    Returns:
        Упорядоченный список.
    """
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

    # Что осталось — в конец, по алфавиту.
    remaining = sorted(set(names) - set(ordered))
    ordered.extend(remaining)

    return ordered


def _save_sample(
    layers: dict[str, Layer],
    grid: Grid,
    output_dir: Path,
    nodata: float,
) -> None:
    """Собирает Sample из слоёв и сохраняет как .npz + .json.

    Порядок каналов фиксирован: dtm, рельеф, cloud.
    target=None — разметка появится позже, роль 2 соединит сама.

    Args:
        layers: словарь имя -> Layer.
        grid: общая сетка.
        output_dir: папка для сохранения.
        nodata: значение nodata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    names = _order_channel_names(list(layers.keys()))
    spec = DataSpec(
        grid=grid,
        channel_names=tuple(names),
        nodata=nodata,
    )
    sample = stack_layers(layers=[layers[n] for n in names], spec=spec)

    # Сохраняем тензор и маску каналов.
    npz_path = output_dir / "sample.npz"
    np.savez_compressed(
        npz_path,
        tensor=sample.tensor,
        channel_mask=sample.channel_mask,
    )

    # Сохраняем метаданные.
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
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "Sample сохранён: %s | C=%d (доступно %d), H=%d, W=%d",
        npz_path,
        sample.n_channels,
        sample.n_available_channels,
        grid.height,
        grid.width,
    )


@dataclass(frozen=True)
class PipelineResult:
    """Результат прогона пайплайна.

    layers — словарь имя -> Layer.
    n_points_total — сколько точек прочитано.
    n_points_ground — сколько точек осталось после фильтра земли.
    """

    layers: dict[str, Layer]
    n_points_total: int
    n_points_ground: int


def _parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Построение всех растровых слоёв из LAS")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Путь к YAML-конфигу датасета",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Отключить кэш, всё пересчитать",
    )
    parser.add_argument(
        "--limit-points",
        type=int,
        default=None,
        help="Ограничить число точек для быстрого прогона",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Список слоёв через запятую (для отладки). Если не указан — строятся все.",
    )
    return parser.parse_args()


def _save_sample(
    layers: dict[str, Layer],
    grid: Grid,
    output_dir: Path,
    nodata: float,
) -> None:
    """Собирает Sample из слоёв и сохраняет как .npz + .json.

    Args:
        layers: словарь имя -> Layer.
        grid: общая сетка.
        output_dir: папка для сохранения.
        nodata: значение nodata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Порядок каналов: dtm первый, потом остальные по алфавиту.
    names = sorted(layers.keys())
    if "dtm" in names:
        names.remove("dtm")
        names = ["dtm"] + names

    spec = DataSpec(
        grid=grid,
        channel_names=tuple(names),
        nodata=nodata,
    )
    sample = stack_layers(layers=[layers[n] for n in names], spec=spec)

    # Сохраняем тензор.
    npz_path = output_dir / "sample.npz"
    np.savez_compressed(
        npz_path,
        tensor=sample.tensor,
        channel_mask=sample.channel_mask,
    )

    # Сохраняем метаданные.
    meta = {
        "channel_names": list(sample.channel_names),
        "grid": {
            "crs": grid.crs,
            "pixel_size": grid.pixel_size,
            "x_min": grid.x_min,
            "y_min": grid.y_min,
            "width": grid.width,
            "height": grid.height,
        },
        "nodata": nodata,
    }
    meta_path = output_dir / "sample_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "Sample сохранён: %s | C=%d, H=%d, W=%d",
        npz_path,
        sample.n_channels,
        grid.height,
        grid.width,
    )


def _read_all_points(
    path: Path,
    chunk_size: int,
    source_crs: str,
    target_crs: str,
    source_z_unit: str,
    max_points: int | None,
    profiler: Profiler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Читает все точки LAS в один массив.

    Args:
        path: путь к LAS.
        chunk_size: размер чанка.
        source_crs: CRS исходных данных.
        target_crs: CRS для внутренних расчётов.
        source_z_unit: единица высоты.
        max_points: ограничение числа точек.
        profiler: профилировщик.

    Returns:
        Кортеж (x, y, z, intensity, return_number).
    """
    chunks = []
    with profiler.stage("чтение LAS"):
        for block in read_points(
            path=path,
            chunk_size=chunk_size,
            source_crs=source_crs,
            target_crs=target_crs,
            source_z_unit=source_z_unit,
            max_points=max_points,
        ):
            chunks.append(block)

    if not chunks:
        raise ValueError("LAS не дал ни одного чанка")

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


def _build_target_grid(
    x: np.ndarray,
    y: np.ndarray,
    config: AppConfig,
) -> Grid:
    """Строит целевую сетку по границам точек.

    Args:
        x, y: координаты точек.
        config: конфиг приложения.

    Returns:
        Grid.
    """
    from expds.features.grid import Grid as GridClass

    bounds = (
        float(x.min()),
        float(y.min()),
        float(x.max()),
        float(y.max()),
    )
    return GridClass.from_bounds(
        crs=config.crs.internal,
        pixel_size=config.grid.pixel_size,
        bounds=bounds,
    )


def _run_pipeline(
    config: AppConfig,
    use_cache: bool,
    limit_points: int | None,
    profiler: Profiler,
) -> PipelineResult:
    """Запускает полный пайплайн.

    Args:
        config: конфиг приложения.
        use_cache: использовать кэш.
        limit_points: ограничение числа точек.
        profiler: профилировщик.

    Returns:
        PipelineResult.
    """
    las_path = config.dataset.lidar_path
    logger.info("Источник: %s", las_path)

    # 1. Читаем точки.
    x, y, z, intensity, return_number = _read_all_points(
        path=las_path,
        chunk_size=config.lidar.chunk_size,
        source_crs=config.crs.source,
        target_crs=config.crs.internal,
        source_z_unit=config.lidar.source_z_unit,
        max_points=limit_points,
        profiler=profiler,
    )
    n_total = len(x)

    # 2. Фильтр земли.
    with profiler.stage("фильтр земли"):
        ground_result = filter_ground(
            x=x,
            y=y,
            z=z,
            config=config.ground,
            crs=config.crs.internal,
        )
        ground_mask = ground_result.ground_mask
        n_ground = int(ground_mask.sum())
        logger.info(
            "Земля: %d из %d (%.1f%%)",
            n_ground,
            n_total,
            100.0 * n_ground / n_total,
        )

    # 3. Целевая сетка.
    with profiler.stage("построение целевой сетки"):
        grid = _build_target_grid(x=x, y=y, config=config)

    # 4. DTM.
    x_g = x[ground_mask]
    y_g = y[ground_mask]
    z_g = z[ground_mask]

    with profiler.stage("DTM"):
        dtm_result = build_dtm(
            x=x_g,
            y=y_g,
            z=z_g,
            grid=grid,
            config=config.dtm,
            source=str(las_path),
        )
        dtm_layer = dtm_result.dtm

    # 5. Производные рельефа.
    with profiler.stage("производные рельефа"):
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
        relief_layers = {
            name: Layer(
                name=name,
                data=arr,
                grid=grid,
                nodata=config.grid.nodata,
                source=str(las_path),
                license="unknown",
                unit="unitless",
            )
            for name, arr in relief_arrays.items()
        }

    # 6. Признаки из облака (по всем точкам, не только земле).
    with profiler.stage("признаки из облака"):
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

    # 7. Собираем всё.
    layers: dict[str, Layer] = {"dtm": dtm_layer}
    layers.update(relief_layers)
    layers.update(cloud_result.layers)

    logger.info("Всего слоёв: %d", len(layers))
    return PipelineResult(
        layers=layers,
        n_points_total=n_total,
        n_points_ground=n_ground,
    )


def main() -> int:
    """Точка входа CLI."""
    args = _parse_args()
    config = load_config(path=args.config)
    setup_logging(config=config.logging)

    profiler = Profiler()

    try:
        result = _run_pipeline(
            config=config,
            use_cache=not args.no_cache,
            limit_points=args.limit_points,
            profiler=profiler,
        )
    except Exception as exc:
        logger.error("Пайплайн упал: %s", exc, exc_info=True)
        profiler.report()
        return 1

    # Сохраняем все слои отдельными GeoTIFF (для отладки, QGIS).
    output_dir = config.dataset.cache_dir / "layers"
    with profiler.stage("сохранение слоёв"):
        for name, layer in result.layers.items():
            layer_path = output_dir / f"{name}.tif"
            write_raster(layer=layer, path=layer_path)

    # Сохраняем Sample (для модели).
    sample_dir = config.dataset.cache_dir / "samples"
    with profiler.stage("сохранение Sample"):
        _save_sample(
            layers=result.layers,
            grid=result.layers["dtm"].grid,
            output_dir=sample_dir,
            nodata=config.grid.nodata,
        )

    logger.info(
        "Итог: %d слоёв, точек всего %d, земли %d",
        len(result.layers),
        result.n_points_total,
        result.n_points_ground,
    )
    profiler.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
