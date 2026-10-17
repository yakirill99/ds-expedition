"""CLI: построение всех растровых слоёв из LAS.

Тонкая обёртка над expds.features.build_site.build_site.
Все параметры — из YAML-конфига.

Функция _save_sample сохранена с прежней сигнатурой: её использует
tools/make_synthetic.py (роль 5) и тесты синтетики. Удалять её нельзя —
это публичный контракт скрипта, даже если имя начинается с подчёркивания.

Пример:
    uv run python scripts/build_layers.py --config configs/dataset_013.yaml
    uv run python scripts/build_layers.py --config configs/dataset_013.yaml \
        --output-crs EPSG:3857 --limit-points 500000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from expds.features.build_site import build_site
from expds.features.config import load_config, setup_logging
from expds.features.grid import DataSpec, Grid, Layer, stack_layers
from expds.utils.profiler import Profiler

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("configs/dataset_raleigh.yaml")

# Константы: имена служебных файлов Sample.
_SAMPLE_NPZ = "sample.npz"
_SAMPLE_META = "sample_meta.json"


def _save_sample(
    layers: dict[str, Layer],
    grid: Grid,
    output_dir: Path,
    nodata: float,
) -> None:
    """Собирает Sample из слоёв и сохраняет как .npz + .json.

    Публичный контракт: функцию использует tools/make_synthetic.py
    (роль 5) и тесты синтетики. Сигнатура не меняется.

    Порядок каналов: dtm первый, hole_mask вторым (если есть), остальные
    по алфавиту. valid_mask сохраняется, если stack_layers её вернул.

    Args:
        layers: словарь имя -> Layer.
        grid: общая сетка.
        output_dir: папка для сохранения.
        nodata: значение nodata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Порядок каналов — как в старом контракте: dtm первым, остальные
    # по алфавиту. hole_mask НЕ выделяем: роль 5 и роль 2 рассчитывают
    # на этот порядок в чекпойнтах и тестах.
    names = sorted(layers.keys())
    if "dtm" in names:
        names.remove("dtm")
        names = ["dtm"] + names

    spec = DataSpec(grid=grid, channel_names=tuple(names), nodata=nodata)
    sample = stack_layers(layers=[layers[n] for n in names], spec=spec)

    npz_path = output_dir / _SAMPLE_NPZ
    np.savez_compressed(
        npz_path,
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
    }
    meta_path = output_dir / _SAMPLE_META
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
        help="Отключить кэш (не используется, оставлено для совместимости)",
    )
    parser.add_argument(
        "--limit-points",
        type=int,
        default=None,
        help="Ограничить число точек для быстрого прогона",
    )
    parser.add_argument(
        "--output-crs",
        type=str,
        default=None,
        help="Перепроецировать слои в указанную CRS перед сохранением "
        "(например, EPSG:3857). По умолчанию — internal CRS.",
    )
    return parser.parse_args()


def main() -> int:
    """Точка входа CLI."""
    args = _parse_args()
    config = load_config(path=args.config)
    setup_logging(config=config.logging)

    profiler = Profiler()
    raw_dir = config.dataset.raw_dir
    out_dir = config.dataset.cache_dir

    try:
        with profiler.stage("build_site"):
            result = build_site(
                raw_dir=raw_dir,
                out_dir=out_dir,
                config=config,
                max_points=args.limit_points,
                output_crs=args.output_crs,
            )
    except Exception as exc:
        logger.error("Пайплайн упал: %s", exc, exc_info=True)
        profiler.report()
        return 1

    valid_fraction = (
        float(result.sample.valid_mask.mean()) if result.sample.valid_mask is not None else 1.0
    )
    logger.info(
        "Итог: %d слоёв, точек всего %d, земли %d, valid=%.1f%%",
        len(result.layers),
        result.n_points_total,
        result.n_points_ground,
        100.0 * valid_fraction,
    )
    profiler.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
