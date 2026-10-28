"""CLI: построение всех растровых слоёв из LAS.

Тонкая обёртка над expds.data.build_site.build_site — весь пайплайн
(чтение LAS -> фильтр земли -> сетка -> DTM -> производные рельефа ->
признаки облака -> GeoTIFF + Sample) живёт в библиотеке, откуда его же
вызывает инференс (expds.predict.run -> resolve_sites) для сырых участков.

Все параметры — из YAML-конфига роли 1 (секции dataset/lidar/crs/grid/
ground/dtm/relief/cloud_features/logging). Слои пишутся в
dataset.cache_dir/layers, Sample — в dataset.cache_dir/samples.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from expds.data.build_site import RawSiteUnsupportedError, build_site
from expds.features.config import load_config, setup_logging
from expds.utils.profiler import Profiler

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("configs/dataset_raleigh.yaml")


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
        "--limit-points",
        type=int,
        default=None,
        help="Ограничить число точек для быстрого прогона",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересобрать слои, даже если cache_dir/layers уже не пуст",
    )
    return parser.parse_args()


def main() -> int:
    """Точка входа CLI."""
    args = _parse_args()
    cfg = load_config(path=args.config)
    setup_logging(config=cfg.logging)

    profiler = Profiler()
    try:
        result = build_site(
            raw_dir=cfg.dataset.raw_dir,
            out_dir=cfg.dataset.cache_dir,
            cfg=cfg,
            profiler=profiler,
            limit_points=args.limit_points,
            force=args.force,
        )
    except RawSiteUnsupportedError as exc:
        logger.error("%s", exc)
        profiler.report()
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI-граница, печатаем и падаем с 1
        logger.error("Пайплайн упал: %s", exc, exc_info=True)
        profiler.report()
        return 1

    if not result.built:
        logger.info("Слои уже были, пересборки не было (force=False)")
    profiler.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
