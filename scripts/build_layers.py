"""CLI: построение всех растровых слоёв из LAS или из растрового DTM.

Два режима (детект по YAML автоматически):

1. Роль-1 конфиг (секции dataset/lidar/crs/grid/ground/dtm/relief/...):

    uv run python scripts/build_layers.py --config configs/dataset_raleigh.yaml

   Тонкая обёртка над expds.data.build_site.build_site — весь пайплайн
   (чтение LAS -> фильтр земли -> сетка -> DTM -> производные рельефа ->
   признаки облака -> GeoTIFF + Sample) живёт в библиотеке, откуда его же
   вызывает инференс (expds.predict.run -> resolve_sites) для сырых участков.
   Слои пишутся в dataset.cache_dir/layers, Sample — в dataset.cache_dir/samples.

2. Pipeline-конфиг костяка (data/targets/tiles/... + sites, например
   configs/dataset_ahn.yaml): обходит участки из data.sites_dir и собирает
   каждый растровый DTM-участок (dtm*.tif) в него же:

    uv run python scripts/build_layers.py --config configs/dataset_ahn.yaml

   LAS-участки в этом режиме не обрабатываются (им нужны секции роли 1) —
   для них используйте режим 1.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from expds.data.build_site import (
    RawSiteUnsupportedError,
    build_site,
    build_site_from_dtm,
    find_dtm_raster,
    site_kind,
)
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


def _is_pipeline_config(path: Path) -> bool:
    """True — YAML костяка (data/targets/tiles + sites), без секции dataset роли 1."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return "sites" in raw and "dataset" not in raw


def _run_pipeline_mode(config_path: Path, force: bool, profiler: Profiler) -> int:
    """Обход data.sites_dir: растровые DTM-участки -> build_site_from_dtm."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sites_dir = Path((raw.get("data") or {}).get("sites_dir", "data/ahn/raw"))
    if not sites_dir.is_dir():
        logger.error(
            "sites_dir не найден: %s — сначала скачайте данные (scripts/fetch_ahn.py)", sites_dir
        )
        return 2

    n_built = n_ready = n_skipped = 0
    for site in sorted(p for p in sites_dir.iterdir() if p.is_dir()):
        kind = site_kind(site)
        if kind == "built" and not force:
            logger.info("%s: layers/ уже есть, пропуск (force=False)", site.name)
            n_ready += 1
            continue
        if find_dtm_raster(site) is not None:
            build_site_from_dtm(site, site, profiler=profiler, force=force)
            n_built += 1
        elif kind == "lidar":
            logger.error(
                "%s: LAS-участок в pipeline-режиме не собирается — "
                "нужен конфиг роли 1 (режим --config <dataset_lidar>.yaml)",
                site.name,
            )
            return 2
        else:
            logger.warning("Пропущен %s: нет dtm*.tif и нет слоёв", site.name)
            n_skipped += 1

    logger.info(
        "Итог: собрано %d, готовых %d, пропущено %d (профиль ниже)",
        n_built,
        n_ready,
        n_skipped,
    )
    return 0


def main() -> int:
    """Точка входа CLI."""
    args = _parse_args()

    profiler = Profiler()
    if _is_pipeline_config(args.config):
        # в pipeline-режиме нет роль-1 секции logging — настраиваем базово
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        if args.limit_points is not None:
            logger.warning(
                "--limit-points применяется только в режиме роль-1 конфига — игнорируется"
            )
        try:
            code = _run_pipeline_mode(args.config, args.force, profiler)
        except RawSiteUnsupportedError as exc:
            logger.error("%s", exc)
            profiler.report()
            return 2
        except Exception as exc:  # noqa: BLE001 — CLI-граница, печатаем и падаем с 1
            logger.error("Пайплайн упал: %s", exc, exc_info=True)
            profiler.report()
            return 1
        profiler.report()
        return code

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
