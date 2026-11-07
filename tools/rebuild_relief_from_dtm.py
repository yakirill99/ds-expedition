"""Пересборка dtm.tif и производных рельефа из существующих слоёв, без LAS.

Зачем: заполнение узких пропусков должно происходить ДО расчёта производных.
Иначе slope/hillshade/TPI считаются по поверхности с -9999 и в пропусках
получают мусор, а после правки relief.py — 0 (= медиана после нормировки),
при valid=True. Лосс тогда считает эти пиксели настоящими данными.

Что делает для каждого участка:
  1. читает layers/dtm.tif и запоминает, где были реальные измерения;
  2. заполняет узкие пропуски методом IDW по расстоянию до валидных данных;
  3. пересчитывает все производные рельефа от заполненного DTM;
  4. пишет dtm.tif + производные в --out (или на месте с --in-place);
  5. печатает покрытие до/после.

Облачные признаки (point_density, mean_intensity, ...) НЕ трогает: они
считаются по точкам, а не по DTM, и их nodata = «точек не было» — это
честная информация.

Запуск:
    python3 tools/rebuild_relief_from_dtm.py --config configs/dataset_real.yaml \
        --site 010_Эсбю_укрепление --out data/rebuilt

    python3 tools/rebuild_relief_from_dtm.py --config configs/dataset_real.yaml \
        --site 010_Эсбю_укрепление --in-place
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from expds.data.raster import read_raster, write_raster
from expds.features.dtm import fill_narrow_gaps
from expds.features.grid import Grid, Layer
from expds.features.relief import build_relief_layers
from expds.pipeline_config import load_pipeline_config

logger = logging.getLogger("rebuild")

_META = Path("samples") / "sample_meta.json"
_LAYERS = "layers"

# Имена производных рельефа, которые пересчитываем. Облачные признаки
# (point_density, mean_intensity, mean_z, ...) сюда не входят намеренно.
_RELIEF_PREFIXES = (
    "slope",
    "aspect_sin",
    "aspect_cos",
    "hillshade_az_",
    "slrm_sigma_",
    "positive_openness",
    "negative_openness",
    "sky_view_factor",
    "curvature_sigma_",
    "tpi_radius_",
    "tri_radius_",
)


def _is_relief(name: str) -> bool:
    """Имя слоя относится к производным рельефа."""
    return name.startswith(_RELIEF_PREFIXES)


def _grid_of(site_dir: Path) -> Grid:
    """Сетка участка: из sample_meta.json, иначе из любого tif."""
    meta = site_dir / _META
    if meta.exists():
        return Grid(**json.loads(meta.read_text(encoding="utf-8"))["grid"])
    tifs = sorted((site_dir / _LAYERS).glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"{site_dir}: нет слоёв в {_LAYERS}/")
    return read_raster(tifs[0]).grid


def _relief_params(layer_names: set[str]) -> dict[str, tuple | float | int]:
    """Восстанавливает параметры рельефа из имён существующих слоёв.

    Так не нужен AppConfig участка: если в папке лежит hillshade_az_135.tif,
    значит азимут 135 использовался. Радиус openness в имени не кодируется —
    его берём из --app-config, если он есть, иначе из дефолтов.
    """
    azimuths = tuple(
        float(n.removeprefix("hillshade_az_"))
        for n in sorted(layer_names)
        if n.startswith("hillshade_az_")
    )
    slrm = tuple(
        float(n.removeprefix("slrm_sigma_"))
        for n in sorted(layer_names)
        if n.startswith("slrm_sigma_")
    )
    curv = tuple(
        float(n.removeprefix("curvature_sigma_"))
        for n in sorted(layer_names)
        if n.startswith("curvature_sigma_")
    )
    tpi = tuple(
        int(n.removeprefix("tpi_radius_"))
        for n in sorted(layer_names)
        if n.startswith("tpi_radius_")
    )
    tri = tuple(
        int(n.removeprefix("tri_radius_"))
        for n in sorted(layer_names)
        if n.startswith("tri_radius_")
    )
    return {
        "hillshade_azimuths": azimuths or (45.0,),
        "slrm_sigmas": slrm or (2.0,),
        "curvature_sigmas": curv or (2.0,),
        "tpi_radii": tpi or (5,),
        "tri_radii": tri or (3,),
    }


def _openness_params(app_config: Path | None, layer_names: set[str]) -> tuple[int, int]:
    """(radius, n_directions) для openness: из AppConfig, иначе дефолт 10/16."""
    if app_config is not None and app_config.exists():
        import yaml

        raw = yaml.safe_load(app_config.read_text(encoding="utf-8")).get("relief", {})
        return int(raw.get("openness_radius", 10)), int(raw.get("openness_n_directions", 16))
    if "positive_openness" in layer_names:
        logger.warning(
            "  openness: радиус неизвестен (нет --app-config). Беру 10/16 — "
            "если исходный участок считался с другим радиусом, positive_openness/"
            "negative_openness/sky_view_factor будут несопоставимы со старыми."
        )
    return 10, 16


def rebuild_site(
    site_dir: Path,
    out_dir: Path | None,
    app_config: Path | None,
    max_dist_px: float,
    in_place: bool,
    dry_run: bool,
) -> dict[str, float] | None:
    """Пересобирает один участок. Возвращает покрытие до/после или None при пропуске."""
    layers_dir = site_dir / _LAYERS
    dtm_path = layers_dir / "dtm.tif"
    if not dtm_path.exists():
        logger.warning("%s: нет dtm.tif, пропуск", site_dir.name)
        return None

    grid = _grid_of(site_dir)
    dtm = read_raster(dtm_path, grid=grid)
    measured = dtm.valid_mask
    names = {p.stem for p in layers_dir.glob("*.tif")}

    # mean_z НЕ используем как донора: он считается по ВСЕМ точкам ячейки,
    # включая растительность и кровли, поэтому в лесу это верх крон, а не
    # земля. Только отчёт — сколько площади он мог бы закрыть.
    if "mean_z" in names:
        mz = read_raster(layers_dir / "mean_z.tif", grid=grid)
        if mz.data.shape == dtm.data.shape:
            potential = float((~measured & mz.valid_mask).mean())
            logger.info("  mean_z закрыл бы ещё %.1f%% площади", 100 * potential)

    filled_data, filled_mask = fill_narrow_gaps(
        values=dtm.data,
        nodata=dtm.nodata,
        grid=grid,
        max_dist_px=max_dist_px,
    )
    still = ~np.isfinite(filled_data) | (filled_data == dtm.nodata)

    stats = {
        "measured": float(measured.mean()),
        "filled": float(filled_mask.mean()),
        "nodata": float(still.mean()),
    }
    logger.info(
        "  покрытие: measured %.1f%% -> +filled %.1f%% = %.1f%% (nodata %.1f%%)",
        stats["measured"] * 100,
        stats["filled"] * 100,
        (stats["measured"] + stats["filled"]) * 100,
        stats["nodata"] * 100,
    )
    if dry_run:
        return stats
    if not filled_mask.any():
        # Дыр нет: производные от этого DTM уже посчитаны корректно.
        # Но участок должен попасть в --out, иначе его не будет в новом sites_dir.
        logger.info("  дыр нет, копирую как есть")
        target = layers_dir if in_place else (out_dir or Path("data/rebuilt")) / site_dir.name
        if not in_place:
            (target / _LAYERS).mkdir(parents=True, exist_ok=True)
            for src_tif in layers_dir.glob("*.tif"):
                shutil.copy2(src_tif, target / _LAYERS / src_tif.name)
            for extra in ("samples",):
                if (site_dir / extra).is_dir():
                    shutil.copytree(site_dir / extra, target / extra, dirs_exist_ok=True)
        return stats

    params = _relief_params(names)
    radius, n_dir = _openness_params(app_config, names)
    logger.info(
        "  рельеф: azimuths=%s slrm=%s curv=%s tpi=%s tri=%s openness=%d/%d",
        params["hillshade_azimuths"],
        params["slrm_sigmas"],
        params["curvature_sigmas"],
        params["tpi_radii"],
        params["tri_radii"],
        radius,
        n_dir,
    )
    relief = build_relief_layers(
        dtm=filled_data,
        pixel_size=grid.pixel_size,
        hillshade_azimuths=params["hillshade_azimuths"],
        hillshade_altitude=45.0,
        hillshade_z_factor=1.0,
        slrm_sigmas=params["slrm_sigmas"],
        openness_radius=radius,
        openness_n_directions=n_dir,
        curvature_sigmas=params["curvature_sigmas"],
        tpi_radii=params["tpi_radii"],
        tri_radii=params["tri_radii"],
        nodata=dtm.nodata,
    )

    target = layers_dir if in_place else (out_dir or Path("data/rebuilt")) / site_dir.name / _LAYERS
    target.mkdir(parents=True, exist_ok=True)

    src = dtm.source
    write_raster(
        Layer(
            name="dtm",
            data=filled_data,
            grid=grid,
            nodata=dtm.nodata,
            source=src,
            unit="meters",
        ),
        target / "dtm.tif",
    )
    written = 0
    for name, arr in relief.items():
        if name not in names:
            continue  # не добавляем слои, которых не было: контракт каналов
        write_raster(
            Layer(name=name, data=arr, grid=grid, nodata=dtm.nodata, source=src),
            target / f"{name}.tif",
        )
        written += 1

    # Облачные признаки и rgb копируем как есть, если пишем в отдельную папку.
    copied = 0
    if not in_place:
        for name in names - {"dtm"} - set(relief):
            shutil.copy2(layers_dir / f"{name}.tif", target / f"{name}.tif")
            copied += 1
        for extra in ("samples",):
            src_dir = site_dir / extra
            if src_dir.is_dir():
                shutil.copytree(src_dir, target.parent / extra, dirs_exist_ok=True)

    logger.info(
        "  записано: dtm + %d производных%s -> %s",
        written,
        f", {copied} слоёв скопировано" if copied else "",
        target,
    )
    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=Path("configs/dataset_real.yaml"))
    p.add_argument("--site", action="append", default=None, help="имя участка (можно повторять)")
    p.add_argument("--sites-dir", type=Path, default=None, help="переопределить data.sites_dir")
    p.add_argument("--out", type=Path, default=Path("data/rebuilt"))
    p.add_argument("--in-place", action="store_true", help="писать в те же layers/ (перезапись!)")
    p.add_argument(
        "--app-config",
        type=Path,
        default=None,
        help="AppConfig участка (configs/sites/*.yaml)",
    )
    p.add_argument("--max-dist-px", type=float, default=4.0)
    p.add_argument("--dry-run", action="store_true", help="только посчитать покрытие")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_pipeline_config(args.config)
    root = args.sites_dir or cfg.data.sites_dir

    if args.site:
        sites = [root / s for s in args.site]
    else:
        sites = sorted(
            d for d in root.iterdir() if d.is_dir() and (d / _LAYERS / "dtm.tif").exists()
        )

    app_cfg = args.app_config
    if app_cfg is None and sites:
        candidate = Path("configs/sites") / f"{sites[0].name}.yaml"
        if candidate.exists():
            app_cfg = candidate
            logger.info("AppConfig: %s", app_cfg)

    rows = []
    for site_dir in sites:
        if not site_dir.is_dir():
            logger.warning("%s: нет папки", site_dir)
            continue
        logger.info("\n=== %s ===", site_dir.name)
        try:
            stats = rebuild_site(
                site_dir=site_dir,
                out_dir=args.out,
                app_config=app_cfg,
                max_dist_px=args.max_dist_px,
                in_place=args.in_place,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001 — один участок не должен ронять остальные
            logger.error("  ОШИБКА: %s: %s", type(exc).__name__, exc)
            continue
        if stats:
            rows.append((site_dir.name, stats))

    if rows:
        print(f"\n{'участок':34} {'measured':>9} {'filled':>8} {'nodata':>8} {'итого':>8}")
        print("-" * 72)
        for name, s in rows:
            print(
                f"{name[:34]:34} {s['measured']:8.1%} {s['filled']:8.1%} "
                f"{s['nodata']:8.1%} {s['measured'] + s['filled']:8.1%}"
            )
        m = float(np.mean([s["measured"] for _, s in rows]))
        f = float(np.mean([s["filled"] for _, s in rows]))
        n = float(np.mean([s["nodata"] for _, s in rows]))
        print("-" * 72)
        print(f"{'СРЕДНЕЕ':34} {m:8.1%} {f:8.1%} {n:8.1%} {m + f:8.1%}")
    print(
        "\nрежим:",
        "DRY-RUN (ничего не записано)"
        if args.dry_run
        else ("IN-PLACE" if args.in_place else str(args.out)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
