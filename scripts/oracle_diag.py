"""Оракульная диагностика по участкам: где пайплайн теряет эталон.

Прогоняет РАЗМЕТКУ через тот же post+матчинг, что и модель, и печатает
потери по стадиям (растеризация / valid / min_area / слияние / NMS / IoU).

Устойчивый режим (без тяжёлых зависимостей) — аудиторам хватит; полный —
загружает участки целиком (нужен torch/rasterio на машине запуска).

Примеры:
    uv run python scripts/oracle_diag.py --config configs/night3.yaml
    uv run python scripts/oracle_diag.py --config configs/night3.yaml \
        --sites 130_Курганы_1,150_Курганы_21 --sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from expds.eval.oracle import OracleReport, oracle_report
from expds.features.grid import Grid
from expds.io.geojson import read_labels
from expds.post.vectorize import PostConfig

# expds.pipeline_config импортируется лениво в main(): он тянет train/models
# (torch), а диагностике torch не нужен — только numpy/rasterio.

_META_PATH = "sample_meta.json"
_LAYERS_DIR = "layers"


def _site_grid(site_dir: Path) -> Grid:
    """Сетка участка — так же, как tiles.dataset._site_grid:
    из sample_meta.json, иначе из первого tif в layers/."""
    meta_path = site_dir / _META_PATH
    if meta_path.exists():
        return Grid(**json.loads(meta_path.read_text(encoding="utf-8"))["grid"])
    tifs = sorted((site_dir / _LAYERS_DIR).glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"{site_dir}: нет ни {_META_PATH}, ни слоёв в {_LAYERS_DIR}/")
    from expds.data.raster import read_raster

    return read_raster(tifs[0]).grid


def _valid_mask(site_dir: Path, grid: Grid):
    """Маска валидности из dtm.tif — как её считает load_site
    (чтение с сеткой участка + заполнение дыр < 7 px ближайшим значением).
    None, если dtm недоступен."""
    from scipy import ndimage

    from expds.data.raster import read_raster

    path = site_dir / _LAYERS_DIR / "dtm.tif"
    if not path.exists():
        print("  (dtm.tif нет — считаем всю площадь валидной)")
        return None
    layer = read_raster(path, grid=grid)
    valid = layer.valid_mask
    # Аналог tiles.dataset._fill_nodata(dtm, 7.0): дыры глубиной <= 7 px
    # заливаются ближайшим валидным значением, дальше остаются пропусками.
    if valid.any() and not valid.all():
        dist = ndimage.distance_transform_edt(~valid)
        valid = valid | ((~valid) & (dist <= 7.0))
    return valid


def diagnose_site(
    site_dir: Path,
    classes: list[str],
    post: PostConfig,
    sigma_px: float,
    labels_crs: str,
    labels_file: str,
    eval_cfg=None,
) -> OracleReport:
    """Полный отчёт по одному участку."""
    grid = _site_grid(site_dir)
    feats = read_labels(site_dir / labels_file, grid, src_crs=labels_crs)
    valid = _valid_mask(site_dir, grid)
    return oracle_report(feats, grid, classes, post, sigma_px, valid=valid, eval_cfg=eval_cfg)


def print_report(name: str, rep: OracleReport) -> None:
    m = rep.match
    print(f"\n=== {name} ===")
    print(
        f"GT {rep.gt_total} | после post: p {m.precision:.3f} r {m.recall:.3f} "
        f"f1 {m.f1:.3f} (tp {m.tp} fp {m.fp} fn {m.fn})"
    )
    print(
        f"потери: вне сетки {rep.raster_zero} | valid/nodata {rep.valid_killed} | "
        f"min_area {rep.raster_below_min_area} | слияние (не разделено) {rep.merged_unresolved} "
        f"(слитых компонент {rep.merged_components}) | NMS {rep.nms_suppressed} "
        f"(было детекций {rep.pre_nms_dets}, осталось {rep.post_nms_dets})"
    )
    if rep.reasons:
        parts = [f"{k}: {v}" for k, v in sorted(rep.reasons.items(), key=lambda kv: -kv[1])]
        print("причины FN: " + ", ".join(parts))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sites", type=str, help="имена через запятую; по умолчанию train.val_sites")
    ap.add_argument("--labels-file", type=str, help="переопределить data.labels_file")
    ap.add_argument(
        "--sweep", action="store_true", help="перебрать варианты post и дать рекомендацию"
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    from expds.pipeline_config import load_pipeline_config

    cfg = load_pipeline_config(args.config)
    names = (
        [s.strip() for s in args.sites.split(",") if s.strip()]
        if args.sites
        else list(cfg.train.val_sites)
    )
    if not names:
        ap.error("нет участков: укажите --sites или train.val_sites в конфиге")
    labels_file = args.labels_file or cfg.data.labels_file

    variants: list[tuple[str, PostConfig]] = [("конфиг как есть", cfg.post)]
    if args.sweep:
        base = cfg.post
        variants += [
            ("split_touching=on", replace(base, split_touching=True)),
            (
                "split_touching=on + min_area=6",
                replace(base, split_touching=True, min_area_px=6.0),
            ),
            (
                "split_touching=on + min_area=6 + mind=3",
                replace(
                    base,
                    split_touching=True,
                    min_area_px=6.0,
                    split_min_distance_px=3,
                ),
            ),
            ("min_area=6", replace(base, min_area_px=6.0)),
            ("nms_radius=0", replace(base, nms_radius_px=0.0)),
        ]

    for vi, (vname, post) in enumerate(variants):
        if vi:
            print(f"\n########## вариант: {vname} ##########")
        total = OracleReport()
        for name in names:
            site_dir = cfg.data.sites_dir / name
            if not (site_dir / labels_file).exists():
                print(f"=== {name} === пропущен: нет {labels_file}")
                continue
            rep = diagnose_site(
                site_dir,
                list(cfg.targets.classes),
                post,
                cfg.targets.heatmap_sigma_px,
                cfg.data.labels_crs,
                labels_file,
                eval_cfg=cfg.eval,
            )
            print_report(name, rep)
            total.gt_total += rep.gt_total
            total.raster_zero += rep.raster_zero
            total.raster_below_min_area += rep.raster_below_min_area
            total.valid_killed += rep.valid_killed
            total.merged_components += rep.merged_components
            total.merged_unresolved += rep.merged_unresolved
            total.pre_nms_dets += rep.pre_nms_dets
            total.post_nms_dets += rep.post_nms_dets
            total.nms_suppressed += rep.nms_suppressed
            total.match.tp += rep.match.tp
            total.match.fp += rep.match.fp
            total.match.fn += rep.match.fn
            for k, v in rep.reasons.items():
                total.reasons[k] = total.reasons.get(k, 0) + v
        print_report(f"ИТОГО ({len(names)} участков)", total)

    if args.sweep:
        print(
            "\nРекомендация: выбирайте вариант с oracle f1 -> 1.0; "
            "его post-секцию переносите в конфиг ретрейна."
        )


if __name__ == "__main__":
    main()
