"""Подбор порогов постобработки по val: инференс один раз, перебор по кэшу.

uv run python scripts/sweep_post.py --config configs/dataset_synthetic.yaml \
    --data data/synthetic --weights runs/<run>/best.pt

Фаза 1: predict_probs по каждому участку из --data -> <cache-dir>/<site>.npz.
Повторный запуск переиспользует кэш, если чекпойнт не менялся (--force — пересчитать).
Фаза 2: перебор сетки seg_thr/hm_thr/min_area_px/nms_radius_px -> таблица топ-N,
YAML-сниппет секции post для конфига, опционально --out report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from expds.eval.metric import EvalObject, objects_from_labels, read_labels
from expds.infer import predict_probs, resolve_device
from expds.models.checkpoint import load_checkpoint
from expds.pipeline_config import PipelineConfig, load_pipeline_config
from expds.post.sweep import (
    PostGrid,
    SweepRow,
    best_post_yaml,
    format_sweep_table,
    load_probs_cache,
    save_probs_cache,
    sweep_postparams,
    sweep_report,
)
from expds.predict import find_sites
from expds.tiles.dataset import load_site

logger = logging.getLogger(__name__)


def _csv_floats(text: str) -> tuple[float, ...]:
    """'0.3, 0.5' -> (0.3, 0.5)."""
    return tuple(float(x) for x in text.split(",") if x.strip())


def _weights_tag(weights: Path) -> str:
    """Первые 12 hex sha256 чекпойнта — тег свежести кэша."""
    h = hashlib.sha256()
    with Path(weights).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _cache_fresh(path: Path, tag: str) -> bool:
    """Кэш существует и посчитан от того же чекпойнта."""
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        return str(z["weights_tag"]) == tag


def infer_and_cache(
    cfg: PipelineConfig,
    data: Path,
    weights: Path,
    cache_dir: Path,
    tag: str,
    force: bool = False,
    device: str = "auto",
    batch_size: int | None = None,
) -> dict[str, Path]:
    """Фаза 1: predict_probs по участкам -> .npz. Возвращает пути кэшей."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    need: list[Path] = []
    for site_dir in find_sites(data):
        path = cache_dir / f"{site_dir.name}.npz"
        paths[site_dir.name] = path
        if force or not _cache_fresh(path, tag):
            need.append(site_dir)
        else:
            logger.info("%s: кэш актуален, инференс не нужен (%s)", site_dir.name, path)
    if not need:
        return paths

    model, _meta = load_checkpoint(
        weights, channel_names=cfg.data.channels, classes=cfg.targets.classes, map_location="cpu"
    )
    dev = resolve_device(device)
    model.to(dev)
    for site_dir in need:
        site = load_site(
            site_dir,
            channels=cfg.data.channels,
            classes=cfg.targets.classes,
            heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
            labels_crs=cfg.data.labels_crs,
            normalize=cfg.data.normalize,
        )
        probs = predict_probs(
            model,
            site,
            len(cfg.targets.classes),
            cfg.tiles.tile_px,
            cfg.tiles.overlap_px,
            cfg.stitch,
            dev,
            batch_size or cfg.train.batch_size,
            cfg.train.amp,
        )
        save_probs_cache(
            paths[site_dir.name],
            site_dir.name,
            probs,
            site.grid,
            site.valid,
            weights_tag=tag,
        )
        logger.info("%s: probs закэшированы (%s)", site_dir.name, paths[site_dir.name])
    return paths


def load_val_gt(
    cfg: PipelineConfig, data: Path, cache_paths: dict[str, Path]
) -> dict[str, list[EvalObject]]:
    """GT val-участков: read_labels по Grid из кэша -> EvalObject (crs.internal)."""
    gts: dict[str, list[EvalObject]] = {}
    for site_dir in find_sites(data):
        name = site_dir.name
        labels = site_dir / cfg.data.labels_file
        if not labels.is_file():
            raise FileNotFoundError(
                f"{name}: нет разметки {labels} — sweep требует val-участки с GT"
            )
        feats = read_labels(labels, load_probs_cache(cache_paths[name]).grid, cfg.data.labels_crs)
        gts[name] = objects_from_labels(feats)
    return gts


def run(
    config: Path,
    data: Path,
    weights: Path,
    cache_dir: Path = Path("runs") / "sweep_cache",
    grid: PostGrid | None = None,
    force: bool = False,
    device: str = "auto",
    batch_size: int | None = None,
) -> tuple[list[SweepRow], PipelineConfig]:
    """Полный свип. Возвращает (строки по убыванию F1, конфиг)."""
    cfg = load_pipeline_config(config)
    tag = _weights_tag(weights)
    cache_paths = infer_and_cache(
        cfg, data, weights, cache_dir, tag, force=force, device=device, batch_size=batch_size
    )
    caches = {name: load_probs_cache(p) for name, p in cache_paths.items()}
    gts = load_val_gt(cfg, data, cache_paths)
    rows = sweep_postparams(caches, gts, cfg.targets.classes, cfg.post, cfg.eval, grid)
    return rows, cfg


def main(argv: list[str] | None = None) -> int:
    """CLI: инференс + кэш + перебор сетки + отчёт."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="YAML датасета")
    ap.add_argument("--data", type=Path, required=True, help="папка val-участков")
    ap.add_argument("--weights", type=Path, required=True, help="чекпойнт best.pt")
    ap.add_argument("--cache-dir", type=Path, default=Path("runs") / "sweep_cache")
    ap.add_argument("--seg-thrs", default="0.35,0.5,0.65", help="CSV значений seg_thr")
    ap.add_argument("--hm-thrs", default="0.25,0.4", help="CSV значений hm_thr")
    ap.add_argument("--min-areas", default="16,48", help="CSV значений min_area_px")
    ap.add_argument("--nms-radii", default="6,10", help="CSV значений nms_radius_px")
    ap.add_argument("--top", type=int, default=10, help="строк в таблице")
    ap.add_argument("--out", type=Path, help="куда сохранить JSON-отчёт")
    ap.add_argument("--force", action="store_true", help="пересчитать кэш, даже если свежий")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pgrid = PostGrid(
        seg_thr=_csv_floats(args.seg_thrs),
        hm_thr=_csv_floats(args.hm_thrs),
        min_area_px=_csv_floats(args.min_areas),
        nms_radius_px=_csv_floats(args.nms_radii),
    )
    rows, cfg = run(
        args.config,
        args.data,
        args.weights,
        args.cache_dir,
        pgrid,
        force=args.force,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(format_sweep_table(rows, top=args.top))
    print()
    print(best_post_yaml(rows, cfg.post))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(sweep_report(rows, cfg.eval, pgrid), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("отчёт сохранён: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
