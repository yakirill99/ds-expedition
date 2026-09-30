"""Инференс: участок(и) со слоями -> GeoJSON детекций в CRS разметки (EPSG:3857).

Участок — папка с layers/ (формат build_layers / make_synthetic). --data указывает на участок
или на папку участков; детекции всех участков сливаются в один FeatureCollection.
Модель и её конфиг берутся из чекпойнта; каналы и классы сверяются с YAML.
Сети нет: torch.load локально, энкодер без предобученных весов.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from expds.infer import detect, predict_probs, resolve_device
from expds.io.geojson import write_detections
from expds.models.checkpoint import load_checkpoint
from expds.models.factory import ModelConfig, input_divisor
from expds.pipeline_config import load_pipeline_config
from expds.tiles.dataset import load_site
from expds.utils.profiler import Profiler

logger = logging.getLogger(__name__)

_LAYERS_DIR = "layers"
_NO_LABELS = "__inference_no_labels__.geojson"  # на инференсе разметку не читаем


def find_sites(data: Path) -> list[Path]:
    """Участок (есть layers/) или список участков в папке."""
    data = Path(data)
    if (data / _LAYERS_DIR).is_dir():
        return [data]
    sites = sorted(p for p in data.iterdir() if (p / _LAYERS_DIR).is_dir()) if data.is_dir() else []
    if not sites:
        raise FileNotFoundError(
            f"{data}: нет участков с папкой {_LAYERS_DIR}/ "
            "(сырьё LAS -> слои: scripts/build_layers.py)"
        )
    return sites


def merge_geojson(parts: Sequence[Path], out: Path) -> None:
    """Сливает FeatureCollection-ы (одна CRS) в один файл."""
    features: list[Any] = []
    crs_member = None
    for p in parts:
        fc = json.loads(Path(p).read_text(encoding="utf-8"))
        features += fc.get("features", [])
        crs_member = crs_member or fc.get("crs")
    merged: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs_member:
        merged["crs"] = crs_member
    Path(out).write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")


def run(
    config: Path,
    data: Path,
    weights: Path,
    out: Path,
    device: str = "auto",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Полный инференс. Возвращает сводку: out, n_sites, n_detections, seconds."""
    t0 = time.perf_counter()
    prof = Profiler()
    with prof.stage("конфиг и веса"):
        cfg = load_pipeline_config(Path(config))
        dev = resolve_device(device)
        model, meta = load_checkpoint(
            weights,
            channel_names=cfg.data.channels,
            classes=cfg.targets.classes,
            map_location="cpu",
        )
        model.to(dev)
        div = input_divisor(ModelConfig.from_dict(meta["model_cfg"]))
        if cfg.tiles.tile_px % div:
            raise ValueError(f"tiles.tile_px={cfg.tiles.tile_px} не делится на {div}")
    sites = find_sites(Path(data))
    classes = cfg.targets.classes
    bs = batch_size or cfg.train.batch_size
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_det = 0
    with tempfile.TemporaryDirectory() as tmp:
        parts: list[Path] = []
        for i, site_dir in enumerate(sites):
            name = site_dir.name
            with prof.stage(f"{name}: слои"):
                site = load_site(
                    site_dir,
                    channels=cfg.data.channels,
                    classes=classes,
                    heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
                    labels_crs=cfg.data.labels_crs,
                    labels_file=_NO_LABELS,
                    normalize=cfg.data.normalize,
                )
            with prof.stage(f"{name}: модель + сшивка"):
                probs = predict_probs(
                    model,
                    site,
                    len(classes),
                    cfg.tiles.tile_px,
                    cfg.tiles.overlap_px,
                    cfg.stitch,
                    dev,
                    bs,
                    cfg.train.amp,
                )
            with prof.stage(f"{name}: постобработка"):
                dets = detect(probs, site, classes, cfg.post)
            del probs
            with prof.stage(f"{name}: запись"):
                part = out if len(sites) == 1 else Path(tmp) / f"part_{i}.geojson"
                write_detections(part, dets, site.grid.crs, cfg.data.labels_crs)
            parts.append(part)
            n_det += len(dets)
        if len(sites) > 1:
            merge_geojson(parts, out)
    prof.report()  # печатает таблицу сам
    return {
        "out": str(out),
        "n_sites": len(sites),
        "n_detections": n_det,
        "seconds": time.perf_counter() - t0,
    }


def main(
    argv: list[str] | None = None,
    default_config: Path | None = None,
    default_weights: Path | None = None,
) -> int:
    """CLI: общий для inference/entrypoint.py и scripts/predict.py."""
    ap = argparse.ArgumentParser(description="Инференс КОЗ №3: слои -> GeoJSON")
    ap.add_argument("--data", type=Path, required=True, help="участок или папка участков")
    ap.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    ap.add_argument(
        "--weights", type=Path, default=default_weights, required=default_weights is None
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stats = run(args.config, args.data, args.weights, args.out, args.device, args.batch_size)
    print(
        f"OK: {stats['n_detections']} детекций, участков {stats['n_sites']}, "
        f"{stats['seconds']:.1f} с -> {stats['out']}",
        flush=True,
    )
    return 0
