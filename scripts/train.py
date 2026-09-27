"""Обучение по YAML датасета.

uv run python scripts/train.py --config configs/dataset_synthetic.yaml [--epochs 10 --arch stub]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from expds.models.factory import input_divisor
from expds.pipeline_config import load_pipeline_config
from expds.train import train_from_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--tiles-per-epoch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--arch", choices=["unet", "stub"])
    ap.add_argument("--run-dir", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_pipeline_config(args.config)
    train_kw = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "tiles_per_epoch": args.tiles_per_epoch,
            "device": args.device,
        }.items()
        if v is not None
    }
    cfg = replace(cfg, train=replace(cfg.train, **train_kw))
    if args.arch:
        cfg = replace(cfg, model=replace(cfg.model, arch=args.arch))
        if cfg.tiles.tile_px % input_divisor(cfg.model):
            ap.error(f"tile_px={cfg.tiles.tile_px} не делится на {input_divisor(cfg.model)}")
    train_from_config(cfg, args.config, args.run_dir)


if __name__ == "__main__":
    main()
