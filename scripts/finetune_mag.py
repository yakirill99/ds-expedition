"""Fine-tune от готового чекпоинта: участки из config + магнит из data/raw.

train_mix.main() собирает модель с нуля, здесь передаём model= в fit.
Участки берутся из cfg.train.train_sites (data.sites_dir), а отсутствующие
там имена 20X/21X доискиваются в data/raw — геофизика лежит только там.

    uv run python scripts/finetune_mag.py --config configs/aero2_mag_v1.yaml \
        --init runs/n2_rgb/last.pt --lr 3e-5 --epochs 40 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

from expds.models.checkpoint import load_checkpoint
from expds.models.factory import input_divisor
from expds.pipeline_config import load_pipeline_config
from expds.train import fit
from expds.train_mix import _val_pairs, format_audit, load_sites, scarce_classes, site_stats

logger = logging.getLogger("finetune")


def resolve(name: str, cfg, raw_dir: Path) -> Path | None:
    """Путь участка: data.sites_dir, иначе data/raw."""
    for base in (Path(cfg.data.sites_dir), raw_dir):
        p = base / name
        if (p / cfg.data.labels_file).is_file() or (p / "layers").is_dir():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=Path("runs/n2_rgb/last.pt"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-sites", type=int, default=0, help="0 — все")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--dry-run", action="store_true", help="только аудит и оценка памяти")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_pipeline_config(args.config)
    if cfg.tiles.tile_px % input_divisor(cfg.model):
        ap.error("tile_px не делится на input_divisor")

    val_names = list(cfg.train.val_sites)
    train_names = [n for n in cfg.train.train_sites if n not in set(val_names)]
    paths, missing = [], []
    for n in train_names:
        p = resolve(n, cfg, args.raw_dir)
        (paths if p else missing).append(p or n)
    if missing:
        logger.warning("не найдено, пропущено: %s", missing)
    if args.max_sites:
        paths = paths[: args.max_sites]
    val_paths = [p for n in val_names if (p := resolve(n, cfg, args.raw_dir))]
    logger.info("train %d | val %d", len(paths), len(val_paths))

    train_sites = load_sites(paths, cfg)
    stats = [st for s in train_sites for st in site_stats(s, tuple(cfg.targets.classes))]
    print(format_audit(stats, tuple(cfg.targets.classes)))
    mem = sum(s.grid.height * s.grid.width for s in train_sites) * len(cfg.data.channels) * 4
    print(f"\nпамять участков: {mem / 2**30:.1f} ГиБ")
    scarce = scarce_classes(stats, tuple(cfg.targets.classes))
    if scarce:
        logger.warning("классы без пикселей: %s", scarce)
    if args.dry_run:
        return 0

    model, payload = load_checkpoint(
        args.init, channel_names=cfg.data.channels, classes=cfg.targets.classes
    )
    logger.info(
        "чекпоинт: %s | каналов %d | классов %d",
        args.init,
        len(payload["channel_names"]),
        len(payload["classes"]),
    )

    tc = replace(cfg.train, lr=args.lr, epochs=args.epochs)
    if args.batch_size:
        tc = replace(tc, batch_size=args.batch_size)
    if args.device:
        tc = replace(tc, device=args.device)
    cfg = replace(cfg, train=tc)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir or Path(cfg.train.runs_dir) / f"{stamp}_mag")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = Path(args.config).read_text(encoding="utf-8")
    (run_dir / "config.yaml").write_text(cfg_text, encoding="utf-8")

    val_pairs = _val_pairs(val_paths, load_sites(val_paths, cfg), cfg)
    res = fit(cfg, train_sites, val_pairs, run_dir, model=model)
    print(f"\nrun_dir: {run_dir}\nлучший ключ: {res['best']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
