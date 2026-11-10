"""Аудит каналов по участкам: таблица «участок -> сколько каналов есть».

Быстрый режим (по файловой системе, без rasterio/torch): для каждого
участка проверяется наличие layers/<канал>.tif из data.channels конфига.
Ответ на вопрос «доступно=3 из 31»: у новых участков не собран полный
стек — модель учится на нулях вместо рельефа (mean_z, z_std и все
производные мертвы автоматически).

Примеры:
    uv run python scripts/audit_channels.py --config configs/night3.yaml
    uv run python scripts/audit_channels.py --config configs/night3.yaml --deep
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_LAYERS_DIR = "layers"


def audit_fs(site_dirs: list[Path], channels: list[str]) -> tuple[list[list[bool]], list[str]]:
    """Матрица «участок x канал» по наличию tif-файлов."""
    rows, present_sites = [], []
    for d in site_dirs:
        layers = d / _LAYERS_DIR
        row = [(layers / f"{c}.tif").exists() for c in channels]
        rows.append(row)
        present_sites.append(d.name)
    return rows, present_sites


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--all-sites",
        action="store_true",
        help="весь sites_dir, а не только train+val",
    )
    ap.add_argument(
        "--deep",
        action="store_true",
        help="загрузить участки (torch/rasterio) и показать n_available",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    from expds.pipeline_config import load_pipeline_config

    cfg = load_pipeline_config(args.config)
    channels = list(cfg.data.channels)
    sites_root = Path(cfg.data.sites_dir)

    names: list[str] = []
    if args.all_sites:
        names = sorted(d.name for d in sites_root.iterdir() if d.is_dir())
    else:
        seen: set[str] = set()
        for s in (*cfg.train.train_sites, *cfg.train.val_sites):
            if s not in seen:
                seen.add(s)
                names.append(s)
    site_dirs = [sites_root / n for n in names]
    site_dirs = [d for d in site_dirs if d.exists()]

    rows, present = audit_fs(site_dirs, channels)

    # Сводная таблица
    short = [c[:12] for c in channels]
    print(f"{'участок':<34}" + "".join(f"{s:>12}" for s in short) + f"{'итого':>8}")
    counts: list[int] = []
    for name, row in zip(present, rows, strict=True):
        counts.append(sum(row))
        line = f"{name:<34}" + "".join(("+" if v else ".")[:12].rjust(12) for v in row)
        print(line + f"{sum(row):>6}/{len(channels)}")

    print("\nСводка по каналам (сколько участков без канала):")
    missing: dict[str, int] = {}
    for j, c in enumerate(channels):
        n_missing = sum(1 for row in rows if not row[j])
        if n_missing:
            missing[c] = n_missing
            sites_miss = [present[i] for i, row in enumerate(rows) if not row[j]]
            preview = ", ".join(sites_miss[:5]) + ("..." if len(sites_miss) > 5 else "")
            print(f"  {c}: нет на {n_missing}/{len(rows)} ({preview})")
    if not missing:
        print("  все каналы на месте на всех участках")

    full = [present[i] for i, r in enumerate(rows) if sum(r) == len(channels)]
    print(f"\nполный стек: {len(full)}/{len(rows)} участков")

    if args.deep:
        print("\nDeep-режим (реальная сборка сэмпла):")
        from expds.tiles.dataset import load_site

        for d in site_dirs:
            try:
                site = load_site(
                    d,
                    channels=channels,
                    classes=list(cfg.targets.classes),
                    heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
                    labels_crs=cfg.data.labels_crs,
                    labels_file=cfg.data.labels_file,
                    normalize=cfg.data.normalize,
                )
                print(f"  {d.name}: доступно {int(site.channel_mask.sum())}/{len(channels)}")
            except Exception as e:  # noqa: BLE001
                print(f"  {d.name}: ОШИБКА {e}")


if __name__ == "__main__":
    main()
