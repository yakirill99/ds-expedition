"""Обучение в смеси синтетических и реальных участков (фаза B2).

train_from_config берёт участки из одного data.sites_dir. Для смеси
синтетики и реальных площадок корни разные, поэтому участки задаются
путями явно: --train-site / --val-site (можно повторять).

Перед обучением печатается аудит баланса: сколько положительных пикселей
и объектов на класс по каждому участку. На 006 только mound, поэтому
rampart/ditch приходят исключительно из синтетики — без аудита это
обнаруживается только по нулевым метрикам после обучения.

    uv run python scripts/train_mix.py \\
        --config configs/dataset_real.yaml \\
        --train-site data/synthetic/site_1 \\
        --train-site data/raw/006_Осечки_1.1км \\
        --val-site data/synthetic/site_3 \\
        --audit-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from expds.eval.metric import read_objects
from expds.models.factory import input_divisor
from expds.pipeline_config import PipelineConfig, load_pipeline_config
from expds.tiles.dataset import SiteData, load_site
from expds.train import fit

logger = logging.getLogger(__name__)

#: Доля положительных пикселей, ниже которой класс считаем непредставленным.
_SCARCE_FRACTION = 1e-5


@dataclass(frozen=True)
class ClassStat:
    """Представленность одного класса на одном участке."""

    site: str
    cls: str
    seg_px: int
    total_px: int
    hm_peaks: int

    @property
    def fraction(self) -> float:
        """Доля положительных пикселей маски."""
        return self.seg_px / self.total_px if self.total_px else 0.0

    @property
    def is_scarce(self) -> bool:
        """Класс практически отсутствует."""
        return self.fraction < _SCARCE_FRACTION


def site_stats(site: SiteData, classes: tuple[str, ...]) -> list[ClassStat]:
    """Считает представленность классов на участке.

    Args:
        site: загруженный участок.
        classes: порядок классов (контракт модели).

    Returns:
        По одной записи на класс.
    """
    if site.y_seg is None:
        return []
    _, h, w = site.y_seg.shape
    total = h * w
    out: list[ClassStat] = []
    for i, name in enumerate(classes):
        seg_px = int((site.y_seg[i] > 0).sum())
        hm_peaks = 0
        if site.y_hm is not None:
            hm_peaks = int((site.y_hm[i] >= 0.5).sum())
        out.append(ClassStat(site.name, name, seg_px, total, hm_peaks))
    return out


def format_audit(stats: list[ClassStat], classes: tuple[str, ...]) -> str:
    """Таблица баланса классов по участкам + итог."""
    if not stats:
        return "аудит: участков с разметкой нет"
    sites = sorted({s.site for s in stats})
    width = max([len("участок"), *(len(s) for s in sites)])
    head = f"{'участок':<{width}}  " + "  ".join(f"{c:>12}" for c in classes)
    lines = [head, "-" * len(head)]
    by = {(s.site, s.cls): s for s in stats}
    for site in sites:
        cells = []
        for c in classes:
            st = by.get((site, c))
            cells.append(f"{st.seg_px:>12}" if st else f"{'—':>12}")
        lines.append(f"{site:<{width}}  " + "  ".join(cells))
    lines.append("-" * len(head))
    totals = []
    for c in classes:
        tot = sum(s.seg_px for s in stats if s.cls == c)
        totals.append(f"{tot:>12}")
    lines.append(f"{'ВСЕГО px':<{width}}  " + "  ".join(totals))
    peaks = []
    for c in classes:
        tot = sum(s.hm_peaks for s in stats if s.cls == c)
        peaks.append(f"{tot:>12}")
    lines.append(f"{'ВСЕГО hm':<{width}}  " + "  ".join(peaks))
    return "\n".join(lines)


def scarce_classes(stats: list[ClassStat], classes: tuple[str, ...]) -> list[str]:
    """Классы, которых почти нет во всей выборке."""
    out = []
    for c in classes:
        if sum(s.seg_px for s in stats if s.cls == c) == 0:
            out.append(c)
    return out


def load_sites(paths: list[Path], cfg: PipelineConfig) -> list[SiteData]:
    """Загружает участки по явным путям, минуя data.sites_dir.

    Args:
        paths: каталоги участков.
        cfg: конфиг (даёт контракт каналов, классов, нормализацию).

    Returns:
        Список SiteData в порядке путей.

    Raises:
        FileNotFoundError: если каталога нет.
    """
    sites: list[SiteData] = []
    for p in paths:
        if not p.is_dir():
            raise FileNotFoundError(f"участок не найден: {p}")
        site = load_site(
            p,
            channels=cfg.data.channels,
            classes=cfg.targets.classes,
            heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
            labels_crs=cfg.data.labels_crs,
            labels_file=cfg.data.labels_file,
            normalize=cfg.data.normalize,
        )
        logger.info(
            "%s: grid %s %dx%d, разметка=%s",
            site.name,
            site.grid.crs,
            site.grid.height,
            site.grid.width,
            site.has_labels,
        )
        sites.append(site)
    return sites


def _val_pairs(
    paths: list[Path], sites: list[SiteData], cfg: PipelineConfig
) -> list[tuple[SiteData, list[Any]]]:
    """(участок, GT в CRS метрики) для валидации.

    GT ищется в каталоге, который передали в --val-site, а не
    реконструируется из data.sites_dir: у синтетики и реальных
    участков разные корни.

    Args:
        paths: каталоги валидационных участков (порядок = sites).
        sites: загруженные участки.
        cfg: конфиг (нужны labels_file и labels_crs).

    Returns:
        Пары (участок, объекты GT). Пустой список объектов — валидация
        на участке не считается, о чём пишется warning.
    """
    pairs: list[tuple[SiteData, list[Any]]] = []
    for path, site in zip(paths, sites, strict=True):
        gt_path = path / cfg.data.labels_file
        if not gt_path.is_file():
            logger.error(
                "%s: нет GT (%s) — val по этому участку будет нулевым",
                site.name,
                gt_path,
            )
            pairs.append((site, []))
            continue
        objs = read_objects(gt_path, cfg.data.labels_crs)
        if not objs:
            logger.error("%s: GT пустой (%s)", site.name, gt_path)
        pairs.append((site, objs))
    return pairs


def main(argv: list[str] | None = None) -> int:
    """CLI смешанного обучения."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--train-site", type=Path, action="append", default=[])
    ap.add_argument("--val-site", type=Path, action="append", default=[])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--tiles-per-epoch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--arch", choices=["unet", "stub"])
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--audit-only", action="store_true", help="только баланс классов, без обучения")
    ap.add_argument("--allow-scarce", action="store_true", help="не падать при пустом классе")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.train_site:
        ap.error("нужен хотя бы один --train-site")

    cfg = load_pipeline_config(args.config)
    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "tiles_per_epoch": args.tiles_per_epoch,
            "device": args.device,
        }.items()
        if v is not None
    }
    if overrides:
        cfg = replace(cfg, train=replace(cfg.train, **overrides))
    if args.arch:
        cfg = replace(cfg, model=replace(cfg.model, arch=args.arch))
    div = input_divisor(cfg.model)
    if cfg.tiles.tile_px % div:
        ap.error(f"tile_px={cfg.tiles.tile_px} не делится на {div}")

    classes = tuple(cfg.targets.classes)
    train_sites = load_sites(args.train_site, cfg)
    val_sites = load_sites(args.val_site, cfg) if args.val_site else []

    stats = [st for s in train_sites for st in site_stats(s, classes)]
    print("\n=== баланс классов (train) ===")
    print(format_audit(stats, classes))

    scarce = scarce_classes(stats, classes)
    if scarce:
        msg = f"классы без единого положительного пикселя: {scarce}"
        if not args.allow_scarce and not args.audit_only:
            ap.error(f"{msg}. Добавьте участки или --allow-scarce")
        logger.warning(msg)

    hm_total = sum(s.hm_peaks for s in stats)
    if hm_total == 0:
        logger.warning("heatmap пустой на всех train-участках: hm-голова учиться не будет")

    if args.audit_only:
        return 0

    run_dir = Path(
        args.run_dir or Path(cfg.train.runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_mix"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sites.json").write_text(
        json.dumps(
            {
                "train": [str(p) for p in args.train_site],
                "val": [str(p) for p in args.val_site],
                "balance": [
                    {"site": s.site, "cls": s.cls, "seg_px": s.seg_px, "hm": s.hm_peaks}
                    for s in stats
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    res = fit(cfg, train_sites, _val_pairs(args.val_site, val_sites, cfg), run_dir)
    print(f"\nrun_dir: {run_dir}")
    print(f"лучший ключ отбора: {res['best']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
