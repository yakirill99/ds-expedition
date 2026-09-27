"""Обучение: AdamW + cosine, AMP (CUDA), WeightedRandomSampler по pos_fraction.

Валидация — полный прогон участка (predict -> stitch -> postprocess -> метрика в eval CRS),
лучший чекпойнт по val F1 (без val-участков — по train loss).
Перед обучением — oracle: таргеты val-участков как «вероятности» через тот же post + метрику.
Oracle F1 заметно < 1 значит ошибку в цепочке post / метрика, а не в модели.
Артефакты: <run_dir>/log.jsonl, run.json, config.yaml, best.pt, last.pt.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from expds.eval.metric import (
    EvalObject,
    MatchResult,
    evaluate,
    objects_from_detections,
    read_objects,
)
from expds.infer import autocast, detect, predict_probs, resolve_device
from expds.io.geojson import Detection, write_detections
from expds.models.base import BaseSegmenter
from expds.models.checkpoint import save_checkpoint
from expds.models.factory import build_model
from expds.models.koz_loss import KozLoss
from expds.tiles.dataset import KozDataset, SiteData, load_site
from expds.utils.seed import seed_everything

if TYPE_CHECKING:
    from expds.pipeline_config import PipelineConfig


@dataclass(frozen=True)
class TrainConfig:
    """Параметры обучения (секция `train` YAML)."""

    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    device: str = "auto"
    num_workers: int = 2
    seed: int = 0
    tiles_per_epoch: int | None = None
    train_sites: tuple[str, ...] = ()
    val_sites: tuple[str, ...] = ()
    runs_dir: str = "runs"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> TrainConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"train: unknown keys {sorted(unknown)}")
        for key in ("train_sites", "val_sites"):
            if key in d:
                d[key] = tuple(str(s) for s in (d[key] or ()))
        for key in ("lr", "weight_decay"):
            if key in d:
                d[key] = float(d[key])  # YAML 1.1 читает 1e-3 как строку
        cfg = cls(**d)
        if cfg.epochs < 1 or cfg.batch_size < 1:
            raise ValueError("train: epochs и batch_size должны быть >= 1")
        if set(cfg.train_sites) & set(cfg.val_sites):
            raise ValueError("train: участки train и val пересекаются")
        return cfg


# --- валидация -----------------------------------------------------------------


def _to_eval_objects(
    dets: Sequence[Detection], src_crs: str, dst_crs: str, tmp_dir: Path
) -> list[EvalObject]:
    """Детекции crs.internal -> EvalObject в CRS метрики (через write_detections, как в сабмите)."""
    if not dets:
        return []
    if src_crs == dst_crs:
        return objects_from_detections(dets)
    path = tmp_dir / "_val_pred.geojson"
    io_logger = logging.getLogger("expds.io.geojson")
    level = io_logger.level
    io_logger.setLevel(logging.WARNING)  # без INFO-строки на каждую валидацию
    try:
        write_detections(path, list(dets), src_crs, dst_crs)
        return read_objects(path, dst_crs)
    finally:
        io_logger.setLevel(level)


def _score(
    probs: np.ndarray,
    site: SiteData,
    gts: Sequence[EvalObject],
    cfg: PipelineConfig,
    eval_crs: str,
    tmp_dir: Path,
    total: MatchResult,
) -> None:
    """Вероятности участка -> детекции -> метрика; счётчики добавляются в total."""
    dets = detect(probs, site, cfg.targets.classes, cfg.post)
    preds = _to_eval_objects(dets, site.grid.crs, eval_crs, tmp_dir)
    r = evaluate(preds, list(gts), cfg.eval)["__all__"]
    total.tp += r.tp
    total.fp += r.fp
    total.fn += r.fn


def _oracle(
    val_sets: Sequence[tuple[SiteData, Sequence[EvalObject]]],
    cfg: PipelineConfig,
    eval_crs: str,
    tmp_dir: Path,
) -> MatchResult:
    """Таргеты (y_seg, y_hm) как идеальные вероятности через post + метрику."""
    total = MatchResult()
    for site, gts in val_sets:
        if site.has_labels:
            probs = np.concatenate([site.y_seg, site.y_hm]).astype(np.float32)
            _score(probs, site, gts, cfg, eval_crs, tmp_dir, total)
    return total


def _validate(
    model: BaseSegmenter,
    val_sets: Sequence[tuple[SiteData, Sequence[EvalObject]]],
    cfg: PipelineConfig,
    device: torch.device,
    eval_crs: str,
    tmp_dir: Path,
) -> MatchResult:
    total = MatchResult()
    classes = cfg.targets.classes
    for site, gts in val_sets:
        probs = predict_probs(
            model,
            site,
            len(classes),
            cfg.tiles.tile_px,
            cfg.tiles.overlap_px,
            cfg.stitch,
            device,
            cfg.train.batch_size,
            cfg.train.amp,
        )
        _score(probs, site, gts, cfg, eval_crs, tmp_dir, total)
    return total


def _fmt(rec: Mapping[str, Any]) -> str:
    s = (
        f"ep {rec['epoch']:3d} loss {rec['train_loss']:.4f} "
        f"(dice {rec['train_seg_dice']:.3f} focal {rec['train_seg_focal']:.4f} "
        f"hm {rec['train_hm']:.3f})"
    )
    if "val_f1" in rec:
        s += (
            f" | val f1 {rec['val_f1']:.3f} p {rec['val_precision']:.3f} "
            f"r {rec['val_recall']:.3f} (tp {rec['val_tp']} fp {rec['val_fp']} fn {rec['val_fn']})"
        )
    return s + f" | {rec['time_s']:.1f}s"


# --- обучение ------------------------------------------------------------------


def fit(
    cfg: PipelineConfig,
    train_sites: Sequence[SiteData],
    val_sets: Sequence[tuple[SiteData, Sequence[EvalObject]]],
    run_dir: Path,
    eval_crs: str | None = None,
    model: BaseSegmenter | None = None,
) -> dict[str, Any]:
    """Цикл обучения. val_sets: (участок, GT в eval_crs). eval_crs по умолчанию data.labels_crs.

    Returns:
        {"best": ключ отбора, "history": записи log.jsonl, "oracle": метрика oracle или None}.
    """
    tc = cfg.train
    seed_everything(tc.seed)
    device = resolve_device(tc.device)
    classes = cfg.targets.classes
    channels = train_sites[0].channel_names
    if tuple(channels) != tuple(cfg.data.channels):
        raise ValueError(f"каналы участков {channels} != data.channels {cfg.data.channels}")
    eval_crs = eval_crs or cfg.data.labels_crs

    model = (model or build_model(cfg.model, len(channels), len(classes))).to(device)
    loss_fn = KozLoss(len(classes), cfg.loss)
    ds = KozDataset(train_sites, cfg.tiles.tile_px, cfg.tiles.overlap_px)
    weights = torch.as_tensor(
        np.asarray(ds.sample_weights(cfg.tiles.pos_fraction), dtype=np.float64)
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=tc.tiles_per_epoch or len(ds),
        replacement=True,
        generator=torch.Generator().manual_seed(tc.seed),
    )
    dl = DataLoader(
        ds,
        batch_size=tc.batch_size,
        sampler=sampler,
        num_workers=tc.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=tc.num_workers > 0,
    )
    opt = AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=tc.epochs)
    use_amp = tc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    best_key = -math.inf
    history: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        oracle = None
        if val_sets:
            oracle = _oracle(val_sets, cfg, eval_crs, Path(tmp)).to_dict()
            print(
                f"oracle val f1 {oracle['f1']:.3f} p {oracle['precision']:.3f} "
                f"r {oracle['recall']:.3f} (tp {oracle['tp']} fp {oracle['fp']} fn {oracle['fn']})",
                flush=True,
            )
        for epoch in range(1, tc.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            sums: dict[str, float] = {}
            nb = 0
            for batch in dl:
                x = batch["x"].to(device, non_blocking=True)
                y_seg = batch["y_seg"].to(device, non_blocking=True)
                y_hm = batch["y_hm"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with autocast(device, use_amp):
                    logits = model(x)
                out = loss_fn(logits, y_seg, y_hm, valid)
                scaler.scale(out["loss"]).backward()
                scaler.step(opt)
                scaler.update()
                for k, v in out.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                nb += 1
            sched.step()

            rec: dict[str, Any] = {"epoch": epoch, "lr": opt.param_groups[0]["lr"]}
            rec.update({f"train_{k}": v / max(nb, 1) for k, v in sums.items()})
            if val_sets:
                res = _validate(model, val_sets, cfg, device, eval_crs, Path(tmp))
                rec.update({f"val_{k}": v for k, v in res.to_dict().items()})
                key = res.f1
            else:
                key = -rec["train_loss"]
            rec["time_s"] = time.perf_counter() - t0
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(_fmt(rec), flush=True)
            history.append(rec)

            extra = {"epoch": epoch, "train_loss": rec["train_loss"], "val_f1": rec.get("val_f1")}
            if key > best_key:
                best_key = key
                save_checkpoint(run_dir / "best.pt", model, cfg.model, channels, classes, extra)
        save_checkpoint(run_dir / "last.pt", model, cfg.model, channels, classes, extra)
    return {"best": best_key, "history": history, "oracle": oracle}


def _git_rev() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train_from_config(
    cfg: PipelineConfig, config_path: Path | None = None, run_dir: Path | None = None
) -> Path:
    """Загружает участки train/val из data.sites_dir и обучает. Возвращает run_dir."""
    tc = cfg.train
    if not tc.train_sites:
        raise ValueError("train.train_sites пуст")

    def load(name: str) -> SiteData:
        return load_site(
            cfg.data.sites_dir / name,
            channels=cfg.data.channels,
            classes=cfg.targets.classes,
            heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
            labels_crs=cfg.data.labels_crs,
            labels_file=cfg.data.labels_file,
            normalize=cfg.data.normalize,
        )

    train_sites = [load(n) for n in tc.train_sites]
    val_sets = [
        (load(n), read_objects(cfg.data.sites_dir / n / cfg.data.labels_file, cfg.data.labels_crs))
        for n in tc.val_sites
    ]
    run_dir = Path(
        run_dir or Path(tc.runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.model.arch}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy(config_path, run_dir / "config.yaml")
    meta = {
        "git": _git_rev(),
        "torch": torch.__version__,
        "device": str(resolve_device(tc.device)),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_train_sites": len(train_sites),
        "n_val_sites": len(val_sets),
        "config": asdict(cfg),
    }
    (run_dir / "run.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"run_dir: {run_dir} | device: {meta['device']} {meta['cuda_device'] or ''}", flush=True)
    fit(cfg, train_sites, val_sets, run_dir)
    return run_dir
