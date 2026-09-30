"""Инференс по участку: тайлы (KozDataset) -> модель -> сшивка -> постобработка."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from expds.fusion.stitch import StitchConfig, Stitcher
from expds.io.geojson import Detection
from expds.models.base import BaseSegmenter
from expds.models.factory import out_channels
from expds.post.vectorize import PostConfig, postprocess
from expds.tiles.dataset import KozDataset, SiteData


def resolve_device(name: str) -> torch.device:
    """auto -> cuda при наличии, иначе cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    """fp16 autocast только на CUDA; иначе no-op."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def predict_probs(
    model: BaseSegmenter,
    site: SiteData,
    num_classes: int,
    tile_px: int,
    overlap_px: int,
    stitch_cfg: StitchConfig | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    amp: bool = False,
) -> np.ndarray:
    """Вероятности (2K, H, W) по всему участку. Режим модели (train/eval) восстанавливается."""
    device = torch.device(device)
    ds = KozDataset([site], tile_px, overlap_px)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    st = Stitcher(out_channels(num_classes), site.grid.height, site.grid.width, tile_px, stitch_cfg)
    was_training = model.training
    model.eval()
    try:
        for batch in dl:
            x = batch["x"].to(device, non_blocking=True)
            with autocast(device, amp):
                logits = model(x)
            st.add_batch(torch.sigmoid(logits.float()).cpu(), batch["window"])
    finally:
        model.train(was_training)
    return st.result(fill=0.0)


def detect(
    probs: np.ndarray, site: SiteData, classes: Sequence[str], post_cfg: PostConfig | None = None
) -> list[Detection]:
    """Вероятности участка -> детекции в crs.internal (nodata участка отсекается)."""
    return postprocess(probs, site.grid, classes, post_cfg, valid=site.valid)
