from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn

from .base import BaseSegmenter
from .stub import StubSegmenter
from .unet import UNetSegmenter

HEADS = ("seg", "hm")
ARCHS = ("unet", "stub")


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "unet"
    encoder: str = "resnet34"
    stub_kernel: int = 3
    prior_prob: float = 0.01

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> ModelConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"model: unknown keys {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.arch not in ARCHS:
            raise ValueError(f"model.arch must be one of {ARCHS}, got {cfg.arch!r}")
        if not 0.0 < cfg.prior_prob < 1.0:
            raise ValueError("model.prior_prob must be in (0, 1)")
        return cfg


def out_channels(num_classes: int) -> int:
    return len(HEADS) * num_classes


def input_divisor(cfg: ModelConfig) -> int:
    """H и W тайла должны делиться на это число."""
    return 32 if cfg.arch == "unet" else 1


def build_model(cfg: ModelConfig, in_channels: int, num_classes: int) -> BaseSegmenter:
    k = out_channels(num_classes)
    if cfg.arch == "stub":
        model: BaseSegmenter = StubSegmenter(in_channels, k, kernel_size=cfg.stub_kernel)
    else:
        model = UNetSegmenter(
            encoder_name=cfg.encoder, encoder_weights=None, in_channels=in_channels, classes=k
        )
    _init_prior(model, k, cfg.prior_prob)
    return model


def _init_prior(model: nn.Module, k: int, prior: float) -> None:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None or last.out_channels != k or last.bias is None:
        raise RuntimeError("output Conv2d with bias not found")
    with torch.no_grad():
        last.bias.fill_(-math.log((1.0 - prior) / prior))


def split_heads(logits: torch.Tensor, num_classes: int) -> dict[str, torch.Tensor]:
    """(B, 2K, H, W) -> {"seg": (B, K, H, W), "hm": (B, K, H, W)}."""
    k2 = out_channels(num_classes)
    if logits.ndim != 4 or logits.shape[1] != k2:
        raise ValueError(f"expected (B, {k2}, H, W), got {tuple(logits.shape)}")
    return dict(zip(HEADS, torch.split(logits, num_classes, dim=1), strict=True))
