from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factory import split_heads


@dataclass(frozen=True)
class LossConfig:
    seg_weight: float = 1.0
    hm_weight: float = 1.0
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    hm_beta: float = 2.0
    hm_min_norm: float = 50.0
    smooth: float = 1.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> LossConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"loss: unknown keys {sorted(unknown)}")
        return cls(**d)


class KozLoss(nn.Module):
    """
    Лосс для выхода [seg_0..K-1, hm_0..K-1].
    seg: Dice по классам + focal с alpha-балансом; hm: soft focal (QFL) по гауссовой heatmap,
    нормировка на max(масса heatmap, hm_min_norm): без пола батч без точек даёт сумму по всем
    пикселям (лосс ~1e4, переполнение градиента в fp16). 50 ~ масса точки 2*pi*sigma^2 при sigma=3.
    valid (B, H, W) исключает nodata и паддинг. Считается в float32 (безопасно для AMP).
    """

    def __init__(self, num_classes: int, cfg: LossConfig | None = None):
        super().__init__()
        self.k = num_classes
        self.cfg = cfg or LossConfig()

    def forward(
        self,
        logits: torch.Tensor,
        y_seg: torch.Tensor,
        y_hm: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        c = self.cfg
        heads = split_heads(logits.float(), self.k)
        seg, hm = heads["seg"], heads["hm"]
        y_seg, y_hm = y_seg.float(), y_hm.float()
        if valid is None:
            m = torch.ones_like(seg[:, :1])
        else:
            m = valid.to(seg.dtype).reshape(seg.shape[0], 1, *seg.shape[2:])
        n = (m.sum() * self.k).clamp_min(1.0)

        p = torch.sigmoid(seg)
        pm, tm = p * m, y_seg * m
        inter = (pm * tm).sum(dim=(0, 2, 3))
        den = pm.sum(dim=(0, 2, 3)) + tm.sum(dim=(0, 2, 3))
        dice = 1.0 - ((2.0 * inter + c.smooth) / (den + c.smooth)).mean()

        bce = F.binary_cross_entropy_with_logits(seg, y_seg, reduction="none")
        p_t = p * y_seg + (1 - p) * (1 - y_seg)
        a_t = c.focal_alpha * y_seg + (1 - c.focal_alpha) * (1 - y_seg)
        focal = (a_t * (1 - p_t).pow(c.focal_gamma) * bce * m).sum() / n

        q = torch.sigmoid(hm)
        bce_h = F.binary_cross_entropy_with_logits(hm, y_hm, reduction="none")
        hm_loss = ((q - y_hm).abs().pow(c.hm_beta) * bce_h * m).sum() / (
            (y_hm * m).sum().clamp_min(c.hm_min_norm)
        )

        total = (
            c.seg_weight * (c.dice_weight * dice + c.focal_weight * focal) + c.hm_weight * hm_loss
        )
        return {
            "loss": total,
            "seg_dice": dice.detach(),
            "seg_focal": focal.detach(),
            "hm": hm_loss.detach(),
        }
