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
        y_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """logits -> скалярный лосс и компоненты.

        valid (B, H, W) — попиксельная маска (nodata, паддинг).
        y_valid (B, K) или (K,) — валидность класса на участке: 0 означает
        «модальность не снималась», а не «объектов нет». Вклад такого
        класса в лосс обнуляется, иначе на участках без геофизики голова
        finds_points учится «находок нет нигде» и глушит те участки,
        где они есть.
        """
        c = self.cfg
        heads = split_heads(logits.float(), self.k)
        seg, hm = heads["seg"], heads["hm"]
        y_seg = y_seg.to(device=seg.device, dtype=seg.dtype)
        y_hm = y_hm.to(device=hm.device, dtype=hm.dtype)
        if y_seg.shape != seg.shape or y_hm.shape != hm.shape:
            raise ValueError("Target shapes must match their respective heads")
        b, k, h, w = seg.shape
        if valid is None:
            m = torch.ones_like(seg[:, :1])
        else:
            if valid.shape != (b, h, w):
                raise ValueError(f"valid: expected {(b, h, w)}, got {tuple(valid.shape)}")
            m = valid.to(device=seg.device, dtype=seg.dtype).unsqueeze(1)
            if not torch.isfinite(m).all() or not ((m == 0) | (m == 1)).all():
                raise ValueError("valid must be a finite binary mask")

        # Учитываем наблюдаемость класса до редукции по батчу.
        if y_valid is None:
            observed = torch.ones((b, k), device=seg.device, dtype=seg.dtype)
        else:
            observed = y_valid.to(device=seg.device, dtype=seg.dtype)
            if observed.shape == (k,):
                observed = observed.unsqueeze(0).expand(b, k)
            elif observed.shape == (1, k):
                observed = observed.expand(b, k)
            elif observed.shape != (b, k):
                raise ValueError(
                    f"y_valid: expected {(k,)}, {(1, k)} or {(b, k)}, got {tuple(observed.shape)}"
                )
            if not torch.isfinite(observed).all():
                raise ValueError("y_valid must contain finite values")
            observed = observed.clamp(0.0, 1.0)

        mask = m * observed[:, :, None, None]
        axes = (0, 2, 3)
        counts = mask.sum(dim=axes)
        # Полностью ненаблюдаемый класс даёт ноль, а не лосс по фону.
        weights = (counts > 0).to(seg.dtype)
        weights = weights / weights.sum().clamp_min(1.0)
        n_px = counts.clamp_min(1.0)

        p = torch.sigmoid(seg)
        inter = (p * y_seg * mask).sum(dim=axes)
        den = (p * mask).sum(dim=axes) + (y_seg * mask).sum(dim=axes)
        dice_k = 1.0 - (2.0 * inter + c.smooth) / (den + c.smooth)
        dice = (dice_k * weights).sum()

        bce = F.binary_cross_entropy_with_logits(seg, y_seg, reduction="none")
        p_t = p * y_seg + (1 - p) * (1 - y_seg)
        a_t = c.focal_alpha * y_seg + (1 - c.focal_alpha) * (1 - y_seg)
        focal_k = (a_t * (1 - p_t).pow(c.focal_gamma) * bce * mask).sum(dim=axes) / n_px
        focal = (focal_k * weights).sum()

        q = torch.sigmoid(hm)
        bce_h = F.binary_cross_entropy_with_logits(hm, y_hm, reduction="none")
        hm_num = ((q - y_hm).abs().pow(c.hm_beta) * bce_h * mask).sum(dim=axes)
        hm_den = (y_hm * mask).sum(dim=axes).clamp_min(c.hm_min_norm)
        hm_loss = ((hm_num / hm_den) * weights).sum()
        total = (
            c.seg_weight * (c.dice_weight * dice + c.focal_weight * focal) + c.hm_weight * hm_loss
        )
        return {
            "loss": total,
            "seg_dice": dice.detach(),
            "seg_focal": focal.detach(),
            "hm": hm_loss.detach(),
        }
