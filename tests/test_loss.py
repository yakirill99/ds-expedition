"""Тесты лосса: per-class validity mask."""

from __future__ import annotations

import torch

from expds.models.koz_loss import KozLoss


def test_y_valid_zeroes_class() -> None:
    """Класс с y_valid=0 не влияет на значение лосса."""
    torch.manual_seed(0)
    k, b, t = 3, 2, 16
    loss_fn = KozLoss(k)
    logits = torch.randn(b, 2 * k, t, t)
    y_seg = torch.zeros(b, k, t, t)
    y_seg[:, 0, 8:12, 8:12] = 1.0
    y_hm = torch.zeros(b, k, t, t)

    full = loss_fn(logits, y_seg, y_hm, None, torch.ones(k))["loss"]
    masked = loss_fn(logits, y_seg, y_hm, None, torch.tensor([1.0, 1.0, 0.0]))["loss"]
    none_valid = loss_fn(logits, y_seg, y_hm, None, torch.zeros(k))["loss"]

    assert masked.item() != full.item()  # маска что-то изменила
    assert torch.isfinite(none_valid)  # все нули не дают nan/inf


def test_y_valid_none_keeps_old_behaviour() -> None:
    """Без y_valid веса равные — прежнее поведение."""
    torch.manual_seed(0)
    k = 2
    loss_fn = KozLoss(k)
    logits = torch.randn(1, 2 * k, 8, 8)
    y_seg = torch.zeros(1, k, 8, 8)
    y_hm = torch.zeros(1, k, 8, 8)

    a = loss_fn(logits, y_seg, y_hm)["loss"]
    b = loss_fn(logits, y_seg, y_hm, None, torch.ones(k))["loss"]
    assert a.item() == b.item()
