"""Тесты лосса: per-class validity mask."""

from __future__ import annotations

import pytest
import torch

from expds.models.koz_loss import _CLASS_NAMES, _CLASS_WEIGHTS_BY_NAME, KozLoss


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


# Регрессионные проверки coverage, nodata и наблюдаемости классов.
def _regression_loss_inputs():
    torch.manual_seed(42)
    logits = torch.randn(3, 4, 5, 6, requires_grad=True)
    seg = torch.rand(3, 2, 5, 6).round()
    hm = torch.rand(3, 2, 5, 6)
    valid = torch.ones(3, 5, 6, dtype=torch.bool)
    observed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    return (logits, seg, hm, valid, observed)


def test_regression_loss_batch_permutation():
    x, s, h, v, o = _regression_loss_inputs()
    fn = KozLoss(2)
    p = torch.tensor([2, 0, 1])
    a = fn(x, s, h, v, o)
    b = fn(x[p], s[p], h[p], v[p], o[p])
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_regression_loss_ignored_logits_and_gradients():
    x, s, h, v, o = _regression_loss_inputs()
    fn = KozLoss(2)
    loss = fn(x, s, h, v, o)["loss"]
    loss.backward()
    ignored = torch.cat([o == 0, o == 0], dim=1)[:, :, None, None].expand_as(x)
    assert torch.all(x.grad[ignored] == 0)
    changed = x.detach().clone()
    changed[ignored] = 30
    torch.testing.assert_close(loss, fn(changed, s, h, v, o)["loss"])


@pytest.mark.parametrize("mask_kind", ["classes", "pixels"])
def test_regression_loss_empty_observation_zero(mask_kind):
    x, s, h, v, o = _regression_loss_inputs()
    if mask_kind == "classes":
        o.zero_()
    else:
        v.zero_()
    loss = KozLoss(2)(x, s, h, v, o)["loss"]
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.all(x.grad == 0)


def test_regression_loss_common_class_mask_matches_broadcast():
    x, s, h, v, o = _regression_loss_inputs()
    fn = KozLoss(2)
    common = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(
        fn(x, s, h, v, common)["loss"], fn(x, s, h, v, common.expand(3, 2))["loss"]
    )


def test_point_only_class_learns_hm() -> None:
    """Точечный класс (y_seg==0, y_hm с пиком) получает градиент в hm-голову.

    Регрессия: присутствие класса определялось по gt_counts>0 (по y_seg),
    поэтому finds_points — единственный Point-класс — не получал градиент
    вообще, hm-голова оставалась на prior-инициализации и находки не
    детектировались.
    """
    torch.manual_seed(0)
    k, b, t = 2, 1, 32
    loss_fn = KozLoss(k)
    logits = torch.zeros(b, 2 * k, t, t, requires_grad=True)
    y_seg = torch.zeros(b, k, t, t)
    y_hm = torch.zeros(b, k, t, t)
    y_hm[:, 1, 16, 16] = 1.0  # класс 1 существует только в heatmap

    out = loss_fn(logits, y_seg, y_hm)
    out["loss"].backward()

    assert out["hm"].item() > 0
    hm_grad = logits.grad[0, k + 1]  # hm-канал класса 1
    assert hm_grad.abs().sum() > 0
    assert torch.isfinite(out["loss"]).all()


def test_point_only_class_respects_y_valid() -> None:
    """Точечный класс с y_valid=0 (модальность не снималась) не получает градиент."""
    torch.manual_seed(0)
    k, b, t = 2, 1, 32
    loss_fn = KozLoss(k)
    y_seg = torch.zeros(b, k, t, t)
    y_hm = torch.zeros(b, k, t, t)
    y_hm[:, 1, 16, 16] = 1.0

    lg1 = torch.zeros(b, 2 * k, t, t, requires_grad=True)
    loss_fn(lg1, y_seg, y_hm, None, torch.ones(k))["loss"].backward()
    assert lg1.grad[0, k + 1].abs().sum() > 0  # класс наблюдаем — hm-голова учится

    lg2 = torch.zeros(b, 2 * k, t, t, requires_grad=True)
    loss_fn(lg2, y_seg, y_hm, None, torch.tensor([1.0, 0.0]))["loss"].backward()
    assert lg2.grad[0, k + 1].abs().sum() == 0  # y_valid=0 — градиента нет


def test_class_weights_follow_names() -> None:
    """Веса привязаны к именам классов, а не к позиции в списке."""
    torch.manual_seed(0)
    # Тот же регламентный набор в другом порядке: вес каждого слота = вес его имени.
    order = (2, 0, 4, 6, 1, 5, 3)
    classes = tuple(_CLASS_NAMES[i] for i in order)
    loss_fn = KozLoss(7, classes=classes)
    total = sum(loss_fn._weights.tolist())
    assert total == pytest.approx(100.0)
    for slot, name in enumerate(classes):
        assert loss_fn._weights[slot].item() == pytest.approx(_CLASS_WEIGHTS_BY_NAME[name])
    # Нормированные веса: сумма = 1, каждый = вес класса / 100.
    w = loss_fn._weighted(torch.ones(7), torch.zeros(1))
    assert w.sum().item() == pytest.approx(1.0)
    for slot, name in enumerate(classes):
        assert w[slot].item() == pytest.approx(_CLASS_WEIGHTS_BY_NAME[name] / total)

    # Вне регламентного набора — равные веса, без ошибок (синтетика/тесты).
    assert KozLoss(2, classes=("mound", "ditch"))._weights is None

    # Регламентный набор с опечаткой — ошибка конфига, не молчаливые чужие веса.
    bad = list(_CLASS_NAMES)
    bad[3] = "fortifikatsiya"
    with pytest.raises(ValueError, match="нет весов"):
        KozLoss(7, classes=tuple(bad))
