import torch

from expds.models import DiceFocalLoss, UNetSegmenter


def test_forward_shape():
    K = 5
    model = UNetSegmenter(in_channels=19, classes=K, encoder_weights=None)
    x = torch.randn(2, 19, 256, 256)
    y = model(x)
    assert y.shape == (2, K, 256, 256)


def test_backward():
    K = 3
    model = UNetSegmenter(in_channels=19, classes=K, encoder_weights=None)
    x = torch.randn(2, 19, 256, 256, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_dice_focal_empty_and_perfect():
    criterion = DiceFocalLoss()

    logits = torch.zeros(2, 3, 32, 32)
    targets = torch.zeros(2, 3, 32, 32)
    loss_empty = criterion(logits, targets)
    assert torch.isfinite(loss_empty)

    logits_perfect = torch.full((2, 3, 32, 32), 10.0)
    targets_perfect = torch.ones(2, 3, 32, 32)
    loss_perfect = criterion(logits_perfect, targets_perfect)
    assert loss_perfect < 0.1
