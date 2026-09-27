import torch
import torch.nn as nn

from .base import BaseSegmenter


class StubSegmenter(BaseSegmenter):
    """Заглушка для CI/e2e: одна свёртка kxk, логиты (B, K, H, W)."""

    def __init__(self, in_channels: int, classes: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv2d(in_channels, classes, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
