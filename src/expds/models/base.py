from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSegmenter(nn.Module, ABC):
    """
    Базовый интерфейс сегментатора.

    Вход:  (B, C, H, W)
    Выход: (B, K, H, W) — логиты (по одному каналу на класс)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты формы (B, K, H, W)."""
        ...

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Инференс: вероятности (B, K, H, W) после sigmoid по каналам."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
