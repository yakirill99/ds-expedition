from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSegmenter(nn.Module, ABC):
    """
    Базовый интерфейс сегментатора.

    Вход:  (B, C, H, W)
    Выход: (B, 1, H, W)  — логиты (или вероятности после sigmoid)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты формы (B, 1, H, W)."""
        ...

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Инференс: возвращает вероятности (B, 1, H, W)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
