import segmentation_models_pytorch as smp
import torch

from .base import BaseSegmenter


class UNetSegmenter(BaseSegmenter):
    """
    U-Net baseline.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        in_channels: int = 19,
        classes: int = 1,
        activation: str | None = None,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
