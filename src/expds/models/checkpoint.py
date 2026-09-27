from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .base import BaseSegmenter
from .factory import ModelConfig, build_model


def save_checkpoint(
    path: str | Path,
    model: BaseSegmenter,
    model_cfg: ModelConfig,
    channel_names: Sequence[str],
    classes: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "channel_names": list(channel_names),
            "classes": list(classes),
            "extra": dict(extra or {}),
        },
        path,
    )


def _check(name: str, expected: Sequence[str] | None, got: list[str]) -> None:
    if expected is not None and list(expected) != got:
        raise ValueError(f"{name} mismatch: checkpoint {got} vs expected {list(expected)}")


def load_checkpoint(
    path: str | Path,
    channel_names: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[BaseSegmenter, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    _check("channel_names", channel_names, payload["channel_names"])
    _check("classes", classes, payload["classes"])
    cfg = ModelConfig.from_dict(payload["model_cfg"])
    model = build_model(cfg, len(payload["channel_names"]), len(payload["classes"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
