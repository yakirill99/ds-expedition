"""Конфиг костяка: секции data / targets / tiles / model / loss / stitch / post / eval того же YAML,
что читает load_config роли 1.

load_config роли 1 лишние секции игнорирует, поэтому один файл на датасет.
data / targets / tiles обязательны; model / loss / stitch / post / eval необязательны
(дефолты dataclass), неизвестные ключи внутри них — ошибка.
Секция train добавится вместе со своими модулями.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from expds.eval.metric import EvalConfig
from expds.fusion.stitch import StitchConfig
from expds.models.factory import ModelConfig, input_divisor
from expds.models.koz_loss import LossConfig
from expds.post.vectorize import PostConfig
from expds.tiles.dataset import SUPPORTED_NORMALIZE

_KEY_DATA = "data"
_KEY_TARGETS = "targets"
_KEY_TILES = "tiles"
_KEY_MODEL = "model"
_KEY_LOSS = "loss"
_KEY_STITCH = "stitch"
_KEY_POST = "post"
_KEY_EVAL = "eval"


@dataclass(frozen=True)
class DataConfig:
    """Где участки и какие каналы (порядок = контракт модели)."""

    sites_dir: Path
    labels_file: str
    channels: tuple[str, ...]
    normalize: str


@dataclass(frozen=True)
class TargetsConfig:
    """Классы и параметры таргетов."""

    classes: tuple[str, ...]
    heatmap_sigma_px: float


@dataclass(frozen=True)
class TilesConfig:
    """Тайлинг и балансировка."""

    tile_px: int
    overlap_px: int
    pos_fraction: float


@dataclass(frozen=True)
class PipelineConfig:
    """Корневой конфиг костяка."""

    data: DataConfig
    targets: TargetsConfig
    tiles: TilesConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    post: PostConfig = field(default_factory=PostConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Секция-словарь или ошибка."""
    value = raw.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"В конфиге нет секции '{key}' (или она не словарь)")
    return value


def _optional_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Секция-словарь или {} если её нет."""
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise KeyError(f"Секция '{key}' не словарь")
    return value


def _unique(values: tuple[str, ...], what: str) -> tuple[str, ...]:
    """Проверка уникальности."""
    if len(set(values)) != len(values):
        raise ValueError(f"{what}: есть дубликаты {values}")
    return values


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Читает секции костяка из YAML.

    Args:
        path: YAML датасета.

    Returns:
        PipelineConfig.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_raw = _section(raw, _KEY_DATA)
    targets_raw = _section(raw, _KEY_TARGETS)
    tiles_raw = _section(raw, _KEY_TILES)

    data = DataConfig(
        sites_dir=Path(data_raw["sites_dir"]),
        labels_file=str(data_raw["labels_file"]),
        channels=_unique(tuple(str(c) for c in data_raw["channels"]), "data.channels"),
        normalize=str(data_raw["normalize"]),
    )
    if data.normalize not in SUPPORTED_NORMALIZE:
        raise ValueError(f"data.normalize '{data.normalize}' не из {SUPPORTED_NORMALIZE}")

    targets = TargetsConfig(
        classes=_unique(tuple(str(c) for c in targets_raw["classes"]), "targets.classes"),
        heatmap_sigma_px=float(targets_raw["heatmap_sigma_px"]),
    )
    tiles = TilesConfig(
        tile_px=int(tiles_raw["tile_px"]),
        overlap_px=int(tiles_raw["overlap_px"]),
        pos_fraction=float(tiles_raw["pos_fraction"]),
    )
    if not 0 <= tiles.overlap_px < tiles.tile_px:
        raise ValueError(f"tiles: нужно 0 <= overlap_px < tile_px, получено {tiles}")
    if not 0.0 < tiles.pos_fraction < 1.0:
        raise ValueError(f"tiles.pos_fraction вне (0, 1): {tiles.pos_fraction}")

    model = ModelConfig.from_dict(_optional_section(raw, _KEY_MODEL))
    div = input_divisor(model)
    if tiles.tile_px % div:
        raise ValueError(
            f"tiles.tile_px={tiles.tile_px} не делится на {div} (model.arch={model.arch})"
        )

    return PipelineConfig(
        data=data,
        targets=targets,
        tiles=tiles,
        model=model,
        loss=LossConfig.from_dict(_optional_section(raw, _KEY_LOSS)),
        stitch=StitchConfig.from_dict(_optional_section(raw, _KEY_STITCH)),
        post=PostConfig.from_dict(_optional_section(raw, _KEY_POST)),
        eval=EvalConfig.from_dict(_optional_section(raw, _KEY_EVAL)),
    )
