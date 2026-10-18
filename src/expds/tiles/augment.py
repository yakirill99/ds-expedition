"""Аугментации тайлов для обучения: flip, rot90, channel dropout.

Все геометрические преобразования применяются СИНХРОННО к x, y_seg, y_hm
и valid — иначе маски разъедутся с изображением и модель будет учиться
на битых таргетах.

Ориентация не важна: flip/rot90 — симметрии квадрата, они не ломают
соглашение "row 0 = юг". Восстановление геометрии при инференсе не требуется,
потому что аугментации живут только в train-цикле (см. AugmentedDataset).

Каналы НЕ переставляются: порядок фиксирован контрактом модели и сохраняется
в чекпоинт. Dropout лишь зануляет выбранный канал и снимает его в channel_mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

#: Сколько раз можно крутить на 90 градусов.
ROT90_CHOICES = (0, 1, 2, 3)


@dataclass(frozen=True)
class AugmentConfig:
    """Параметры аугментаций (секция `train.augment` в YAML).

    hflip_p / vflip_p / rot90_p — вероятности на тайл.
    channel_dropout_p — вероятность занулить один доступный канал.
    """

    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rot90_p: float = 0.5
    channel_dropout_p: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AugmentConfig:
        """Строит конфиг из словаря YAML с валидацией."""
        d = dict(d or {})
        known = {"hflip_p", "vflip_p", "rot90_p", "channel_dropout_p"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"train.augment: unknown keys {sorted(unknown)}")
        cfg = cls(**{k: float(v) for k, v in d.items()})
        for name in ("hflip_p", "vflip_p", "rot90_p", "channel_dropout_p"):
            p = getattr(cfg, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"train.augment.{name}={p} должен быть в [0, 1]")
        return cfg

    @property
    def enabled(self) -> bool:
        """Включена ли хоть одна аугментация."""
        return (
            self.hflip_p > 0.0
            or self.vflip_p > 0.0
            or self.rot90_p > 0.0
            or self.channel_dropout_p > 0.0
        )


def _flip_h(t: torch.Tensor) -> torch.Tensor:
    """Отражение по горизонтали для (..., H, W) и (C, H, W)."""
    return torch.flip(t, dims=[-1])


def _flip_v(t: torch.Tensor) -> torch.Tensor:
    """Отражение по вертикали для (..., H, W) и (C, H, W)."""
    return torch.flip(t, dims=[-2])


def _rot90(t: torch.Tensor, k: int) -> torch.Tensor:
    """Поворот на k*90 градусов против часовой стрелки в плоскости (H, W)."""
    if k % 4 == 0:
        return t
    return torch.rot90(t, k=k % 4, dims=[-2, -1])


def geometric(
    x: torch.Tensor,
    y_seg: torch.Tensor | None,
    y_hm: torch.Tensor | None,
    valid: torch.Tensor,
    *,
    hflip: bool = False,
    vflip: bool = False,
    rot90: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Применяет одну и ту же геометрию ко всем тайлам батча.

    Args:
        x: (B, C, T, T).
        y_seg: (B, K, T, T) или None (инференс без разметки).
        y_hm: (B, K, T, T) или None.
        valid: (B, T, T).
        hflip: отразить по горизонтали.
        vflip: отразить по вертикали.
        rot90: число поворотов на 90 градусов.

    Returns:
        Кортеж (x, y_seg, y_hm, valid) той же формы.
    """
    if hflip:
        x = _flip_h(x)
        valid = _flip_h(valid)
        y_seg = None if y_seg is None else _flip_h(y_seg)
        y_hm = None if y_hm is None else _flip_h(y_hm)
    if vflip:
        x = _flip_v(x)
        valid = _flip_v(valid)
        y_seg = None if y_seg is None else _flip_v(y_seg)
        y_hm = None if y_hm is None else _flip_v(y_hm)
    if rot90 % 4:
        x = _rot90(x, rot90)
        valid = _rot90(valid, rot90)
        y_seg = None if y_seg is None else _rot90(y_seg, rot90)
        y_hm = None if y_hm is None else _rot90(y_hm, rot90)
    return x, y_seg, y_hm, valid


def channel_dropout(
    x: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    p: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Зануляет один случайный доступный канал на весь батч.

    Затрагиваются только каналы, у которых channel_mask = True: выключенный
    канал и так нулевой, гасить его бессмысленно.

    Args:
        x: (B, C, T, T).
        channel_mask: (B, C) bool.
        p: вероятность применить dropout к батчу.
        generator: генератор для воспроизводимости.

    Returns:
        (x, channel_mask) — изменённые копии.
    """
    if p <= 0.0:
        return x, channel_mask

    if channel_mask.shape != (x.shape[0], x.shape[1]):
        raise ValueError(
            f"channel_dropout: channel_mask {tuple(channel_mask.shape)} "
            f"должен быть {(x.shape[0], x.shape[1])} как (B, C) от x"
        )

    b = x.shape[0]
    available = channel_mask.any(dim=0)  # (C,) — канал есть хотя бы у одного элемента
    idx = torch.nonzero(available, as_tuple=False).flatten()
    if idx.numel() == 0:
        return x, channel_mask

    x = x.clone()
    mask = channel_mask.clone()
    u = torch.rand(b, generator=generator)
    for i in range(b):
        if float(u[i]) >= p:
            continue
        pick = int(idx[torch.randint(idx.numel(), (1,), generator=generator).item()])
        x[i, pick] = 0.0
        mask[i, pick] = False
    return x, mask


class AugmentedDataset(Dataset):
    """Обёртка над KozDataset: применяет аугментации к элементу.

    Наследуется от Dataset и проксирует len / item_window / sample_weights,
    поэтому подставляется в DataLoader без изменений в train.fit.
    Валидация и инференс используют исходный датасет без аугментаций.
    """

    def __init__(
        self,
        base: Dataset,
        cfg: AugmentConfig,
        *,
        seed: int = 0,
    ) -> None:
        self._base = base
        self._cfg = cfg
        self._seed = seed
        self._epoch = 0

    # --- прокси -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._base)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    def set_epoch(self, epoch: int) -> None:
        """Меняет зерно аугментаций между эпохами."""
        self._epoch = int(epoch)

    # --- аугментация --------------------------------------------------------

    def _rng(self, idx: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self._seed + 1_000_003 * self._epoch + idx)
        return g

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        item = dict(self._base[idx])
        if not self._cfg.enabled:
            return item

        g = self._rng(idx)
        hflip = bool(torch.rand(1, generator=g).item() < self._cfg.hflip_p)
        vflip = bool(torch.rand(1, generator=g).item() < self._cfg.vflip_p)
        rot = 0
        if torch.rand(1, generator=g).item() < self._cfg.rot90_p:
            rot = int(torch.randint(len(ROT90_CHOICES), (1,), generator=g).item())

        x = item["x"].unsqueeze(0)
        valid = item["valid"].unsqueeze(0)
        y_seg = item["y_seg"].unsqueeze(0) if "y_seg" in item else None
        y_hm = item["y_hm"].unsqueeze(0) if "y_hm" in item else None

        x, y_seg, y_hm, valid = geometric(
            x, y_seg, y_hm, valid, hflip=hflip, vflip=vflip, rot90=rot
        )

        item["x"] = x.squeeze(0)
        item["valid"] = valid.squeeze(0)
        if y_seg is not None:
            item["y_seg"] = y_seg.squeeze(0)
        if y_hm is not None:
            item["y_hm"] = y_hm.squeeze(0)

        if self._cfg.channel_dropout_p > 0.0:
            x1 = item["x"].unsqueeze(0)
            m1 = item["channel_mask"].unsqueeze(0)
            x1, m1 = channel_dropout(x1, m1, p=self._cfg.channel_dropout_p, generator=g)
            item["x"] = x1.squeeze(0)
            item["channel_mask"] = m1.squeeze(0)

        return item


def random_crop_starts(
    height: int,
    width: int,
    tile_px: int,
    *,
    count: int,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Случайные старты кропов (row0, col0) вместо сетки TileIndex.

    Используется для train: сетка окон покрывает участок регулярно и даёт
    коррелированные тайлы. Случайные кропы перемешивают контекст.

    Args:
        height: высота участка.
        width: ширина участка.
        tile_px: сторона кропа.
        count: сколько кропов нужно.
        generator: numpy-генератор для воспроизводимости.

    Returns:
        (count, 2) int64 массив (row0, col0).
    """
    if tile_px <= 0:
        raise ValueError(f"tile_px должен быть > 0, получено {tile_px}")
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    rng = generator or np.random.default_rng(0)

    max_row = max(height - tile_px, 0)
    max_col = max(width - tile_px, 0)
    rows = rng.integers(0, max_row + 1, size=count, dtype=np.int64)
    cols = rng.integers(0, max_col + 1, size=count, dtype=np.int64)
    return np.stack([rows, cols], axis=1)


def warmup_hm_weight(epoch: int, hm_weight: float) -> float:
    """Вес hm-лосса с warmup: в первую эпоху 0, дальше линейно к норме.

    В 1-й эпохе heatmap-голова ещё не обучена, и её градиент только мешает
    seg-голове. Отключаем hm на epoch=1, к 3-й выходим на полный вес.

    Args:
        epoch: номер эпохи с 1.
        hm_weight: целевой вес из конфига.

    Returns:
        Вес hm-лосса для этой эпохи.
    """
    if epoch <= 1:
        return 0.0
    if epoch >= 3:
        return float(hm_weight)
    return float(hm_weight) * (epoch - 1) / 2.0


def cosine_warmup_lr(step: int, total: int, base_lr: float, warmup_steps: int) -> float:
    """LR с линейным warmup и косинусным спадом.

    Args:
        step: текущий шаг (0-based).
        total: всего шагов.
        base_lr: пиковый LR.
        warmup_steps: длина warmup в шагах.

    Returns:
        LR для этого шага.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(total - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
