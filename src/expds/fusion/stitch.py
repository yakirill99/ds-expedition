"""Сшивка предсказаний тайлов в карту участка с весовым окном.

Каждый пиксель = взвешенное среднее всех покрывающих его тайлов.
Окно Ханна с полом eps: края тайла весят мало, но не ноль, поэтому пиксели у края
участка, покрытые одним тайлом, получают его значение без деления на ноль.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from expds.tiles.index import Window

WINDOW_KINDS = ("hann", "flat")


@dataclass(frozen=True)
class StitchConfig:
    """Параметры сшивки (секция `stitch` YAML)."""

    window: str = "hann"
    eps: float = 1e-3

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> StitchConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"stitch: unknown keys {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.window not in WINDOW_KINDS:
            raise ValueError(f"stitch.window must be one of {WINDOW_KINDS}, got {cfg.window!r}")
        if not 0.0 < cfg.eps <= 1.0:
            raise ValueError("stitch.eps must be in (0, 1]")
        return cfg


def make_window(tile_px: int, kind: str = "hann", eps: float = 1e-3) -> np.ndarray:
    """Весовое окно (tile_px, tile_px) float32, значения в [eps**2, 1].

    Пол eps накладывается на 1D-окна до внешнего произведения: так у края участка
    сохраняется соотношение весов соседних тайлов (пол на 2D-окне его обнуляет, шов виден).
    """
    if tile_px <= 0:
        raise ValueError("tile_px must be > 0")
    if kind == "flat":
        return np.ones((tile_px, tile_px), dtype=np.float32)
    if kind != "hann":
        raise ValueError(f"unknown window kind {kind!r}")
    n = np.arange(tile_px, dtype=np.float64)
    w1 = np.maximum(np.sin(np.pi * (n + 0.5) / tile_px) ** 2, eps)
    return np.outer(w1, w1).astype(np.float32)


def _to_numpy(a: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().float().cpu().numpy()
    return np.asarray(a, dtype=np.float32)


def _as_window(w: Window | Sequence[int] | np.ndarray | torch.Tensor) -> Window:
    if isinstance(w, Window):
        return w
    if isinstance(w, (torch.Tensor, np.ndarray)):
        w = w.tolist()
    vals = [int(v) for v in w]
    if len(vals) != 4:
        raise ValueError(f"window must be (row0, col0, height, width), got {vals}")
    return Window(row0=vals[0], col0=vals[1], height=vals[2], width=vals[3])


class Stitcher:
    """Аккумулятор (K, H, W): add/add_batch по тайлам, result() в конце."""

    def __init__(
        self,
        num_channels: int,
        height: int,
        width: int,
        tile_px: int,
        cfg: StitchConfig | None = None,
    ):
        cfg = cfg or StitchConfig()
        self.k, self.h, self.w, self.tile_px = num_channels, height, width, tile_px
        self.window = make_window(tile_px, cfg.window, cfg.eps)
        self._acc = np.zeros((num_channels, height, width), dtype=np.float32)
        self._wsum = np.zeros((height, width), dtype=np.float32)

    def add(self, pred: np.ndarray | torch.Tensor, window: Any) -> None:
        """pred (K, >=h, >=w) — тайл, возможно с паддингом справа/снизу; window — его окно."""
        win = _as_window(window)
        p = _to_numpy(pred)
        if p.ndim != 3 or p.shape[0] != self.k:
            raise ValueError(f"pred must be ({self.k}, H, W), got {p.shape}")
        h, w = win.height, win.width
        if h > min(p.shape[1], self.tile_px) or w > min(p.shape[2], self.tile_px):
            raise ValueError(f"window {win} larger than pred {p.shape} or tile {self.tile_px}")
        if win.row0 < 0 or win.col0 < 0 or win.row0 + h > self.h or win.col0 + w > self.w:
            raise ValueError(f"window {win} outside grid ({self.h}, {self.w})")
        wt = self.window[:h, :w]
        rs = slice(win.row0, win.row0 + h)
        cs = slice(win.col0, win.col0 + w)
        self._acc[:, rs, cs] += p[:, :h, :w] * wt
        self._wsum[rs, cs] += wt

    def add_batch(self, preds: np.ndarray | torch.Tensor, windows: Any) -> None:
        """preds (B, K, T, T); windows — тензор (B, 4) или последовательность из B окон."""
        p = _to_numpy(preds)
        rows = list(windows)
        if len(rows) != len(p):
            raise ValueError(f"{len(p)} preds vs {len(rows)} windows")
        for pi, wi in zip(p, rows, strict=True):
            self.add(pi, wi)

    @property
    def coverage(self) -> np.ndarray:
        """(H, W) bool: пиксель покрыт хотя бы одним тайлом."""
        return self._wsum > 0

    def result(self, fill: float = 0.0) -> np.ndarray:
        """(K, H, W) float32; непокрытые пиксели = fill."""
        out = np.full_like(self._acc, fill)
        m = self.coverage
        out[:, m] = self._acc[:, m] / self._wsum[m]
        return out
