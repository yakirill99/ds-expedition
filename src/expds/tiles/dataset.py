"""Участки и тайловый torch Dataset поверх слоёв роли 1 и таргетов.

Участок (site_dir) — формат выхода make_synthetic / build_layers:
    layers/<channel>.tif, samples/sample_meta.json (Grid), labels.geojson (опц.).

Каналы собирает stack_layers роли 1 по DataSpec: отсутствующий слой -> нули и
channel_mask = False. Попиксельная валидность — из dtm.tif (до стака, т.к.
stack_layers её теряет). Массивы в конвенции Grid (row 0 = юг).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from expds.data.raster import read_raster
from expds.features.grid import DataSpec, Grid, Layer, stack_layers
from expds.io.geojson import read_labels
from expds.targets.rasterize import labels_to_targets
from expds.tiles.index import TileIndex, Window

logger = logging.getLogger(__name__)

NORMALIZE_NONE = "none"
NORMALIZE_SITE_ROBUST = "site_robust"
SUPPORTED_NORMALIZE = (NORMALIZE_NONE, NORMALIZE_SITE_ROBUST)

_LAYERS_DIR = "layers"
_META_PATH = Path("samples") / "sample_meta.json"
_VALID_LAYER = "dtm"
_ROBUST_PERCENTILES = (2.0, 50.0, 98.0)
_SCALE_EPS = 1e-6
_POS_HEATMAP_THRESHOLD = 0.5


@dataclass(frozen=True, eq=False)
class SiteData:
    """Участок в памяти.

    x: (C, H, W) float32; channel_mask: (C,) bool; valid: (H, W) bool;
    y_seg, y_hm: (K, H, W) float32 или None (нет разметки — инференс).
    """

    name: str
    grid: Grid
    x: np.ndarray
    channel_names: tuple[str, ...]
    channel_mask: np.ndarray
    valid: np.ndarray
    y_seg: np.ndarray | None = None
    y_hm: np.ndarray | None = None

    @property
    def has_labels(self) -> bool:
        """Есть ли таргеты."""
        return self.y_seg is not None and self.y_hm is not None


def _site_grid(site_dir: Path) -> Grid:
    """Grid участка: из sample_meta.json, иначе из первого tif в layers/."""
    meta_path = site_dir / _META_PATH
    if meta_path.exists():
        return Grid(**json.loads(meta_path.read_text(encoding="utf-8"))["grid"])
    tifs = sorted((site_dir / _LAYERS_DIR).glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"{site_dir}: нет ни {_META_PATH}, ни слоёв в {_LAYERS_DIR}/")
    return read_raster(tifs[0]).grid


def normalize_layer(layer: Layer) -> Layer:
    """Робастная нормализация по валидным пикселям: (x - median) / (p98 - p2).

    Nodata остаётся nodata (после stack_layers станет 0 = медиана).
    """
    valid = layer.valid_mask
    if not valid.any():
        return layer
    values = layer.data[valid]
    lo, med, hi = np.percentile(values, _ROBUST_PERCENTILES)
    scale = float(hi - lo)
    if scale < _SCALE_EPS:
        scale = 1.0
    data = layer.data.copy()
    data[valid] = (values - med) / scale
    return Layer(
        name=layer.name,
        data=data.astype(np.float32, copy=False),
        grid=layer.grid,
        nodata=layer.nodata,
        source=layer.source,
        license=layer.license,
        unit="normalized",
    )


def load_site(
    site_dir: Path,
    channels: Sequence[str],
    classes: Sequence[str],
    heatmap_sigma_px: float,
    labels_crs: str,
    labels_file: str = "labels.geojson",
    normalize: str = NORMALIZE_SITE_ROBUST,
) -> SiteData:
    """Загружает участок: каналы по списку, валидность, таргеты (если есть разметка).

    Args:
        site_dir: папка участка.
        channels: порядок каналов модели.
        classes: порядок классов таргетов.
        heatmap_sigma_px: сигма heatmap точек.
        labels_crs: CRS файла разметки (crs.output).
        labels_file: имя файла разметки в site_dir.
        normalize: none | site_robust.

    Returns:
        SiteData.
    """
    if normalize not in SUPPORTED_NORMALIZE:
        raise ValueError(f"normalize '{normalize}' не поддерживается: {SUPPORTED_NORMALIZE}")
    site_dir = Path(site_dir)
    grid = _site_grid(site_dir)

    layers: list[Layer] = []
    for name in channels:
        path = site_dir / _LAYERS_DIR / f"{name}.tif"
        if not path.exists():
            continue
        layer = read_raster(path, grid=grid)
        layers.append(normalize_layer(layer) if normalize == NORMALIZE_SITE_ROBUST else layer)

    valid_path = site_dir / _LAYERS_DIR / f"{_VALID_LAYER}.tif"
    if valid_path.exists():
        valid = read_raster(valid_path, grid=grid).valid_mask
    else:
        valid = np.ones(grid.shape, dtype=bool)

    sample = stack_layers(layers, DataSpec(grid=grid, channel_names=tuple(channels)))

    y_seg = y_hm = None
    labels_path = site_dir / labels_file
    if labels_path.exists():
        features = read_labels(labels_path, grid, src_crs=labels_crs)
        y_seg, y_hm = labels_to_targets(features, grid, classes, heatmap_sigma_px)

    logger.info(
        "Участок %s: C=%d (доступно %d), %dx%d, разметка=%s",
        site_dir.name,
        sample.n_channels,
        sample.n_available_channels,
        grid.height,
        grid.width,
        y_seg is not None,
    )
    return SiteData(
        name=site_dir.name,
        grid=grid,
        x=sample.tensor,
        channel_names=sample.channel_names,
        channel_mask=sample.channel_mask,
        valid=valid,
        y_seg=y_seg,
        y_hm=y_hm,
    )


class KozDataset(Dataset):
    """Тайлы по всем участкам. Элемент — dict тензоров.

    Ключи: x (C, T, T) float32, channel_mask (C,) bool, valid (T, T) bool,
    site (int), window (4,) int64 = (row0, col0, height, width);
    при разметке — y_seg, y_hm (K, T, T) float32. Тайлы меньше T дополняются нулями
    справа/сверху (valid = False в паддинге).
    """

    def __init__(self, sites: Sequence[SiteData], tile_px: int, overlap_px: int) -> None:
        """Строит индекс (участок, окно) по всем участкам."""
        if not sites:
            raise ValueError("KozDataset: пустой список участков")
        names = sites[0].channel_names
        for site in sites:
            if site.channel_names != names:
                raise ValueError(f"Участок {site.name}: каналы не совпадают с {sites[0].name}")
        labels = {site.has_labels for site in sites}
        if len(labels) != 1:
            raise ValueError("KozDataset: участки с разметкой и без неё смешивать нельзя")

        self._sites = tuple(sites)
        self._tile_px = tile_px
        self._has_labels = labels.pop()
        self._items: list[tuple[int, Window]] = [
            (i, window)
            for i, site in enumerate(self._sites)
            for window in TileIndex(site.grid, tile_px, overlap_px)
        ]
        self._positive = np.array(
            [self._is_positive(i, w) for i, w in self._items] if self._has_labels else [],
            dtype=bool,
        )

    def _is_positive(self, site_idx: int, window: Window) -> bool:
        """В окне есть пиксель маски или пик heatmap."""
        site = self._sites[site_idx]
        rows, cols = window.slices
        assert site.y_seg is not None and site.y_hm is not None
        return bool(
            site.y_seg[:, rows, cols].max() > 0
            or site.y_hm[:, rows, cols].max() >= _POS_HEATMAP_THRESHOLD
        )

    @property
    def sites(self) -> tuple[SiteData, ...]:
        """Участки."""
        return self._sites

    @property
    def positive(self) -> np.ndarray:
        """(N,) bool: позитивные тайлы (пусто без разметки)."""
        return self._positive

    def item_window(self, idx: int) -> tuple[int, Window]:
        """(индекс участка, окно) элемента — для сшивки."""
        return self._items[idx]

    def sample_weights(self, pos_fraction: float) -> np.ndarray:
        """Веса для WeightedRandomSampler: доля позитивных тайлов = pos_fraction.

        Если позитивных или негативных нет — равные веса.
        """
        if not 0.0 < pos_fraction < 1.0:
            raise ValueError(f"pos_fraction должна быть в (0, 1), получено {pos_fraction}")
        n = len(self._items)
        n_pos = int(self._positive.sum())
        if not self._has_labels or n_pos in (0, n):
            return np.full(n, 1.0 / n)
        return np.where(self._positive, pos_fraction / n_pos, (1.0 - pos_fraction) / (n - n_pos))

    def __len__(self) -> int:
        """Число тайлов."""
        return len(self._items)

    def _pad(self, arr: np.ndarray, window: Window, dtype: type) -> np.ndarray:
        """Вырез окна из (..., H, W) с паддингом до (..., T, T)."""
        rows, cols = window.slices
        out = np.zeros((*arr.shape[:-2], self._tile_px, self._tile_px), dtype=dtype)
        out[..., : window.height, : window.width] = arr[..., rows, cols]
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Тайл idx."""
        site_idx, window = self._items[idx]
        site = self._sites[site_idx]
        item: dict[str, torch.Tensor | int] = {
            "x": torch.from_numpy(self._pad(site.x, window, np.float32)),
            "channel_mask": torch.from_numpy(site.channel_mask.copy()),
            "valid": torch.from_numpy(self._pad(site.valid, window, bool)),
            "site": site_idx,
            "window": torch.tensor(window.as_tuple(), dtype=torch.int64),
        }
        if site.has_labels:
            assert site.y_seg is not None and site.y_hm is not None
            item["y_seg"] = torch.from_numpy(self._pad(site.y_seg, window, np.float32))
            item["y_hm"] = torch.from_numpy(self._pad(site.y_hm, window, np.float32))
        return item
