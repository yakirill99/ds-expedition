"""Тесты expds.tiles.index."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.grid import Grid
from expds.tiles.index import TileIndex, axis_starts


def _grid(width: int, height: int) -> Grid:
    return Grid(crs="EPSG:32637", pixel_size=1.0, x_min=0.0, y_min=0.0, width=width, height=height)


@pytest.mark.parametrize(
    ("width", "height", "tile", "overlap"), [(1024, 1024, 256, 32), (300, 250, 128, 16)]
)
def test_full_coverage_in_bounds(width: int, height: int, tile: int, overlap: int) -> None:
    grid = _grid(width, height)
    index = TileIndex(grid, tile, overlap)
    cover = np.zeros(grid.shape, dtype=np.int32)
    for w in index:
        assert (w.height, w.width) == (tile, tile)
        assert w.row0 >= 0 and w.col0 >= 0
        assert w.row0 + w.height <= height and w.col0 + w.width <= width
        rows, cols = w.slices
        cover[rows, cols] += 1
    assert cover.min() >= 1


def test_axis_starts_stride_and_last_at_edge() -> None:
    starts = axis_starts(1024, 256, 224)
    assert starts == [0, 224, 448, 672, 768]
    assert axis_starts(256, 256, 224) == [0]


def test_small_grid_single_window() -> None:
    index = TileIndex(_grid(100, 90), 128, 16)
    assert len(index) == 1
    assert index[0].as_tuple() == (0, 0, 90, 100)


@pytest.mark.parametrize(("tile", "overlap"), [(128, 128), (128, -1), (0, 0)])
def test_invalid_params_raise(tile: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        TileIndex(_grid(256, 256), tile, overlap)
