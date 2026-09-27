"""Разбиение Grid на тайлы с перекрытием.

Окна — в пикселях Grid (row 0 = юг). Последнее окно по каждой оси прижимается
к краю, поэтому объединение окон = вся сетка без паддинга. Если сетка по оси
меньше тайла, окно по этой оси = вся сетка (паддинг до tile_px делает датасет).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from expds.features.grid import Grid


@dataclass(frozen=True)
class Window:
    """Окно в пикселях Grid: [row0, row0 + height) x [col0, col0 + width)."""

    row0: int
    col0: int
    height: int
    width: int

    @property
    def slices(self) -> tuple[slice, slice]:
        """Срезы (rows, cols) для массива (..., H, W)."""
        return slice(self.row0, self.row0 + self.height), slice(self.col0, self.col0 + self.width)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """(row0, col0, height, width)."""
        return self.row0, self.col0, self.height, self.width


def axis_starts(size: int, tile: int, stride: int) -> list[int]:
    """Начала окон по оси: шаг stride, последнее прижато к краю.

    Args:
        size: длина оси, пиксели.
        tile: размер тайла.
        stride: шаг (tile - overlap).

    Returns:
        Отсортированные уникальные начала.
    """
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile, stride))
    starts.append(size - tile)
    return starts


class TileIndex:
    """Окна tile_px x tile_px с перекрытием overlap_px, покрывающие всю Grid."""

    def __init__(self, grid: Grid, tile_px: int, overlap_px: int) -> None:
        """Строит окна.

        Args:
            grid: сетка участка.
            tile_px: размер тайла, пиксели.
            overlap_px: перекрытие соседних тайлов, пиксели (0 <= overlap < tile).
        """
        if tile_px <= 0:
            raise ValueError(f"tile_px должен быть > 0, получено {tile_px}")
        if not 0 <= overlap_px < tile_px:
            raise ValueError(f"Нужно 0 <= overlap_px < tile_px, получено {overlap_px}, {tile_px}")

        self._grid = grid
        self._tile_px = tile_px
        self._overlap_px = overlap_px
        stride = tile_px - overlap_px
        height = min(tile_px, grid.height)
        width = min(tile_px, grid.width)
        rows = axis_starts(grid.height, tile_px, stride)
        cols = axis_starts(grid.width, tile_px, stride)
        self._windows = tuple(Window(r, c, height, width) for r in rows for c in cols)

    @property
    def grid(self) -> Grid:
        """Сетка участка."""
        return self._grid

    @property
    def tile_px(self) -> int:
        """Размер тайла."""
        return self._tile_px

    @property
    def overlap_px(self) -> int:
        """Перекрытие."""
        return self._overlap_px

    def __len__(self) -> int:
        """Число окон."""
        return len(self._windows)

    def __iter__(self) -> Iterator[Window]:
        """Окна по строкам (с юга на север), внутри — по столбцам."""
        return iter(self._windows)

    def __getitem__(self, idx: int) -> Window:
        """Окно по индексу."""
        return self._windows[idx]
