"""Регрессия: resample_layer в конвенции Grid (row 0 = юг).

Баг: north-up Affine для массивов в конвенции Grid без флипа давал
сдвиг на 2 * (y_center_dst - y_center_src) при разных extent.
"""

from __future__ import annotations

import numpy as np

from expds.features.align import resample_layer
from expds.features.grid import Grid, Layer

_CRS = "EPSG:32617"


def _row_index_layer(grid: Grid) -> Layer:
    """Значение пикселя = его row в конвенции Grid."""
    data = np.tile(np.arange(grid.height, dtype=np.float32)[:, None], (1, grid.width))
    return Layer(name="rows", data=data, grid=grid)


def test_resample_subgrid_shifted_north() -> None:
    """Сетка B = строки 2..5 сетки A -> значения 2..5 без сдвига."""
    grid_a = Grid(crs=_CRS, pixel_size=1.0, x_min=0.0, y_min=0.0, width=4, height=10)
    grid_b = Grid(crs=_CRS, pixel_size=1.0, x_min=0.0, y_min=2.0, width=4, height=4)

    out = resample_layer(_row_index_layer(grid_a), grid_b, method="nearest")

    expected = np.tile(np.arange(2, 6, dtype=np.float32)[:, None], (1, 4))
    assert np.array_equal(out.data, expected)


def test_resample_world_value_preserved() -> None:
    """Значение в мировой точке одинаково до и после ресэмплинга."""
    grid_a = Grid(crs=_CRS, pixel_size=1.0, x_min=0.0, y_min=0.0, width=8, height=12)
    grid_b = Grid(crs=_CRS, pixel_size=1.0, x_min=1.0, y_min=5.0, width=5, height=6)
    layer_a = _row_index_layer(grid_a)

    out = resample_layer(layer_a, grid_b, method="nearest")

    for row_b in range(grid_b.height):
        x, y = grid_b.transform_to_world(2, row_b)
        col_a, row_a = grid_a.transform_to_pixel(x, y)
        assert (
            out.data[row_b, 2] == layer_a.data[int(round(float(row_a))), int(round(float(col_a)))]
        )
