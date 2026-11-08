"""GPR-каналы: имена файлов, ориентация north-up -> Grid, попадание в stack_layers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from expds.data.build_gpr import build_gpr_layers, read_slice_to_grid
from expds.data.raster import read_raster
from expds.features.grid import DataSpec, Grid, stack_layers

_UTM = "EPSG:32636"


def _grid(width: int = 8, height: int = 8) -> Grid:
    return Grid(
        crs=_UTM,
        pixel_size=1.0,
        x_min=500_000.0,
        y_min=6_000_000.0,
        width=width,
        height=height,
    )


def _write_slice(path: Path, arr: np.ndarray, grid: Grid, nodata: float | None = None) -> None:
    """Сырой GPR-слайс: north-up, как приходит от георадара."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=from_origin(
            grid.x_min,
            grid.y_min + grid.height,
            grid.pixel_size,
            grid.pixel_size,
        ),
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_read_slice_flips_north_up(tmp_path: Path) -> None:
    """Северная строка файла -> последняя строка Grid."""
    grid = _grid(4, 4)
    arr = np.zeros((4, 4), dtype=np.float32)
    arr[0, :] = 7.0  # строка 0 файла = север
    src = tmp_path / "slice.tif"
    _write_slice(src, arr, grid)

    data, cover = read_slice_to_grid(src, grid)
    assert cover.all()
    assert data[-1, :].tolist() == [7.0] * 4  # юг = последняя строка Grid
    assert data[0, :].tolist() == [0.0] * 4


def test_build_gpr_names_from_depth(tmp_path: Path) -> None:
    """gpr_1.5m.tif -> gpr_slice_1_5.tif; gpr_2m.tif -> gpr_slice_2.tif."""
    grid = _grid()
    meta = tmp_path / "samples" / "sample_meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"grid": asdict(grid)}), encoding="utf-8")

    (tmp_path / "07_gpr").mkdir()
    _write_slice(tmp_path / "07_gpr" / "gpr_1.5m.tif", np.ones((8, 8), np.float32), grid)
    _write_slice(tmp_path / "07_gpr" / "gpr_2m.tif", np.ones((8, 8), np.float32) * 2, grid)

    built = build_gpr_layers(tmp_path)
    assert set(built) == {"gpr_slice_1_5", "gpr_slice_2"}
    assert (tmp_path / "layers" / "gpr_slice_1_5.tif").is_file()
    assert (tmp_path / "layers" / "gpr_slice_2.tif").is_file()


def test_build_gpr_idempotent_and_force(tmp_path: Path) -> None:
    """Повторный вызов без force не перезаписывает; force пересобирает."""
    grid = _grid()
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "gpr_slice_0_5.tif").write_bytes(b"")
    (tmp_path / "07_gpr").mkdir()
    _write_slice(tmp_path / "07_gpr" / "gpr_0.5m.tif", np.ones((8, 8), np.float32), grid)

    assert build_gpr_layers(tmp_path) != {}  # уже есть — вернёт существующий
    before = (tmp_path / "layers" / "gpr_slice_0_5.tif").stat().st_size
    assert before == 0
    build_gpr_layers(tmp_path, force=True)
    assert (tmp_path / "layers" / "gpr_slice_0_5.tif").stat().st_size > 0


def test_gpr_channels_reach_stack_layers(tmp_path: Path) -> None:
    """Собранный gpr_slice_1_5 попадает в Sample, если объявлен в channels."""
    grid = _grid()
    meta = tmp_path / "samples" / "sample_meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"grid": asdict(grid)}), encoding="utf-8")
    (tmp_path / "07_gpr").mkdir()
    _write_slice(tmp_path / "07_gpr" / "gpr_1.5m.tif", np.full((8, 8), 5.0, np.float32), grid)

    build_gpr_layers(tmp_path)

    layer = read_raster(tmp_path / "layers" / "gpr_slice_1_5.tif", grid=grid)
    spec = DataSpec(grid=grid, channel_names=("gpr_slice_1_5",))
    sample = stack_layers([layer], spec)
    assert sample.channel_mask.tolist() == [True]
    assert float(sample.tensor[0].mean()) == pytest.approx(5.0)


def test_build_gpr_without_sources_is_empty(tmp_path: Path) -> None:
    """Нет 07_/08_/09_ — пустой словарь, без исключения."""
    assert build_gpr_layers(tmp_path) == {}
