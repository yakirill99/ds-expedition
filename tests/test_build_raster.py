"""D4.3: каноническая сетка, ориентация, мозаика, панхром, нормировка rgb."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from expds.data.build_raster import (
    build_site_rasters,
    canonical_grid,
    composite_rgb,
    grid_from_sources,
    read_to_grid,
)
from expds.features.grid import Grid, Layer
from expds.tiles.dataset import normalize_rgb_group

_MERC = "EPSG:3857"


def _write_src(path: Path, arr: np.ndarray, crs: str, transform, nodata=None) -> None:
    """Внешний north-up tif, как у организаторов."""
    bands = arr.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[2],
        height=arr.shape[1],
        count=bands,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype("float32"))


def _labels(tmp_path: Path, x: float, y: float, size: float = 10.0) -> Path:
    """Разметка из одного квадрата в EPSG:3857."""
    ring = [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
    feats = [
        {
            "type": "Feature",
            "properties": {"class": "kurgany_tselye"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
    ]
    path = tmp_path / "labels.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return path


def test_canonical_grid_uses_utm_and_meters(tmp_path: Path) -> None:
    """Сетка в UTM, пиксель 1 м, bbox разметки + отступ."""
    _labels(tmp_path, 3_600_000.0, 6_000_000.0)
    g = canonical_grid(tmp_path / "labels.geojson", _MERC, pad_px=8)
    assert g.crs.startswith("EPSG:326"), g.crs  # северное полушарие
    assert g.pixel_size == 1.0
    assert 10 < g.width < 60 and 10 < g.height < 60  # 10 м разметки + 2*8 м отступа


def test_read_to_grid_flips_north_up(tmp_path: Path) -> None:
    """Северная строка файла -> последняя строка Grid."""
    grid = Grid(crs="EPSG:32636", pixel_size=1.0, x_min=0.0, y_min=0.0, width=4, height=4)
    arr = np.zeros((1, 4, 4), dtype=np.float32)
    arr[0, 0, :] = 1.0  # строка 0 файла = север
    src = tmp_path / "north.tif"
    _write_src(src, arr, "EPSG:32636", from_origin(0.0, 4.0, 1.0, 1.0))

    out, cover = read_to_grid(src, grid)
    assert cover.all()
    assert out[0, -1, :].tolist() == [1.0] * 4  # юг = последняя строка Grid
    assert out[0, 0, :].tolist() == [0.0] * 4


def test_composite_finer_wins(tmp_path: Path) -> None:
    """Более детальный источник перекрывает грубый там, где покрывает."""
    (tmp_path / "04_x_SpOr_спутник").mkdir()
    (tmp_path / "05_x_Or_ортофото").mkdir()
    grid = Grid(crs="EPSG:32636", pixel_size=1.0, x_min=0.0, y_min=0.0, width=8, height=8)

    coarse = np.full((3, 4, 4), 10.0, dtype=np.float32)
    fine = np.full((3, 4, 4), 200.0, dtype=np.float32)
    _write_src(
        tmp_path / "04_x_SpOr_спутник" / "a.tif",
        coarse,
        "EPSG:32636",
        from_origin(0.0, 8.0, 2.0, 2.0),
    )
    _write_src(
        tmp_path / "05_x_Or_ортофото" / "b.tif", fine, "EPSG:32636", from_origin(0.0, 4.0, 1.0, 1.0)
    )

    arr, coverage, _ = composite_rgb(tmp_path, grid)
    assert coverage > 0.0
    assert float(arr[0, 0, 0]) == pytest.approx(200.0)  # юго-запад: мелкий поверх
    assert float(arr[0, -1, -1]) == pytest.approx(10.0)  # северо-восток: только грубый


def test_panchromatic_replicates_to_three(tmp_path: Path) -> None:
    """Однополосный источник (012_Сож) дублируется в rgb_r/g/b."""
    (tmp_path / "03_x_Ae_немецкая").mkdir()
    grid = Grid(crs="EPSG:32636", pixel_size=1.0, x_min=0.0, y_min=0.0, width=4, height=4)
    _write_src(
        tmp_path / "03_x_Ae_немецкая" / "p.tif",
        np.full((1, 4, 4), 77.0, dtype=np.float32),
        "EPSG:32636",
        from_origin(0.0, 4.0, 1.0, 1.0),
    )

    arr, _, _ = composite_rgb(tmp_path, grid)
    assert arr.shape == (3, 4, 4)
    assert np.allclose(arr, 77.0)


def test_build_writes_three_layers_and_meta(tmp_path: Path) -> None:
    """build_site_rasters пишет rgb_*.tif + samples/sample_meta.json."""
    _labels(tmp_path, 3_600_000.0, 6_000_000.0, size=20.0)
    (tmp_path / "04_x_SpOr_спутник").mkdir()
    # Пишем источник уже в UTM той зоны, что выберет canonical_grid.
    g = canonical_grid(tmp_path / "labels.geojson", _MERC, pad_px=8)
    _write_src(
        tmp_path / "04_x_SpOr_спутник" / "s.tif",
        np.full((4, 8, 8), 120.0, dtype=np.float32),
        g.crs,
        from_origin(g.x_min, g.y_min + g.height, g.pixel_size, g.pixel_size),
    )

    res = build_site_rasters(tmp_path, _MERC, pad_px=8)
    assert res["built"], res
    for name in ("rgb_r", "rgb_g", "rgb_b"):
        assert (tmp_path / "layers" / f"{name}.tif").exists()
    meta = json.loads((tmp_path / "samples" / "sample_meta.json").read_text(encoding="utf-8"))
    assert meta["grid"]["pixel_size"] == 1.0


def test_build_skips_oversized_site(tmp_path: Path) -> None:
    """Предохранитель: сетка больше max_pixels не собирается (012, 058)."""
    _labels(tmp_path, 3_600_000.0, 6_000_000.0, size=5000.0)
    res = build_site_rasters(tmp_path, _MERC, max_pixels=1000)
    assert not res["built"] and "Мпкс" in res["reason"]


def test_normalize_rgb_group_shares_scale() -> None:
    """Один масштаб на все полосы: отношение размахов сохраняется.

    Поканальная нормировка растянула бы каждую полосу на свой p2..p98 —
    узкий синий стал бы таким же контрастным, как красный. Совместная
    делит все три на общий (p98 - p2), поэтому 10-кратная разница
    в размахе остаётся 10-кратной.
    """
    grid = Grid(crs="EPSG:32636", pixel_size=1.0, x_min=0.0, y_min=0.0, width=4, height=4)

    def _ramp(lo: float, hi: float) -> np.ndarray:
        row = np.linspace(lo, hi, grid.width, dtype=np.float32)
        return np.tile(row, (grid.height, 1))

    layers = [
        Layer(name="rgb_r", data=_ramp(0.0, 200.0), grid=grid),
        Layer(name="rgb_g", data=_ramp(0.0, 200.0), grid=grid),
        Layer(name="rgb_b", data=_ramp(90.0, 110.0), grid=grid),  # в 10 раз уже
    ]
    out = {o.name: o for o in normalize_rgb_group(layers)}
    spread_r = float(out["rgb_r"].data.max() - out["rgb_r"].data.min())
    spread_b = float(out["rgb_b"].data.max() - out["rgb_b"].data.min())

    assert spread_b > 0.0
    assert spread_r == pytest.approx(10.0 * spread_b, rel=1e-4)


def test_build_keeps_existing_grid_from_lidar(tmp_path: Path) -> None:
    """Участок с готовыми слоями сохраняет свою сетку, meta не перезаписывается.

    Сетка лидарного участка — контракт для 27 рельефных слоёв. Пересчёт
    по разметке заставил бы read_raster ресемплить их на чужую сетку,
    а таргеты — считаться на других пикселях. Ошибка тихая: CRS совпадает,
    extent нет, предупреждение тонет в логе.
    """
    _labels(tmp_path, 3_600_000.0, 6_000_000.0, size=10.0)

    lidar = Grid(
        crs="EPSG:32636",
        pixel_size=1.0,
        x_min=500_000.0,
        y_min=6_000_000.0,
        width=8,
        height=8,
    )
    meta = tmp_path / "samples" / "sample_meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"grid": asdict(lidar)}), encoding="utf-8")
    before = meta.read_text(encoding="utf-8")

    # Источник покрывает лидарную сетку целиком.
    (tmp_path / "04_x_SpOr_спутник").mkdir()
    _write_src(
        tmp_path / "04_x_SpOr_спутник" / "s.tif",
        np.full((3, 8, 8), 90.0, dtype=np.float32),
        "EPSG:32636",
        from_origin(lidar.x_min, lidar.y_min + lidar.height, 1.0, 1.0),
    )

    res = build_site_rasters(tmp_path, _MERC, pad_px=8)
    assert res["built"], res
    g = res["grid"]
    assert (g.crs, g.x_min, g.y_min, g.width, g.height) == (
        lidar.crs,
        lidar.x_min,
        lidar.y_min,
        lidar.width,
        lidar.height,
    )
    assert meta.read_text(encoding="utf-8") == before, "мета лидарного участка перезаписана"


def test_build_adopts_lidar_dtm_grid(tmp_path: Path) -> None:
    """Есть dtm.tif, нет sample_meta.json — сетка берётся из dtm.

    У 001, 035, 105 слои собраны из LAS, но мета отсутствует. Сетку
    задаёт рельеф: посчитать её по разметке значит заставить read_raster
    ресемплить 27 слоёв на чужие пиксели.
    """
    _labels(tmp_path, 3_600_000.0, 6_000_000.0, size=10.0)

    lidar = Grid(
        crs="EPSG:32636", pixel_size=1.0, x_min=500_000.0, y_min=6_000_000.0, width=8, height=8
    )
    (tmp_path / "layers").mkdir()
    transform = from_origin(lidar.x_min, lidar.y_min + lidar.height, 1.0, 1.0)
    _write_src(
        tmp_path / "layers" / "dtm.tif",
        np.zeros((1, 8, 8), dtype=np.float32),
        "EPSG:32636",
        transform,
    )
    (tmp_path / "04_x_SpOr_спутник").mkdir()
    _write_src(
        tmp_path / "04_x_SpOr_спутник" / "s.tif",
        np.full((3, 8, 8), 90.0, dtype=np.float32),
        "EPSG:32636",
        transform,
    )

    res = build_site_rasters(tmp_path, _MERC, pad_px=8)
    assert res["built"], res
    g = res["grid"]
    assert (g.x_min, g.y_min, g.width, g.height) == (
        lidar.x_min,
        lidar.y_min,
        lidar.width,
        lidar.height,
    )
    meta = json.loads((tmp_path / "samples" / "sample_meta.json").read_text(encoding="utf-8"))
    assert meta["grid"]["width"] == lidar.width


def test_build_repairs_meta_mismatched_with_dtm(tmp_path: Path) -> None:
    """Мета расходится с dtm.tif — побеждает dtm, мета перезаписывается.

    Именно это случилось на 001, 035, 105: rgb_* собрались по сетке из
    разметки, мета её зафиксировала, и следующая загрузка получила
    ресемплинг вместо лидарных пикселей.
    """
    _labels(tmp_path, 3_600_000.0, 6_000_000.0, size=10.0)

    lidar = Grid(
        crs="EPSG:32636", pixel_size=1.0, x_min=500_000.0, y_min=6_000_000.0, width=8, height=8
    )
    wrong = Grid(
        crs="EPSG:32636", pixel_size=1.0, x_min=500_000.0, y_min=6_000_000.0, width=4, height=4
    )
    (tmp_path / "layers").mkdir()
    _write_src(
        tmp_path / "layers" / "dtm.tif",
        np.zeros((1, 8, 8), dtype=np.float32),
        "EPSG:32636",
        from_origin(lidar.x_min, lidar.y_min + lidar.height, 1.0, 1.0),
    )
    meta = tmp_path / "samples" / "sample_meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"grid": asdict(wrong)}), encoding="utf-8")

    (tmp_path / "04_x_SpOr_спутник").mkdir()
    _write_src(
        tmp_path / "04_x_SpOr_спутник" / "s.tif",
        np.full((3, 8, 8), 90.0, dtype=np.float32),
        "EPSG:32636",
        from_origin(lidar.x_min, lidar.y_min + lidar.height, 1.0, 1.0),
    )

    res = build_site_rasters(tmp_path, _MERC, pad_px=8)
    assert res["built"], res
    assert res["grid"].width == 8, "сетка взята из меты, а не из dtm"
    assert json.loads(meta.read_text(encoding="utf-8"))["grid"]["width"] == 8


def test_grid_from_sources_without_labels(tmp_path: Path) -> None:
    """Без разметки сетка — объединение bbox источников, а не падение.

    Путь публичной валидации: labels.geojson там нет по определению.
    """
    (tmp_path / "04_x_SpOr_спутник").mkdir()
    for dx in (0, 10):
        _write_src(
            tmp_path / "04_x_SpOr_спутник" / f"{dx}.tif",
            np.zeros((3, 4, 4), dtype=np.float32),
            "EPSG:32636",
            from_origin(500_000.0 + dx, 6_000_000.0, 1.0, 1.0),
        )

    g = grid_from_sources(tmp_path, pad_px=2)
    assert g.crs == "EPSG:32636"
    # тайлы X 500000..500004 и 500010..500014, Y 5999996..6000000
    # объединение + отступ 2 м: X 499998..500016, Y 5999994..6000002
    assert (g.x_min, g.y_min) == (499_998.0, 5_999_994.0)
    assert (g.width, g.height) == (18, 8)


def test_build_without_labels_uses_sources(tmp_path: Path) -> None:
    """Инференс: разметки нет — слои собираются, мета пишется."""
    (tmp_path / "04_x_SpOr_спутник").mkdir()
    _write_src(
        tmp_path / "04_x_SpOr_спутник" / "s.tif",
        np.full((3, 8, 8), 55.0, dtype=np.float32),
        "EPSG:32636",
        from_origin(500_000.0, 6_000_000.0, 1.0, 1.0),
    )

    res = build_site_rasters(tmp_path, _MERC, pad_px=2)
    assert res["built"], res
    assert res["grid"].crs == "EPSG:32636"
    for name in ("rgb_r", "rgb_g", "rgb_b"):
        assert (tmp_path / "layers" / f"{name}.tif").exists()
    meta = json.loads((tmp_path / "samples" / "sample_meta.json").read_text(encoding="utf-8"))
    assert meta["grid"]["crs"] == "EPSG:32636"
