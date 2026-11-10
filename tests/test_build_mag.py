"""Регрессии build_mag: какие файлы 07_ становятся каналами mag_*."""

from pathlib import Path

from expds.data.build_mag import _mag_sources


def test_regression_mag_skips_colour_map(tmp_path: Path) -> None:
    """Из пары серый/цвет в 07_ берётся только серая карта."""
    mag = tmp_path / "07_site_MagGr_карты"
    mag.mkdir()
    (mag / "01_site_MagGr_серый.tif").write_bytes(b"")
    (mag / "02_site_MagGr_цвет.tif").write_bytes(b"")
    assert [p.name for p in _mag_sources(tmp_path)] == ["01_site_MagGr_серый.tif"]


def test_build_mag_writes_canonical_channel(tmp_path: Path) -> None:
    """Сборщик пишет и mag_<field>, и канонический mag.tif из конфигов night*.

    Контракт data.channels — ровно "mag" (как в make_synthetic); без
    mag.tif канал магнитки на реальных участках всегда нулевой.
    """
    import json as _json
    from dataclasses import asdict

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from expds.data.build_mag import build_mag_layers
    from expds.data.raster import read_raster
    from expds.features.grid import Grid

    grid = Grid(
        crs="EPSG:32636",
        pixel_size=1.0,
        x_min=500_000.0,
        y_min=6_000_000.0,
        width=8,
        height=8,
    )
    meta = tmp_path / "samples" / "sample_meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(_json.dumps({"grid": asdict(grid)}), encoding="utf-8")

    mag = tmp_path / "07_site_MagGr"
    mag.mkdir()
    with rasterio.open(
        mag / "01_site_MagGr_серый.tif",
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=from_origin(grid.x_min, grid.y_min + 8, 1.0, 1.0),
    ) as dst:
        dst.write(np.full((8, 8), 5.0, np.float32), 1)

    built = build_mag_layers(tmp_path)
    assert "mag_tmi" in built
    assert "mag" in built
    assert (tmp_path / "layers" / "mag.tif").is_file()
    layer = read_raster(tmp_path / "layers" / "mag.tif")
    assert float(layer.data[0, 0]) == 5.0
