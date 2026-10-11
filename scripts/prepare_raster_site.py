"""Готовит растровый участок: читает GeoTIFF, ресемплит в общую сетку,
сохраняет как layers/*.tif в формате build_layers.

Упрощённый build_site для растровых участков. Не обрабатывает лидар.

Использование:
    uv run python scripts/prepare_raster_site.py \
        --site data/raw/048_Kalmykia \
        --out data/cache/048_Kalmykia \
        --pixel-size 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from expds.data.raster import write_raster
from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)


def read_utm(site_dir: Path) -> str:
    """Читает UTM.json из папки участка."""
    utm_path = site_dir / "UTM.json"
    with utm_path.open() as f:
        data = json.load(f)
    return data["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")


def find_tifs(site_dir: Path) -> list[Path]:
    """Находит первый .tif в папке каждой модальности (кроме разметки).

    Берём только первый файл (главный). Дополнительные TIF с другим
    разрешением или другим участком пока не используем.
    """
    tifs = []
    for subdir in site_dir.iterdir():
        if not subdir.is_dir():
            continue
        if "разметка" in subdir.name.lower():
            continue
        sub_tifs = sorted(subdir.rglob("*.tif"))
        if sub_tifs:
            tifs.append(sub_tifs[0])
    return tifs


def build_target_grid(
    tif_path: Path,
    pixel_size: float,
    crs: str,
) -> Grid:
    """Строит целевую сетку по bounds растра."""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
    return Grid.from_bounds(
        crs=crs,
        pixel_size=pixel_size,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
    )


def resample_to_grid(
    tif_path: Path,
    grid: Grid,
    channel: int,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Ресемплит один канал растра в целевую сетку."""
    with rasterio.open(tif_path) as src:
        src_transform = src.transform
        src_crs = src.crs
        source = src.read(channel)
        src_nodata = src.nodata

    # Целевой transform.
    _, _, _, y_max = grid.bounds
    dst_transform = Affine(grid.pixel_size, 0, grid.x_min, 0, -grid.pixel_size, y_max)

    dst = np.zeros(grid.shape, dtype=np.float32)
    reproject(
        source=source.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=grid.crs,
        dst_nodata=0.0,
        resampling=resampling,
    )
    return dst


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--prefix", type=str, default="spor")
    args = parser.parse_args()

    site_dir = args.site
    out_dir = args.out
    layers_dir = out_dir / "layers"
    samples_dir = out_dir / "samples"
    layers_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    crs = read_utm(site_dir)
    print(f"CRS участка: {crs}")

    tifs = find_tifs(site_dir)
    print(f"Найдено TIF: {len(tifs)}")
    for t in tifs:
        print(f"  {t.relative_to(site_dir)}")

    if not tifs:
        print("ОШИБКА: нет TIF")
        return 1

    # Берём первый (главный) файл для сетки.
    main_tif = tifs[0]
    grid = build_target_grid(main_tif, args.pixel_size, crs)
    print(f"Целевая сетка: {grid.width}x{grid.height}, pixel_size={grid.pixel_size}")

    # Обрабатываем каждый TIF, каждый канал отдельно.
    channel_names = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            n_channels = src.count
            tif_prefix = tif.stem.replace("_SpOr_спутник", "").replace(" ", "_")
        # Берём только первые 3 канала (RGB). 4-й — бесполезен (все 255).
        n_channels = min(n_channels, 3)
        for ch in range(1, n_channels + 1):
            name = f"{args.prefix}_{tif_prefix}_ch{ch}"
            # Убираем кириллицу из имени файла.
            name = name.replace("_Калмыкия_1_", "_").replace("Калмыкия_1_", "")
            data = resample_to_grid(tif, grid=grid, channel=ch)
            layer = Layer(
                name=name,
                data=data,
                grid=grid,
                nodata=0.0,
                source=str(tif),
                license="unknown",
                unit="uint8",
            )
            write_raster(layer=layer, path=layers_dir / f"{name}.tif")
            channel_names.append(name)
            print(f"  Сохранён: {name}")

    # Сохраняем sample_meta.json.
    meta = {
        "channel_names": channel_names,
        "channel_mask": [True] * len(channel_names),
        "grid": {
            "crs": grid.crs,
            "pixel_size": grid.pixel_size,
            "x_min": grid.x_min,
            "y_min": grid.y_min,
            "width": grid.width,
            "height": grid.height,
        },
        "nodata": 0.0,
        "n_channels": len(channel_names),
        "n_available_channels": len(channel_names),
    }
    with (samples_dir / "sample_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nВсего каналов: {len(channel_names)}")
    print(f"Метаданные: {samples_dir / 'sample_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
