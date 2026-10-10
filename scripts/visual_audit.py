"""Визуальный аудит препроцессинга.

Читает слои из data/cache/<site>/layers/, читает разметку из указанной
папки, строит картинки: DTM + контуры, SLRM + точки, hillshade + контуры.

Использование:
    uv run python scripts/visual_audit.py \
        --site 013_Nora_Vinnon \
        --labels "/path/to/разметка"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from expds.data.raster import read_raster


def read_labels_geojson(
    path: Path,
    target_crs: str,
) -> list[tuple[float, float, str]]:
    """Читает GeoJSON, возвращает список (x, y, name) в target_crs.

    Разметка в EPSG:3857, слои — в internal_crs (UTM). Перепроецируем
    через pyproj.

    Args:
        path: путь к GeoJSON.
        target_crs: CRS слоёв (например, "EPSG:25833").

    Returns:
        Список (x, y, name).
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:3857", target_crs, always_xy=True)

    with path.open() as f:
        data = json.load(f)

    points = []
    for feat in data["features"]:
        geom = feat.get("geometry")
        if geom is None:
            continue
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            x_3857 = sum(xs) / len(xs)
            y_3857 = sum(ys) / len(ys)
        elif geom["type"] == "Point":
            x_3857 = geom["coordinates"][0]
            y_3857 = geom["coordinates"][1]
        else:
            continue

        x_target, y_target = transformer.transform(x_3857, y_3857)
        points.append((x_target, y_target, path.stem))

    return points


def plot_layer_with_labels(
    layer_path: Path,
    labels: list[tuple[float, float, str]],
    output_path: Path,
    cmap: str = "terrain",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Строит слой + точки разметки, сохраняет PNG."""
    layer = read_raster(layer_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(
        layer.data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=layer.grid.bounds[:2] + layer.grid.bounds[2:],
        origin="lower",
    )
    # Точки разметки — в 3857, нужно перепроецировать в internal.
    # Пока строим как есть, для отладки.
    if labels:
        xs = [p[0] for p in labels]
        ys = [p[1] for p in labels]
        ax.scatter(xs, ys, c="red", s=20, alpha=0.7, label="labels (3857)")
        ax.legend()
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(layer_path.stem)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Сохранено: {output_path}")


def zoom_one_mound(
    site_dir: Path,
    audit_dir: Path,
    labels: list[tuple[float, float, str]],
    grid,
    index: int = 0,
    half_m: float = 25.0,
) -> None:
    """Строит кроп вокруг одного кургана: DTM и SLRM.

    Args:
        site_dir: папка layers/.
        audit_dir: папка audit/.
        labels: список точек разметки (x, y, name) в CRS слоёв.
        grid: Grid слоёв.
        index: индекс кургана в labels.
        half_m: половина размера кропа в метрах.
    """

    x0, y0, name = labels[index]

    slrm = read_raster(site_dir / "slrm_sigma_8.0.tif")
    dtm = read_raster(site_dir / "dtm.tif")

    x_min = x0 - half_m
    x_max = x0 + half_m
    y_min = y0 - half_m
    y_max = y0 + half_m

    col_min, row_min = grid.transform_to_pixel(x_min, y_min)
    col_max, row_max = grid.transform_to_pixel(x_max, y_max)
    col_min, col_max = int(col_min), int(col_max)
    row_min, row_max = int(row_min), int(row_max)

    slrm_crop = slrm.data[row_min:row_max, col_min:col_max]
    dtm_crop = dtm.data[row_min:row_max, col_min:col_max]

    # Маска валидных пикселей (исключаем nodata).
    dtm_valid = dtm_crop != dtm.nodata
    slrm_valid = slrm_crop != slrm.nodata
    # Исключаем также заведомо мусорные значения SLRM (|SLRM| > 10 м).
    slrm_valid = slrm_valid & (np.abs(slrm_crop) < 10.0)

    if dtm_valid.sum() > 0:
        dtm_lo, dtm_hi = np.percentile(dtm_crop[dtm_valid], [2, 98])
    else:
        dtm_lo, dtm_hi = 0.0, 1.0

    if slrm_valid.sum() > 0:
        slrm_lo, slrm_hi = np.percentile(slrm_crop[slrm_valid], [2, 98])
        slrm_abs = max(abs(slrm_lo), abs(slrm_hi))
    else:
        slrm_abs = 0.5
    if slrm_abs < 0.01:
        slrm_abs = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    dtm_masked = np.ma.masked_where(~dtm_valid, dtm_crop)
    axes[0].imshow(dtm_masked, cmap="terrain", vmin=dtm_lo, vmax=dtm_hi)
    axes[0].scatter(
        [x0 - x_min],
        [y0 - y_min],
        c="red",
        s=200,
        marker="x",
        linewidths=3,
    )
    axes[0].set_title(f"DTM [{index}], {name}\nZ={dtm_lo:.1f}..{dtm_hi:.1f} м")

    slrm_masked = np.ma.masked_where(~slrm_valid, slrm_crop)
    im = axes[1].imshow(slrm_masked, cmap="RdBu_r", vmin=-slrm_abs, vmax=slrm_abs)
    axes[1].scatter(
        [x0 - x_min],
        [y0 - y_min],
        c="red",
        s=200,
        marker="x",
        linewidths=3,
    )
    axes[1].set_title(
        f"SLRM sigma=8 [{index}], {name}\nmin={slrm_crop.min():.2f}, max={slrm_crop.max():.2f}"
    )
    plt.colorbar(im, ax=axes[1], fraction=0.03)

    plt.tight_layout()
    output_path = audit_dir / f"zoom_mound_{index:02d}.png"
    plt.savefig(output_path, dpi=100)
    plt.close()

    # Значения в центре кропа (где крестик).
    cy = y0 - y_min
    cx = x0 - x_min
    row_c = int(cy / grid.pixel_size)
    col_c = int(cx / grid.pixel_size)
    print(
        f"[{index}] {name} | DTM={dtm_crop[row_c, col_c]:.2f} м | "
        f"SLRM={slrm_crop[row_c, col_c]:.3f} | "
        f"DTM диапазон={dtm_crop.min():.2f}..{dtm_crop.max():.2f} | "
        f"SLRM диапазон={slrm_crop.min():.2f}..{slrm_crop.max():.2f}"
    )
    print(f"Сохранено: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    args = parser.parse_args()

    site_dir = args.cache_dir / args.site / "layers"
    audit_dir = args.cache_dir / args.site / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Читаем sample_meta.json, чтобы узнать CRS слоёв.
    meta_path = args.cache_dir / args.site / "samples" / "sample_meta.json"
    with meta_path.open() as f:
        meta = json.load(f)
    target_crs = meta["grid"]["crs"]
    print(f"CRS слоёв: {target_crs}")

    labels = []
    if args.labels and args.labels.exists():
        for gj in args.labels.rglob("*.geojson"):
            labels.extend(read_labels_geojson(gj, target_crs=target_crs))
        print(f"Разметка: {len(labels)} объектов (перепроецировано в {target_crs})")

    # DTM.
    plot_layer_with_labels(
        layer_path=site_dir / "dtm.tif",
        labels=labels,
        output_path=audit_dir / "dtm.png",
        cmap="terrain",
    )
    # SLRM sigma=8 — главный признак для курганов.
    plot_layer_with_labels(
        layer_path=site_dir / "slrm_sigma_8.0.tif",
        labels=labels,
        output_path=audit_dir / "slrm_sigma_8.png",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
    )
    # Hillshade.
    plot_layer_with_labels(
        layer_path=site_dir / "hillshade_az_315.tif",
        labels=labels,
        output_path=audit_dir / "hillshade.png",
        cmap="gray",
    )

    # Зум на все курганы.
    grid = read_raster(site_dir / "dtm.tif").grid
    for i in range(len(labels)):
        zoom_one_mound(
            site_dir=site_dir,
            audit_dir=audit_dir,
            labels=labels,
            grid=grid,
            index=i,
            half_m=25.0,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
