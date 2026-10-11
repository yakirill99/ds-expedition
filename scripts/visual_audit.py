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
        elif geom["type"] == "MultiPolygon":
            # Берём первый полигон из мультиполигона.
            coords = geom["coordinates"][0][0]
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


def read_labels_polygons(
    path: Path,
    target_crs: str,
) -> list[list[tuple[float, float]]]:
    """Читает GeoJSON, возвращает список полигонов (контуров).

    Каждый полигон — список (x, y) в target_crs. Polygon превращается
    в свой контур, Point — в маленький квадрат вокруг точки.

    Args:
        path: путь к GeoJSON.
        target_crs: CRS слоёв.

    Returns:
        Список полигонов: [[(x1, y1), (x2, y2), ...], ...].
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:3857", target_crs, always_xy=True)

    with path.open() as f:
        data = json.load(f)

    polygons: list[list[tuple[float, float]]] = []
    for feat in data["features"]:
        geom = feat.get("geometry")
        if geom is None:
            continue
        if geom["type"] == "Polygon":
            ring = geom["coordinates"][0]
            transformed = [transformer.transform(x, y) for x, y in ring]
            polygons.append(transformed)
        elif geom["type"] == "MultiPolygon":
            # Каждый вложенный полигон — отдельный контур.
            for poly in geom["coordinates"]:
                ring = poly[0]
                transformed = [transformer.transform(x, y) for x, y in ring]
                polygons.append(transformed)
        elif geom["type"] == "Point":
            x, y = transformer.transform(*geom["coordinates"])
            half = 2.0  # 2 м — размер квадрата для точечных объектов
            polygons.append(
                [
                    (x - half, y - half),
                    (x + half, y - half),
                    (x + half, y + half),
                    (x - half, y + half),
                    (x - half, y - half),
                ]
            )
    return polygons


def plot_polygons(
    ax,
    polygons: list[list[tuple[float, float]]],
    grid,
    edge_color: str = "red",
    fill_color: str = "red",
    alpha: float = 0.4,
    linewidth: float = 3.0,
) -> None:
    """Рисует контуры полигонов разметки на осях.

    Args:
        ax: оси matplotlib.
        polygons: список полигонов [(x, y), ...] в CRS слоёв.
        grid: Grid слоёв (для перевода в пиксели).
        edge_color: цвет контура.
        fill_color: цвет заливки.
        alpha: прозрачность заливки.
        linewidth: толщина контура.
    """
    from matplotlib.patches import Polygon as MplPolygon

    for ring in polygons:
        if len(ring) < 3:
            continue
        # Переводим мировые координаты в пиксели слоя.
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        cols, rows = grid.transform_to_pixel(np.array(xs), np.array(ys))
        # imshow с origin='lower': row 0 = низ = y_min. Grid: row 0 = юг.
        # Совпадает, инверсия не нужна.
        verts = list(zip(cols, rows, strict=True))

        patch = MplPolygon(
            verts,
            closed=True,
            edgecolor=edge_color,
            facecolor=fill_color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(patch)


def plot_layer_zoomed(
    layer_path: Path,
    polygons: list[list[tuple[float, float]]],
    output_path: Path,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    cmap: str = "gray",
) -> None:
    """Строит слой с зумом на заданную область (мировые координаты).

    Args:
        layer_path: путь к слою.
        polygons: полигоны разметки в CRS слоя.
        output_path: путь к PNG.
        x_range: (x_min, x_max) — область зума в мировых координатах.
        y_range: (y_min, y_max) — область зума.
        cmap: цветовая схема.
    """
    layer = read_raster(layer_path)

    # Область зума.
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Переводим в пиксельные индексы.
    col_min, row_min = layer.grid.transform_to_pixel(x_min, y_min)
    col_max, row_max = layer.grid.transform_to_pixel(x_max, y_max)
    col_min = max(0, int(np.floor(col_min)))
    row_min = max(0, int(np.floor(row_min)))
    col_max = min(layer.grid.width, int(np.ceil(col_max)))
    row_max = min(layer.grid.height, int(np.ceil(row_max)))

    crop = layer.data[row_min:row_max, col_min:col_max]
    if crop.size == 0:
        print(f"  Пропускаем {layer_path.stem}: пустой кроп")
        return

    # Локальные перцентили.
    valid = crop != layer.nodata
    if valid.any():
        lo, hi = np.percentile(crop[valid], [2, 98])
    else:
        lo, hi = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(
        crop,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
    )

    # Полигоны рисуем в мировых координатах через ax.plot.
    for ring in polygons:
        if len(ring) < 3:
            continue
        xs = [p[0] for p in ring] + [ring[0][0]]
        ys = [p[1] for p in ring] + [ring[0][1]]
        ax.plot(xs, ys, color="red", linewidth=2.0, alpha=0.8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{layer_path.stem} (zoom)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Сохранено: {output_path}")


def plot_layer_with_labels(
    layer_path: Path,
    labels: list[tuple[float, float, str]],
    polygons: list[list[tuple[float, float]]],
    output_path: Path,
    cmap: str = "terrain",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Строит слой + точки разметки, сохраняет PNG."""
    layer = read_raster(layer_path)
    fig, ax = plt.subplots(figsize=(12, 12))

    # Локальные шкалы — по перцентилям валидных пикселей.
    valid = layer.data != layer.nodata
    if valid.any():
        if vmin is None:
            vmin = float(np.percentile(layer.data[valid], 2))
        if vmax is None:
            vmax = float(np.percentile(layer.data[valid], 98))
    else:
        vmin = 0.0
        vmax = 1.0

    im = ax.imshow(
        layer.data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(
            layer.grid.x_min,
            layer.grid.x_max,
            layer.grid.y_min,
            layer.grid.y_max,
        ),
        origin="lower",
    )
    # Точки разметки — в 3857, нужно перепроецировать в internal.
    # Пока строим как есть, для отладки.
    if polygons:
        plot_polygons(ax, polygons, layer.grid)
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
    polygons = []
    if args.labels and args.labels.exists():
        for gj in args.labels.rglob("*.geojson"):
            labels.extend(read_labels_geojson(gj, target_crs=target_crs))
            polygons.extend(read_labels_polygons(gj, target_crs=target_crs))
        print(f"Разметка: {len(labels)} объектов (перепроецировано в {target_crs})")
        print(f"Полигонов: {len(polygons)}")

    # Все слои из layers/ — строим картинку для каждого.
    # Пропускаем огромные (больше 2000x2000) — они не поместятся.
    layer_files = sorted(site_dir.glob("*.tif"))
    print(f"Слоёв для аудита: {len(layer_files)}")

    for layer_path in layer_files:
        # Выбираем cmap по имени слоя.
        name = layer_path.stem
        if "slrm" in name or "tpi" in name or "curvature" in name:
            cmap = "RdBu_r"
        elif "hillshade" in name:
            cmap = "gray"
        elif "slope" in name:
            cmap = "viridis"
        elif "aspect" in name:
            cmap = "hsv"
        else:
            cmap = "gray"

        plot_layer_with_labels(
            layer_path=layer_path,
            labels=labels,
            polygons=polygons,
            output_path=audit_dir / f"{name}.png",
            cmap=cmap,
        )

    # Зум на область разметки — если есть полигоны.
    if polygons:
        all_x = [p[0] for poly in polygons for p in poly]
        all_y = [p[1] for poly in polygons for p in poly]
        # Запас 10%.
        span_x = max(all_x) - min(all_x)
        span_y = max(all_y) - min(all_y)
        x_range = (min(all_x) - 0.1 * span_x, max(all_x) + 0.1 * span_x)
        y_range = (min(all_y) - 0.1 * span_y, max(all_y) + 0.1 * span_y)
        print(f"Зум: x={x_range[0]:.1f}..{x_range[1]:.1f}, y={y_range[0]:.1f}..{y_range[1]:.1f}")

        for layer_path in layer_files:
            plot_layer_zoomed(
                layer_path=layer_path,
                polygons=polygons,
                output_path=audit_dir / f"{layer_path.stem}_zoom.png",
                x_range=x_range,
                y_range=y_range,
                cmap="gray",
            )

    # Зум на курганы — только если есть dtm.tif (лидарный участок).
    dtm_path = site_dir / "dtm.tif"
    if dtm_path.exists() and labels:
        grid = read_raster(dtm_path).grid
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
