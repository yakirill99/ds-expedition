"""Подготовка тайлов локации из исходных данных (GeoTIFF, LAS, GeoJSON).

Пример:
    uv run python scripts/prepare_tiles.py \\
        --location data/001_Гнёздово_Нивленское \\
        --output web/tiles \\
        --zooms 13 14 15 16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import laspy
import mercantile
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds as window_from_bounds

TILE_SIZE = 256


def slugify(name: str) -> str:
    """Простая транслитерация имени папки в ASCII-slug."""
    # Для MVP оставляем только цифры, латиницу и дефисы.
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return "".join(c if c in allowed else "-" for c in name).strip("-").lower()


def load_manifest(location_dir: Path) -> dict:
    """Прочитать manifest.json локации или вернуть пустой словарь."""
    manifest_path = location_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def save_manifest(location_dir: Path, manifest: dict) -> None:
    """Сохранить manifest.json локации."""
    manifest_path = location_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_layers(location_dir: Path) -> list[tuple[str, Path]]:
    """Найти исходные файлы и сопоставить их со slug-ами слоёв.

    Возвращает список пар (layer_slug, file_path).
    """
    layers: list[tuple[str, Path]] = []
    for subdir in location_dir.iterdir():
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()

        # Точки LiDAR
        if "la_" in name or ("lidar" in name and "карты" not in name):
            for ext in ("*.las", "*.laz"):
                for f in sorted(subdir.glob(ext)):
                    layers.append(("lidar_points", f))

        # Карты LiDAR
        elif "li_" in name or "lidar" in name:
            for ext in ("*.tif", "*.tiff"):
                for f in sorted(subdir.glob(ext)):
                    suffix = f.stem.split("_")[-1]
                    layer_slug = f"lidar_{suffix}"
                    layers.append((layer_slug, f))

        # Аэрофотосъёмка
        elif "ae_" in name or "aerial" in name or "аэро" in name:
            for ext in ("*.tif", "*.tiff"):
                for f in sorted(subdir.glob(ext)):
                    layers.append(("aerial", f))

        # Разметка
        elif "разметка" in name or "annotation" in name:
            for f in sorted(subdir.glob("*.geojson")):
                suffix = f.stem.split("_")[-1] if "_" in f.stem else "annotation"
                layers.append((f"annotation_{suffix}", f))

    return layers


def _normalize_for_png(data: np.ndarray) -> np.ndarray:
    """Привести растровый массив к uint8 для сохранения в PNG."""
    data = np.asarray(data)

    # Отсекаем маски nodata (обычно 0 или очень большие значения)
    if np.issubdtype(data.dtype, np.floating):
        valid = data[np.isfinite(data) & (data != 0)]
        if valid.size == 0:
            return np.zeros(data.shape, dtype=np.uint8)
        vmin, vmax = np.percentile(valid, [2, 98])
        norm = (data - vmin) / (vmax - vmin + 1e-9) * 255
        return np.clip(norm, 0, 255).astype(np.uint8)

    if data.dtype == np.uint16:
        return (data // 256).astype(np.uint8)

    if data.dtype == np.uint8:
        return data

    # Прочие целые типы
    vmin, vmax = data.min(), data.max()
    if vmin == vmax:
        return np.zeros(data.shape, dtype=np.uint8)
    norm = (data.astype(np.float32) - vmin) / (vmax - vmin) * 255
    return np.clip(norm, 0, 255).astype(np.uint8)


def _save_png(data: np.ndarray, path: Path) -> None:
    """Сохранить numpy-массив (bands, h, w) как PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.shape[0] == 1:
        img = Image.fromarray(data[0])
    else:
        arr = np.transpose(data, (1, 2, 0))
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        img = Image.fromarray(arr)
    img.save(path, "PNG")


def geotiff_to_tiles(
    src_path: Path,
    output_dir: Path,
    zooms: tuple[int, ...],
    crs_epsg: int,
) -> None:
    """Нарезать GeoTIFF на тайлы формата {z}/{x}/{y}.png."""
    _ = crs_epsg
    src_crs = rasterio.CRS.from_epsg(crs_epsg)

    with rasterio.open(src_path) as src:
        src_bounds = src.bounds

        # Границы растра в WGS84 для поиска тайлов
        west, south, east, north = rasterio.warp.transform_bounds(src_crs, "EPSG:4326", *src_bounds)

        for z in zooms:
            tiles = list(mercantile.tiles(west, south, east, north, zooms=z))
            if not tiles:
                continue

            for tile in tiles:
                tile_bounds = mercantile.bounds(tile)
                xs, ys = warp_transform(
                    "EPSG:4326",
                    src_crs,
                    [tile_bounds.west, tile_bounds.east],
                    [tile_bounds.south, tile_bounds.north],
                )
                left, right = min(xs), max(xs)
                bottom, top = min(ys), max(ys)

                window = window_from_bounds(left, bottom, right, top, src.transform)
                row_off, col_off, height, width = window.flatten()

                # Пропускаем тайлы, полностью лежащие за пределами растра
                if col_off >= src.width or row_off >= src.height:
                    continue
                if col_off + width <= 0 or row_off + height <= 0:
                    continue

                data = src.read(
                    window=window,
                    out_shape=(src.count, TILE_SIZE, TILE_SIZE),
                    resampling=Resampling.bilinear,
                    boundless=True,
                )
                data = np.nan_to_num(data, nan=0.0)
                data = _normalize_for_png(data)

                out_path = output_dir / str(z) / str(tile.x) / f"{tile.y}.png"
                _save_png(data, out_path)


def las_to_geotiff(
    las_path: Path,
    output_path: Path,
    crs_epsg: int,
    resolution: float = 0.5,
    attribute: str = "z",
) -> None:
    """Растрировать LAS-облако точек во временный GeoTIFF.

    По умолчанию используется высота (z). Если в файле есть intensity,
    можно передать --las-attribute intensity.
    """
    las = laspy.read(las_path)
    x = np.asarray(las.x)
    y = np.asarray(las.y)

    if attribute == "z":
        values = np.asarray(las.z)
    elif attribute in las.point_format.dimension_names:
        values = np.asarray(getattr(las, attribute))
    else:
        raise ValueError(f"Атрибут '{attribute}' отсутствует в LAS-файле")

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    width = max(1, int(np.ceil((xmax - xmin) / resolution)))
    height = max(1, int(np.ceil((ymax - ymin) / resolution)))

    ix = ((x - xmin) / resolution).astype(np.int64)
    iy = ((ymax - y) / resolution).astype(np.int64)
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)

    grid = np.full((height, width), -np.inf, dtype=np.float32)
    np.maximum.at(grid, (iy, ix), values)
    grid[np.isinf(grid)] = 0.0

    # Нормализация высот в диапазон 0-255
    valid = grid[grid > 0]
    if valid.size > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
        grid_uint8 = np.clip((grid - vmin) / (vmax - vmin + 1e-9) * 255, 0, 255).astype(np.uint8)
    else:
        grid_uint8 = np.zeros((height, width), dtype=np.uint8)

    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    dst_crs_str = f"EPSG:{crs_epsg}"
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=grid_uint8.dtype,
        crs=dst_crs_str,
        transform=transform,
    ) as dst:
        dst.write(grid_uint8, 1)


def update_manifest(location_dir: Path, manifest: dict, layers: list[tuple[str, Path]]) -> None:
    """Дополнить manifest списком слоёв и общими границами, если их ещё нет."""
    manifest["layers"] = [
        {"slug": slug, "source": str(path.relative_to(location_dir).as_posix())}
        for slug, path in layers
    ]

    if "bounds" not in manifest or "center" not in manifest:
        # Пробуем вычислить bounds из первого GeoTIFF
        geotiffs = [p for s, p in layers if p.suffix.lower() in (".tif", ".tiff")]
        if geotiffs:
            with rasterio.open(geotiffs[0]) as src:
                west, south, east, north = rasterio.warp.transform_bounds(
                    src.crs, "EPSG:4326", *src.bounds
                )
                manifest.setdefault(
                    "bounds",
                    {"west": west, "south": south, "east": east, "north": north},
                )
                manifest.setdefault(
                    "center",
                    {"lat": (south + north) / 2, "lon": (west + east) / 2},
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Подготовить тайлы локации из исходных данных")
    parser.add_argument("--location", required=True, type=Path, help="Папка локации")
    parser.add_argument("--output", default="web/tiles", type=Path, help="Куда сохранять тайлы")
    parser.add_argument(
        "--zooms",
        type=int,
        nargs="+",
        default=[13, 14, 15, 16],
        help="Уровни масштабирования для нарезки (по умолчанию 13 14 15 16)",
    )
    parser.add_argument(
        "--crs",
        type=int,
        default=None,
        help="EPSG-код проекции исходных данных (по умолчанию из manifest или 32636)",
    )
    parser.add_argument(
        "--las-resolution",
        type=float,
        default=0.5,
        help="Разрешение растра LAS в метрах (по умолчанию 0.5)",
    )
    parser.add_argument(
        "--las-attribute",
        default="z",
        help="Атрибут LAS для растрирования: z или intensity (по умолчанию z)",
    )
    args = parser.parse_args(argv)

    if not args.location.is_dir():
        print(f"Ошибка: папка не найдена: {args.location}", file=sys.stderr)
        return 1

    manifest = load_manifest(args.location)
    crs_epsg = args.crs or int(manifest.get("crs", "EPSG:32636").split(":")[-1])
    slug = manifest.get("slug") or slugify(args.location.name)

    output_root = args.output / slug
    output_root.mkdir(parents=True, exist_ok=True)

    layers = discover_layers(args.location)
    if not layers:
        print("Не найдено слоёв для обработки.", file=sys.stderr)
        return 1

    print(f"Локация: {slug}")
    print(f"CRS: EPSG:{crs_epsg}")
    print(f"Zoom-уровни: {args.zooms}")
    print("Найдены слои:")
    for layer_slug, src_path in layers:
        print(f"  - {layer_slug}: {src_path}")

    # Обработка растровых слоёв
    for layer_slug, src_path in layers:
        if src_path.suffix.lower() in (".tif", ".tiff"):
            layer_output = output_root / layer_slug
            print(f"\nНарезка {layer_slug} ...")
            geotiff_to_tiles(src_path, layer_output, tuple(args.zooms), crs_epsg)

    # Обработка LAS
    las_layers = [(s, p) for s, p in layers if p.suffix.lower() in (".las", ".laz")]
    if las_layers:
        layer_slug, src_path = las_layers[0]
        layer_output = output_root / layer_slug
        with tempfile_path(output_root / "_tmp_lidar_points.tif") as tmp_tif:
            print(f"\nРастрирование LiDAR-точек {src_path.name} ...")
            las_to_geotiff(
                src_path,
                tmp_tif,
                crs_epsg,
                resolution=args.las_resolution,
                attribute=args.las_attribute,
            )
            print(f"Нарезка {layer_slug} ...")
            geotiff_to_tiles(tmp_tif, layer_output, tuple(args.zooms), crs_epsg)

    # GeoJSON пока копируем как есть в корень output (не как слой тайлов)
    geojson_layers = [(s, p) for s, p in layers if p.suffix.lower() == ".geojson"]
    if geojson_layers:
        print("\nКопирование GeoJSON-разметки ...")
        for _, src_path in geojson_layers:
            dst_name = f"annotation_{src_path.name}"
            shutil.copy(src_path, output_root / dst_name)

    update_manifest(args.location, manifest, layers)
    save_manifest(args.location, manifest)
    # Копируем манифест рядом с тайлами, чтобы import_location мог его найти.
    shutil.copy(args.location / "manifest.json", output_root / "manifest.json")
    print(f"\nГотово. Тайлы сохранены в: {output_root}")
    print(f"Манифест обновлён: {args.location / 'manifest.json'}")
    return 0


@contextmanager
def tempfile_path(path: Path):
    """Контекстный менеджер для временного файла, который удаляется после выхода."""
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
