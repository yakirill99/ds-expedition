"""Сборка растровых каналов rgb_* в каноническую сетку участка (D4.3).

Участки без лидара содержат только исходники организаторов:
    03_*_Ae_*.tif   немецкое аэрофото  -> rgb_r/g/b
    04_*_SpOr_*.tif спутник            -> rgb_r/g/b
    05_*_Or_*.tif   ортофото           -> rgb_r/g/b
    07..09 (MagGr/EtGr/TerFl) — D4.4, здесь не собираются.

Сетка: UTM по центроиду разметки, пиксель 1.0 м (как у всех 13 лидарных
участков), bbox разметки + отступ. Разрешения источников различаются в 60
раз (0.02..1.2 м), поэтому всё приводится к 1 м через WarpedVRT — потоково,
без загрузки исходника в память (008_Герите: 10.7 Гпкс, ~32 ГБ).

ОРИЕНТАЦИЯ: WarpedVRT отдаёт north-up (строка 0 = север), Grid требует
row 0 = юг — массив переворачивается сразу после чтения. write_raster
переворачивает обратно при записи. Несовпадение здесь не даёт ошибки,
только обнуляет F1.

Значения пишутся СЫРЫМИ (0..255). Робастная нормировка — в load_site,
где rgb_* нормируются совместно (см. normalize_rgb_group).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from expds.data.raster import write_raster
from expds.features.grid import Grid, Layer

Say = Callable[[str], None]

RGB_PREFIXES = ("03_", "04_", "05_")
RGB_CHANNELS = ("rgb_r", "rgb_g", "rgb_b")
PIXEL_SIZE = 1.0
PAD_PX = 64
MIN_COVERAGE = 0.05
NODATA = -9999.0
READ_STEP_ROWS = 256
_META_PATH = Path("samples") / "sample_meta.json"


def _noop(_: str) -> None:
    """Колбэк прогресса по умолчанию: молчит."""


def _iter_points(coords: object):
    """Все (x, y) из geometry.coordinates любой вложенности."""
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        yield coords[0], coords[1]
    else:
        for part in coords:
            yield from _iter_points(part)


def utm_epsg(lon: float, lat: float) -> str:
    """EPSG UTM-зоны по долготе/широте."""
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def canonical_grid(
    labels_path: Path,
    labels_crs: str,
    pixel_size: float = PIXEL_SIZE,
    pad_px: int = PAD_PX,
) -> Grid:
    """Сетка участка: UTM по центроиду разметки, bbox + отступ, выровнен по пикселю.

    Строится по разметке, а не по растру: у 012_Сож растр 19x18 км при
    49 объектах на всю площадь — сетка по растру дала бы 446 Мпкс.
    """
    feats = json.loads(Path(labels_path).read_text(encoding="utf-8"))["features"]
    xy = [
        p for f in feats for p in _iter_points((f.get("geometry") or {}).get("coordinates") or [])
    ]
    if not xy:
        raise ValueError(f"{labels_path}: в разметке нет координат")
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]

    lon, lat = Transformer.from_crs(labels_crs, "EPSG:4326", always_xy=True).transform(
        sum(xs) / len(xs), sum(ys) / len(ys)
    )
    crs = utm_epsg(lon, lat)
    ux, uy = Transformer.from_crs(labels_crs, crs, always_xy=True).transform(xs, ys)

    pad = pad_px * pixel_size
    x0 = np.floor((min(ux) - pad) / pixel_size) * pixel_size
    y0 = np.floor((min(uy) - pad) / pixel_size) * pixel_size
    x1 = np.ceil((max(ux) + pad) / pixel_size) * pixel_size
    y1 = np.ceil((max(uy) + pad) / pixel_size) * pixel_size
    return Grid(
        crs=crs,
        pixel_size=pixel_size,
        x_min=float(x0),
        y_min=float(y0),
        width=int(round((x1 - x0) / pixel_size)),
        height=int(round((y1 - y0) / pixel_size)),
    )


def grid_transform(grid: Grid):
    """North-up Affine из Grid (верхний левый угол)."""
    return from_origin(
        grid.x_min,
        grid.y_min + grid.height * grid.pixel_size,
        grid.pixel_size,
        grid.pixel_size,
    )


def read_to_grid(
    path: Path,
    grid: Grid,
    resampling: Resampling = Resampling.average,
    say: Say | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Источник -> (bands, H, W) float32 в конвенции Grid + маска покрытия (H, W).

    Чтение полосами по READ_STEP_ROWS строк: на /mnt/d один большой tif
    читается минутами, и без прогресса не понять, жив процесс или упал.
    """
    emit = say or _noop
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid_transform(grid),
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            src_nodata=src.nodata,
            nodata=NODATA,
        ) as vrt:
            out = np.empty((vrt.count, grid.height, grid.width), dtype=np.float32)
            for r0 in range(0, grid.height, READ_STEP_ROWS):
                h = min(READ_STEP_ROWS, grid.height - r0)
                out[:, r0 : r0 + h] = vrt.read(window=Window(0, r0, grid.width, h))
                done = (r0 + h) / grid.height
                emit(f"    {done:5.1%}  строки {r0 + h}/{grid.height}")
    out = np.flip(out, axis=-2)  # north-up -> row 0 = юг
    return out, (out != NODATA).any(axis=0)


def _pixel_size(path: Path) -> float:
    with rasterio.open(path) as ds:
        return float(abs(ds.res[0]))


def _grid_from_tif(path: Path) -> Grid:
    """Сетка GeoTIFF по метаданным, без чтения пикселей."""
    with rasterio.open(path) as ds:
        return Grid(
            crs=ds.crs.to_string(),
            pixel_size=float(abs(ds.res[0])),
            x_min=float(ds.bounds.left),
            y_min=float(ds.bounds.bottom),
            width=int(ds.width),
            height=int(ds.height),
        )


def composite_rgb(
    site_dir: Path, grid: Grid, say: Say | None = None
) -> tuple[np.ndarray, float, list[str]]:
    """Все 03_/04_/05_ -> (3, H, W). Грубые первыми, мелкие поверх.

    Листы одного участка перекрываются (048, 185) — побеждает более
    детальный источник там, где он есть. Панхром (012_Сож, 1 полоса)
    дублируется в три канала: цвета из него не восстановить.
    """
    emit = say or _noop
    tifs: list[Path] = []
    for sub in sorted(Path(site_dir).iterdir()):
        if sub.is_dir() and sub.name[:3] in RGB_PREFIXES:
            tifs.extend(sorted(sub.glob("*.tif")))
    if not tifs:
        return (
            np.full((3, grid.height, grid.width), NODATA, np.float32),
            0.0,
            ["нет 03_/04_/05_"],
        )

    tifs.sort(key=_pixel_size, reverse=True)
    out = np.full((3, grid.height, grid.width), NODATA, np.float32)
    covered = np.zeros(grid.shape, dtype=bool)
    notes: list[str] = []
    for tif in tifs:
        size_mb = tif.stat().st_size / 1024**2
        px = _pixel_size(tif)
        unit = "м/px" if px >= 0.01 else "°/px"
        emit(f"  читаю {tif.name} ({size_mb:.0f} МБ, {px:.4f} {unit}) ...")
        t1 = time.perf_counter()
        arr, cover = read_to_grid(tif, grid, say=emit)
        emit(f"    готово за {time.perf_counter() - t1:.1f} c, покрыто {cover.mean():.1%}")
        if not cover.any():
            notes.append(f"{tif.name}: мимо сетки")
            continue
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] > 3:
            arr = arr[:3]
        out[:, cover] = arr[:, cover]
        covered |= cover
        notes.append(f"{tif.name}: {int(cover.sum())} px, {px:.3f} м/px")
    return out, float(covered.mean()), notes


def site_grid(
    site_dir: Path,
    labels: Path,
    labels_crs: str,
    pixel_size: float = PIXEL_SIZE,
    pad_px: int = PAD_PX,
    say: Say | None = None,
) -> tuple[Grid, str]:
    """Сетка участка и имя её источника: meta | dtm | labels.

    Приоритет источников:
      1. samples/sample_meta.json — уже зафиксированный контракт;
      2. layers/dtm.tif — участок собран из LAS, сетку задаёт рельеф;
      3. разметка — участок только растровый, сетка считается впервые.

    Пункт 2 обязателен: у 001, 035, 105 есть layers/dtm.tif, но
    sample_meta.json отсутствует. Без него сетка считалась бы по разметке,
    а записанная мета перебила бы лидарную — и 27 рельефных слоёв уехали
    бы в ресемплинг (001: 516x775 → 202x192).

    Returns:
        (grid, source). source == "meta" — мета уже есть и актуальна,
        писать её не нужно. "dtm" и "labels" — писать обязательно.
    """
    emit = say or _noop
    meta_path = site_dir / _META_PATH
    dtm = site_dir / "layers" / "dtm.tif"
    lidar = _grid_from_tif(dtm) if dtm.exists() else None

    if meta_path.exists():
        grid = Grid(**json.loads(meta_path.read_text(encoding="utf-8"))["grid"])
        if lidar is None or (grid.crs, grid.x_min, grid.y_min, grid.width, grid.height) == (
            lidar.crs,
            lidar.x_min,
            lidar.y_min,
            lidar.width,
            lidar.height,
        ):
            return grid, "meta"
        emit(
            f"  мета не совпадает с dtm.tif: {grid.width}x{grid.height} против "
            f"{lidar.width}x{lidar.height} — беру сетку dtm, мету перезапишу"
        )
        return lidar, "dtm"

    if lidar is not None:
        return lidar, "dtm"

    return canonical_grid(labels, labels_crs, pixel_size, pad_px), "labels"


def build_site_rasters(
    site_dir: Path,
    labels_crs: str,
    pixel_size: float = PIXEL_SIZE,
    pad_px: int = PAD_PX,
    min_coverage: float = MIN_COVERAGE,
    max_pixels: int = 12_000_000,
    force: bool = False,
    log: Say | None = None,
) -> dict:
    """Собирает rgb_*.tif в site_dir/layers и пишет samples/sample_meta.json.

    Args:
        site_dir: папка участка с labels.geojson.
        labels_crs: CRS разметки (data.labels_crs).
        pixel_size: целевой пиксель, м.
        pad_px: отступ вокруг bbox разметки, px.
        min_coverage: минимум покрытия сетки, иначе rgb_* не пишутся
            (channel_mask останется False — «не снимали», не «нули»).
        max_pixels: предохранитель от 012_Сож/058_Перынь.
        force: пересобрать поверх готовых rgb_*.tif. Сетка при этом
            берётся заново по приоритету мета → dtm.tif → разметка:
            если мета успела испортиться, --force её починит.
        log: колбэк прогресса; по умолчанию print.

    Returns:
        dict с ключами site, built, reason, grid, coverage, notes.
    """
    say = log or print
    t0 = time.perf_counter()

    site_dir = Path(site_dir)
    labels = site_dir / "labels.geojson"
    if not labels.exists():
        raise FileNotFoundError(labels)

    say(f"  сетка: читаю {labels.name}")
    grid, grid_src = site_grid(site_dir, labels, labels_crs, pixel_size, pad_px, say=say)
    n_px = grid.width * grid.height
    say(
        f"  сетка: {grid.width}x{grid.height} {grid.crs} = {n_px / 1e6:.2f} Мпкс, "
        f"источник: {grid_src} [{time.perf_counter() - t0:.1f} c]"
    )
    if n_px > max_pixels:
        return {
            "site": site_dir.name,
            "built": False,
            "reason": (
                f"сетка {grid.width}x{grid.height} = {n_px / 1e6:.1f} Мпкс > {max_pixels / 1e6:.0f}"
            ),
            "grid": grid,
        }

    out_dir = site_dir / "layers"
    if not force and any(out_dir.glob("rgb_*.tif")):
        return {"site": site_dir.name, "built": False, "reason": "уже собран", "grid": grid}

    arr, coverage, notes = composite_rgb(site_dir, grid, say=say)
    say(f"  покрытие {coverage:.1%} [{time.perf_counter() - t0:.1f} c]")
    if coverage < min_coverage:
        return {
            "site": site_dir.name,
            "built": False,
            "reason": f"покрытие {coverage:.1%} < {min_coverage:.0%}",
            "grid": grid,
            "notes": notes,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(RGB_CHANNELS):
        write_raster(
            layer=Layer(name=name, data=arr[i], grid=grid, nodata=NODATA, unit="uint8"),
            path=out_dir / f"{name}.tif",
        )

    if grid_src != "meta":
        meta = site_dir / _META_PATH
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(
            json.dumps({"grid": asdict(grid)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    say(f"  записано [{time.perf_counter() - t0:.1f} c]")

    return {
        "site": site_dir.name,
        "built": True,
        "grid": grid,
        "coverage": coverage,
        "notes": notes,
    }
