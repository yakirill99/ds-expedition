"""Сборка каналов mag_* из сырых геофизических источников (D4.4, приоритет gorodishcha).

Контракт каналов: configs/dataset_real.yaml -> data.channels -> mag_<field>.
kind: raster, как у gpr_slice_* и rgb_*. Сетка — та же, что у участка
(dtm.tif или samples/sample_meta.json), чтобы stack_layers не ресемплил.

Источники: подпапки 07_/08_/09_ (MagGr / EtGr / TerFl). Здесь берём только
магнитометрию (MagGr), георадар и электротомографию оставляем gpr-модулю
и следующим PR.

Ориентация: Grid row 0 = юг, сырой north-up — переворачивается после чтения.
Привязка — по границам через WarpedVRT (transform от нижнего левого угла сетки),
без floor от центров ячеек, поэтому баг _bin_indices сюда не протекает.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from expds.data.raster import write_raster
from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)

MAG_PREFIXES = ("07_",)  # MagGr; EtGr/TerFl — не магнитометрия
MAG_CHANNEL_PREFIX = "mag_"
NODATA = -9999.0
READ_STEP_ROWS = 256
META_PATH = Path("samples") / "sample_meta.json"

# Поля, которые умеет писать магнитометрия. Если в имени файла не распознано —
# используем "tmi" (total magnetic intensity) как дефолт.
DEFAULT_FIELD = "tmi"


def _mag_sources(site_dir: Path) -> list[Path]:
    """Все магнитные растры участка (07_* с .tif)."""
    tifs: list[Path] = []
    for sub in sorted(Path(site_dir).iterdir()):
        if sub.is_dir() and sub.name[:3] in MAG_PREFIXES:
            tifs.extend(sorted(sub.glob("*.tif")))
    return tifs


def _grid_from_dtm(site_dir: Path) -> Grid | None:
    """Сетка участка из layers/dtm.tif (приоритет) или sample_meta.json."""
    dtm = site_dir / "layers" / "dtm.tif"
    if dtm.exists():
        with rasterio.open(dtm) as src:
            return Grid(
                crs=str(src.crs),
                pixel_size=float(abs(src.res[0])),
                x_min=float(src.bounds.left),
                y_min=float(src.bounds.bottom),
                width=int(src.width),
                height=int(src.height),
            )
    meta = site_dir / META_PATH
    if meta.exists():
        return Grid(**json.loads(meta.read_text(encoding="utf-8"))["grid"])
    return None


def _grid_from_mag(site_dir: Path) -> Grid:
    """Запасной путь: сетка по bbox первого магнитного растра."""
    tifs = _mag_sources(site_dir)
    if not tifs:
        raise FileNotFoundError(f"{site_dir}: нет магнитных растров (07_*)")
    with rasterio.open(tifs[0]) as src:
        return Grid(
            crs=str(src.crs) if src.crs else "EPSG:3857",
            pixel_size=float(abs(src.res[0])),
            x_min=float(src.bounds.left),
            y_min=float(src.bounds.bottom),
            width=int(src.width),
            height=int(src.height),
        )


def _field_from_name(path: Path) -> str:
    """Поле из имени файла: mag_tmi.tif -> tmi, mag_hg.tif -> hg, иначе tmi."""
    stem = path.stem.lower().replace("-", "_")
    for token in stem.split("_"):
        if token in {"tmi", "hg", "vg", "rtp", "up", "drm"}:
            return token
    return DEFAULT_FIELD


def read_mag_to_grid(path: Path, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Магнитный растр -> (H, W) float32 в конвенции Grid + маска покрытия.

    Сырой north-up, поэтому после WarpedVRT массив переворачивается.
    Чтение полосами — на /mnt/d большие tif читаются минутами.
    """
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=from_origin(
                grid.x_min,
                grid.y_min + grid.height * grid.pixel_size,
                grid.pixel_size,
                grid.pixel_size,
            ),
            width=grid.width,
            height=grid.height,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            nodata=NODATA,
        ) as vrt:
            out = np.empty((grid.height, grid.width), dtype=np.float32)
            for r0 in range(0, grid.height, READ_STEP_ROWS):
                h = min(READ_STEP_ROWS, grid.height - r0)
                out[r0 : r0 + h] = vrt.read(1, window=Window(0, r0, grid.width, h))
    out = np.flip(out, axis=0)  # north-up -> Grid (row 0 = юг)
    return out, out != NODATA


def build_mag_layers(
    site_dir: Path,
    grid: Grid | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Собирает mag_*.tif в site_dir/layers. Возвращает имя -> путь.

    Поле берётся из имени файла (tmi/hg/vg/rtp/up/drm), иначе tmi.
    Если полей несколько — каждое отдельным каналом mag_<field>.
    """
    site_dir = Path(site_dir)
    out_dir = site_dir / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(out_dir.glob(f"{MAG_CHANNEL_PREFIX}*.tif")):
        return {p.stem: p for p in out_dir.glob(f"{MAG_CHANNEL_PREFIX}*.tif")}

    sources = _mag_sources(site_dir)
    if not sources:
        return {}

    grid = grid or _grid_from_dtm(site_dir) or _grid_from_mag(site_dir)
    written: dict[str, Path] = {}

    for idx, path in enumerate(sources):
        field = _field_from_name(path)
        name = f"{MAG_CHANNEL_PREFIX}{field}"
        if name in written:
            name = f"{name}_{idx}"

        data, cover = read_mag_to_grid(path, grid)
        if not cover.any():
            logger.warning("%s: магнитный растр не покрывает сетку участка", path.name)
            continue

        layer = Layer(
            name=name,
            data=data.astype(np.float32),
            grid=grid,
            nodata=NODATA,
            source=str(path),
            unit="nT",
        )
        dst = out_dir / f"{name}.tif"
        write_raster(layer=layer, path=dst)
        written[name] = dst
        logger.info(
            "%s -> %s (поле %s, покрытие %.1f%%)",
            path.name,
            dst.name,
            field,
            cover.mean() * 100,
        )

    return written
