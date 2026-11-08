"""Сборка каналов gpr_slice из сырых GPR-источников (D4.4, приоритет finds_points/object_poly).

Контракт каналов: configs/dataset_real.yaml -> data.channels -> gpr_slice_<depth>.
kind: raster, как у rgb_* и рельефа. Сетка — та же, что у участка (dtm.tif или
sample_meta.json), чтобы stack_layers не ресемплил и не терял пиксели.

Ориентация: Grid row 0 = юг. Сырой GPR-слайс приходит north-up (строка 0 = север),
поэтому после чтения массив переворачивается. write_raster переворачивает обратно.

ВАЖНО про _bin_indices: в cloud_features он использует transform_to_pixel (центр
ячейки) + floor, что сдвигает выборку на -0.5 px. Здесь для привязки GPR-амплитуд
к сетке используется Grid.world_to_cell — он работает по границам, без сдвига.
Если переиспользуется cloud-фича, сдвиг нужно компенсировать явно (+0.5).
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

GPR_PREFIXES = ("07_", "08_", "09_")  # MagGr / EtGr / TerFl; GPR-слайсы могут быть в любом
GPR_CHANNEL_PREFIX = "gpr_slice_"
PIXEL_SIZE = 1.0
NODATA = -9999.0
READ_STEP_ROWS = 256
META_PATH = Path("samples") / "sample_meta.json"

# Глубины слайсов, под которые в dataset_real.yaml уже объявлены каналы.
# Ключ — суффикс канала после gpr_slice_; значение — целевая глубина в метрах,
# если в имени файла её нет.
DEFAULT_DEPTHS_M = (0.5, 1.0, 1.5, 2.0)


def _gpr_sources(site_dir: Path) -> list[Path]:
    """Все GPR-растры участка (07_/08_/09_ с .tif)."""
    tifs: list[Path] = []
    for sub in sorted(Path(site_dir).iterdir()):
        if sub.is_dir() and sub.name[:3] in GPR_PREFIXES:
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


def _grid_from_gpr(site_dir: Path) -> Grid:
    """Запасной путь: сетка по bbox первого GPR-растра, приведённая к 1 м UTM."""
    tifs = _gpr_sources(site_dir)
    if not tifs:
        raise FileNotFoundError(f"{site_dir}: нет GPR-растров (07_/08_/09_)")
    with rasterio.open(tifs[0]) as src:
        bounds = src.bounds
        crs = src.crs
        # Если CRS не UTM-метры, всё равно берём как есть: build_site_rasters
        # уже умеет приводить к UTM через grid_from_sources, но здесь
        # контракт — сетка dtm. Для участков без dtm оставляем родной CRS.
        return Grid(
            crs=str(crs) if crs else "EPSG:3857",
            pixel_size=float(abs(src.res[0])),
            x_min=float(bounds.left),
            y_min=float(bounds.bottom),
            width=int(src.width),
            height=int(src.height),
        )


def _depth_from_name(path: Path) -> float | None:
    """Глубина из имени файла: gpr_1.5m.tif / slice_150cm.tif -> метры."""
    stem = path.stem.lower()
    for token in stem.replace("-", "_").split("_"):
        token = token.strip()
        try:
            value = float(token.rstrip("m"))
        except ValueError:
            continue
        if "cm" in token:
            return value / 100.0
        if value < 10.0:  # 0.5, 1.0, 1.5, 2.0 — уже метры
            return value
    return None


def read_slice_to_grid(path: Path, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """GPR-слайс -> (H, W) float32 в конвенции Grid + маска покрытия.

    Сырой слайс north-up, поэтому после WarpedVRT массив переворачивается.
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


def build_gpr_layers(
    site_dir: Path,
    grid: Grid | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Собирает gpr_slice_*.tif в site_dir/layers. Возвращает имя -> путь.

    Глубины берутся из имён файлов, иначе — DEFAULT_DEPTHS_M по числу источников.
    Если источников больше, чем имён глубин, лишние получают суффикс _n.
    """
    site_dir = Path(site_dir)
    out_dir = site_dir / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(out_dir.glob(f"{GPR_CHANNEL_PREFIX}*.tif")):
        return {p.stem: p for p in out_dir.glob(f"{GPR_CHANNEL_PREFIX}*.tif")}

    sources = _gpr_sources(site_dir)
    if not sources:
        return {}

    grid = grid or _grid_from_dtm(site_dir) or _grid_from_gpr(site_dir)
    written: dict[str, Path] = {}

    for idx, path in enumerate(sources):
        depth = _depth_from_name(path)
        if depth is None:
            depth = DEFAULT_DEPTHS_M[idx] if idx < len(DEFAULT_DEPTHS_M) else float(idx + 1)
        suffix = f"{depth:g}".replace(".", "_")
        name = f"{GPR_CHANNEL_PREFIX}{suffix}"
        if name in written:
            name = f"{name}_{idx}"

        data, cover = read_slice_to_grid(path, grid)
        if not cover.any():
            logger.warning("%s: GPR-слайс не покрывает сетку участка", path.name)
            continue

        layer = Layer(
            name=name,
            data=data.astype(np.float32),
            grid=grid,
            nodata=NODATA,
            source=str(path),
            unit="amplitude",
        )
        dst = out_dir / f"{name}.tif"
        write_raster(layer=layer, path=dst)
        written[name] = dst
        logger.info(
            "%s -> %s (глубина %s м, покрытие %.1f%%)",
            path.name,
            dst.name,
            suffix,
            cover.mean() * 100,
        )

    return written
