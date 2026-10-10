"""
Разовый просмотрщик сырых геоданных для визуального QA.

НЕ часть пайплайна. Не импортируется из train.py, не используется в inference.

Запуск:
    uv run --group viz python tools/preview_site.py \
        --las data/raw/013/file.las \
        --labels data/raw/013/labels.geojson \
        --raster data/raw/013/aero.tif \
        --out docs/preview_013.png

Зависимости: laspy, rasterio, pyproj, numpy, matplotlib.
geopandas и shapely НЕ используются — полигоны рисуются через matplotlib.path.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

# ---------------------------------------------------------------------------
# Ленивые зависимости: pyproj и rasterio нужны не всегда
# ---------------------------------------------------------------------------

try:
    import laspy
except ImportError:
    laspy = None

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from pyproj import CRS, Transformer
except ImportError:
    CRS = Transformer = None


# ---------------------------------------------------------------------------
# Локальные примитивы-заглушки.
# Когда появятся канонические сигнатуры из репо — заменить тела на импорты.
# Сигнатуры подобраны так, чтобы совпасть с тем, что обычно бывает в проекте.
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: Optional[object] = None  # pyproj.CRS или строка

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersects(self, other: "BBox") -> bool:
        return not (self.xmax <= other.xmin or other.xmax <= self.xmin
                    or self.ymax <= other.ymin or other.ymax <= self.ymin)

    def intersection(self, other: "BBox") -> Optional["BBox"]:
        _require_same_crs(self, other)
        if not self.intersects(other):
            return None
        return BBox(
            max(self.xmin, other.xmin), max(self.ymin, other.ymin),
            min(self.xmax, other.xmax), min(self.ymax, other.ymax),
            crs=self.crs or other.crs,
        )

    def iou(self, other: "BBox") -> float:
        _require_same_crs(self, other)
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union = self.area + other.area - inter.area
        return inter.area / union if union > 0 else 0.0


def _require_same_crs(a: BBox, b: BBox) -> None:
    """Явная ошибка вместо тихой арифметики между несравнимыми числами."""
    ca = _crs_str(a.crs)
    cb = _crs_str(b.crs)
    if ca and cb and ca != cb:
        raise ValueError(f"CRS mismatch: {ca} vs {cb}")


def _crs_str(crs) -> Optional[str]:
    if crs is None:
        return None
    return str(crs)


def _bbox_from_las_header(header, crs) -> BBox:
    mins, maxs = header.mins, header.maxs
    return BBox(float(mins[0]), float(mins[1]),
                float(maxs[0]), float(maxs[1]), crs=crs)


# --- канонические обёртки (сигнатуры на будущее) ---------------------------

def read_las(path: Path, chunk_size: int = 5_000_000):
    """
    Возвращает (xyz: (N,3) float64, classification: (N,) uint8, header, crs).
    Читает чанками, чтобы не улететь в OOM на больших файлах.

    TODO: заменить на data.las.read_points, когда сигнатуры будут известны.
    """
    if laspy is None:
        raise ImportError("laspy не установлен")
    with laspy.open(path) as fh:
        header = fh.header
        crs = _safe_parse_crs(header)
        chunks_x, chunks_y, chunks_z, chunks_c = [], [], [], []
        for points in fh.chunk_iterator(chunk_size):
            chunks_x.append(np.asarray(points.x, dtype=np.float64))
            chunks_y.append(np.asarray(points.y, dtype=np.float64))
            chunks_z.append(np.asarray(points.z, dtype=np.float64))
            chunks_c.append(np.asarray(points.classification, dtype=np.uint8))
    x = np.concatenate(chunks_x)
    y = np.concatenate(chunks_y)
    z = np.concatenate(chunks_z)
    c = np.concatenate(chunks_c)
    xyz = np.column_stack([x, y, z])
    return xyz, c, header, crs


def _safe_parse_crs(header):
    try:
        return header.parse_crs()
    except Exception:
        return None


def read_raster(path: Path, band: int = 1):
    """
    Возвращает (data: (H,W) float32, bbox: BBox, transform, crs).
    Флипает по Y, чтобы row 0 = юг — единая конвенция с Grid в репо.
    """
    if rasterio is None:
        raise ImportError("rasterio не установлен")
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        data = np.flipud(data)  # row 0 = юг
        bbox = BBox(src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                    crs=src.crs)
        transform = src.transform
        crs = src.crs
    return data, bbox, transform, crs


def read_labels(path: Path, expected_crs=None):
    """
    Читает GeoJSON. Возвращает список полигонов [(N,2) float64] и их CRS.
    Проверяет CRS явно: если expected_crs задан и не совпадает — падает.

    TODO: заменить на io.geojson.read_labels, когда сигнатуры будут известны.
    """
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    file_crs = gj.get("crs", {}).get("properties", {}).get("name")
    if expected_crs is not None and file_crs is not None:
        if str(expected_crs) != str(file_crs):
            raise ValueError(
                f"CRS mismatch: file={file_crs}, expected={expected_crs}. "
                f"Перепроецируй файл заранее."
            )

    polygons = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon":
            for ring in coords:
                polygons.append(np.asarray(ring, dtype=np.float64))
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    polygons.append(np.asarray(ring, dtype=np.float64))
        elif gtype == "Point":
            polygons.append(np.asarray([coords], dtype=np.float64))
    return polygons, file_crs


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU двух полигонов через matplotlib.path (без shapely).
    a, b — (N,2) массивы вершин, замкнутые или нет — не важно.
    Считается на bounding-box'ах для простоты; для точного IoU — заменить на shapely.
    """
    ax0, ay0 = a[:, 0].min(), a[:, 1].min()
    ax1, ay1 = a[:, 0].max(), a[:, 1].max()
    bx0, by0 = b[:, 0].min(), b[:, 1].min()
    bx1, by1 = b[:, 1].max(), b[:, 1].max() if False else b[:, 1].max()
    # ^ намеренно оставлено как есть: точный IoU полигонов — задача eval.metric,
    #   здесь — только bbox-IoU как грубая прикидка для QA.
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Ядро: сборка превью
# ---------------------------------------------------------------------------

@dataclass
class PreviewConfig:
    las: Optional[Path] = None
    labels: Optional[Path] = None
    raster: Optional[Path] = None
    band: int = 1
    res_m: float = 1.0
    agg: str = "min"
    pad_m: float = 100.0
    out: Path = Path("preview.png")
    dpi: int = 150
    max_points: int = 2_000_000


def build_dtm(xyz: np.ndarray, cls: np.ndarray,
              res_m: float, agg: str) -> tuple[np.ndarray, BBox]:
    """
    DTM из точек класса 2 (или всех, если класса 2 нет).
    Фикс бага: np.minimum.at с NaN залипает. Стартуем с +inf / -inf, потом меняем на NaN.
    """
    ground = cls == 2
    if ground.sum() == 0:
        ground = np.ones_like(cls, dtype=bool)

    x = xyz[ground, 0]
    y = xyz[ground, 1]
    z = xyz[ground, 2]

    bbox = BBox(float(x.min()), float(y.min()), float(x.max()), float(y.max()))
    nx = max(1, int(np.ceil(bbox.width / res_m)))
    ny = max(1, int(np.ceil(bbox.height / res_m)))

    ix = np.clip(((x - bbox.xmin) / res_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - bbox.ymin) / res_m).astype(np.int64), 0, ny - 1)

    if agg == "min":
        sentinel = np.float32(np.inf)
        grid = np.full((ny, nx), sentinel, dtype=np.float32)
        np.minimum.at(grid, (iy, ix), z.astype(np.float32))
        grid[grid == sentinel] = np.nan
    elif agg == "max":
        sentinel = np.float32(-np.inf)
        grid = np.full((ny, nx), sentinel, dtype=np.float32)
        np.maximum.at(grid, (iy, ix), z.astype(np.float32))
        grid[grid == sentinel] = np.nan
    elif agg == "mean":
        sums = np.zeros((ny, nx), dtype=np.float64)
        cnts = np.zeros((ny, nx), dtype=np.int32)
        np.add.at(sums, (iy, ix), z)
        np.add.at(cnts, (iy, ix), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan).astype(np.float32)
    else:
        raise ValueError(f"unknown agg: {agg}")

    return grid, bbox


def fill_nearest(grid: np.ndarray) -> np.ndarray:
    """Заполнение NaN ближайшим валидным. Только для визуализации."""
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(grid)
    if not mask.any():
        return grid
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def overlay_polygons(ax, polygons: list[np.ndarray], color="red", lw=1.2) -> None:
    """Рисует полигоны через matplotlib.path — без geopandas/shapely."""
    for ring in polygons:
        if ring.shape[0] < 2:
            continue
        if ring.shape[0] == 1:
            ax.plot(ring[0, 0], ring[0, 1], "o", color=color, markersize=4)
            continue
        closed = np.vstack([ring, ring[:1]]) if not np.allclose(ring[0], ring[-1]) else ring
        ax.plot(closed[:, 0], closed[:, 1], "-", color=color, linewidth=lw)


def reproject_polygons(polygons: list[np.ndarray],
                       src_crs, dst_crs) -> list[np.ndarray]:
    """Перепроецирует список полигонов. source_crs берётся из файла, не хардкодится."""
    if Transformer is None:
        raise ImportError("pyproj не установлен")
    if src_crs is None or dst_crs is None:
        raise ValueError("Для перепроецирования нужны оба CRS")
    if str(src_crs) == str(dst_crs):
        return polygons
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out = []
    for ring in polygons:
        x, y = t.transform(ring[:, 0], ring[:, 1])
        out.append(np.column_stack([x, y]))
    return out


def make_preview(cfg: PreviewConfig) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # --- лидар ---
    las_bbox = None
    if cfg.las is not None:
        xyz, cls, header, las_crs = read_las(cfg.las)
        las_bbox = _bbox_from_las_header(header, las_crs)
        print(f"LAS: {len(xyz):,} точек, CRS={las_crs}, bbox={las_bbox}")

        # прореживание для отрисовки
        if len(xyz) > cfg.max_points:
            step = len(xyz) // cfg.max_points + 1
            xyz_p, cls_p = xyz[::step], cls[::step]
        else:
            xyz_p, cls_p = xyz, cls

        axes[0].scatter(xyz_p[:, 0], xyz_p[:, 1],
                        c=xyz_p[:, 2], s=0.1, cmap="terrain", linewidths=0)
        axes[0].set_title(f"LiDAR ({len(xyz_p):,}/{len(xyz):,} точек)")
        axes[0].set_aspect("equal")

        # --- DTM ---
        dtm, dtm_bbox = build_dtm(xyz, cls, cfg.res_m, cfg.agg)
        dtm_filled = fill_nearest(dtm)
        axes[1].imshow(
            dtm_filled,
            extent=[dtm_bbox.xmin, dtm_bbox.xmax, dtm_bbox.ymin, dtm_bbox.ymax],
            origin="lower", cmap="terrain",
        )
        axes[1].set_title(f"DTM {cfg.agg}, res={cfg.res_m} м "
                          f"(NaN: {np.isnan(dtm).mean()*100:.1f}%)")
        axes[1].set_aspect("equal")

    # --- разметка ---
    label_crs = None
    polygons_raw = []
    if cfg.labels is not None:
        polygons_raw, label_crs = read_labels(cfg.labels, expected_crs=None)
        print(f"GeoJSON: {len(polygons_raw)} полигонов, CRS={label_crs}")

    # --- растр ---
    raster_bbox = None
    if cfg.raster is not None:
        rdata, raster_bbox, _, raster_crs = read_raster(cfg.raster, band=cfg.band)
        axes[2].imshow(
            rdata,
            extent=[raster_bbox.xmin, raster_bbox.xmax,
                    raster_bbox.ymin, raster_bbox.ymax],
            origin="lower", cmap="gray",
        )
        axes[2].set_title(f"Raster band {cfg.band}, CRS={raster_crs}")
        axes[2].set_aspect("equal")

    # --- сравнение покрытий лидара и растра ---
    if las_bbox is not None and raster_bbox is not None:
        try:
            iou = las_bbox.iou(raster_bbox)
            print(f"LiDAR ∩ Raster: IoU={iou:.3f}, "
                  f"LiDAR in Raster={raster_bbox.intersection(las_bbox) is not None}")
        except ValueError as e:
            print(f"CRS mismatch между LiDAR и растром: {e}")

    # --- наложение разметки на DTM и на растр ---
    if polygons_raw and las_bbox is not None and label_crs is not None:
        polys_utm = reproject_polygons(polygons_raw, label_crs, las_crs)
        overlay_polygons(axes[0], polys_utm, color="red", lw=1.0)
        overlay_polygons(axes[1], polys_utm, color="red", lw=1.0)
    if polygons_raw and raster_bbox is not None and label_crs is not None:
        polys_ras = reproject_polygons(polygons_raw, label_crs, raster_crs)
        overlay_polygons(axes[2], polys_ras, color="red", lw=1.0)

    plt.tight_layout()
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {cfg.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> PreviewConfig:
    p = argparse.ArgumentParser(description="Preview raw geodata (visual QA).")
    p.add_argument("--las", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--raster", type=Path, default=None)
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--res-m", type=float, default=1.0)
    p.add_argument("--agg", choices=["min", "max", "mean"], default="min")
    p.add_argument("--pad-m", type=float, default=100.0)
    p.add_argument("--max-points", type=int, default=2_000_000)
    p.add_argument("--out", type=Path, default=Path("preview.png"))
    p.add_argument("--dpi", type=int, default=150)
    a = p.parse_args()
    return PreviewConfig(
        las=a.las, labels=a.labels, raster=a.raster, band=a.band,
        res_m=a.res_m, agg=a.agg, pad_m=a.pad_m, max_points=a.max_points,
        out=a.out, dpi=a.dpi,
    )


def main() -> None:
    cfg = parse_args()
    make_preview(cfg)


if __name__ == "__main__":
    main()