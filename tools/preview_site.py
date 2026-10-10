"""
Разовый просмотрщик сырых геоданных для визуального QA.

НЕ часть пайплайна. Не импортируется из train.py, не используется в inference.

Запуск:
    uv run --group viz python tools/preview_site.py \
        --las data/raw/013/file.las --las-crs EPSG:32636 \
        --labels data/raw/013/labels.geojson \
        --raster data/raw/013/aero.tif \
        --out docs/preview_013.png

Зависимости: laspy, rasterio, pyproj, numpy, scipy, matplotlib.
geopandas и shapely НЕ используются — полигоны рисуются через matplotlib.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import laspy
except ImportError:
    laspy = None

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from pyproj import Transformer
except ImportError:
    Transformer = None


# ---------------------------------------------------------------------------
# Примитивы
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: Optional[str] = None

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
        _warn_if_crs_unknown(self, other)
        if not self.intersects(other):
            return None
        return BBox(
            max(self.xmin, other.xmin), max(self.ymin, other.ymin),
            min(self.xmax, other.xmax), min(self.ymax, other.ymax),
            crs=self.crs or other.crs,
        )

    def iou(self, other: "BBox") -> float:
        _warn_if_crs_unknown(self, other)
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union = self.area + other.area - inter.area
        return inter.area / union if union > 0 else 0.0


def _crs_str(crs) -> Optional[str]:
    if crs is None:
        return None
    return str(crs)


def _require_same_crs(a: BBox, b: BBox) -> None:
    ca, cb = _crs_str(a.crs), _crs_str(b.crs)
    if ca and cb and ca != cb:
        raise ValueError(f"CRS mismatch: {ca} vs {cb}")


def _warn_if_crs_unknown(a: BBox, b: BBox) -> None:
    ca, cb = _crs_str(a.crs), _crs_str(b.crs)
    if ca is None or cb is None:
        warnings.warn(
            f"CRS не определён у одного из bbox ({ca} vs {cb}); "
            f"результат может быть бессмысленным."
        )


def _bbox_from_las_header(header, crs) -> BBox:
    mins, maxs = header.mins, header.maxs
    return BBox(float(mins[0]), float(mins[1]),
                float(maxs[0]), float(maxs[1]), crs=crs)


# ---------------------------------------------------------------------------
# Чтение
# ---------------------------------------------------------------------------

def read_las(path: Path, chunk_size: int = 5_000_000):
    """Читает LAS чанками. Возвращает (xyz, classification, header, crs_or_None)."""
    if laspy is None:
        raise ImportError("laspy не установлен")
    with laspy.open(path) as fh:
        header = fh.header
        try:
            crs = header.parse_crs()
        except Exception:
            crs = None
        chunks = []
        for points in fh.chunk_iterator(chunk_size):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            c = np.asarray(points.classification, dtype=np.uint8)
            chunks.append((x, y, z, c))
    xs = np.concatenate([c[0] for c in chunks])
    ys = np.concatenate([c[1] for c in chunks])
    zs = np.concatenate([c[2] for c in chunks])
    cs = np.concatenate([c[3] for c in chunks])
    xyz = np.column_stack([xs, ys, zs])
    return xyz, cs, header, crs


def read_raster(path: Path, band: int = 1):
    """Читает GeoTIFF. Флипает по Y: row 0 = юг."""
    if rasterio is None:
        raise ImportError("rasterio не установлен")
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        data = np.flipud(data)
        bbox = BBox(src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                    crs=str(src.crs) if src.crs else None)
        transform = src.transform
        crs = str(src.crs) if src.crs else None
    return data, bbox, transform, crs


def read_labels(path: Path, expected_crs=None):
    """Читает GeoJSON. Возвращает (список полигонов (N,2), CRS или None)."""
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    file_crs = gj.get("crs", {}).get("properties", {}).get("name")
    if expected_crs is not None and file_crs is not None:
        if str(expected_crs) != str(file_crs):
            raise ValueError(
                f"CRS mismatch: file={file_crs}, expected={expected_crs}."
            )

    polygons = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if coords is None:
            continue
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


# ---------------------------------------------------------------------------
# Обработка
# ---------------------------------------------------------------------------

def build_dtm(xyz: np.ndarray, cls: np.ndarray,
              res_m: float, agg: str) -> tuple[np.ndarray, BBox]:
    """DTM из точек класса 2. Фикс: inf вместо nan при minimum/maximum.at."""
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
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(grid)
    if not mask.any():
        return grid
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def reproject_polygons(polygons: list[np.ndarray],
                       src_crs, dst_crs) -> list[np.ndarray]:
    if Transformer is None:
        raise ImportError("pyproj не установлен")
    if src_crs is None or dst_crs is None:
        raise ValueError(f"Нужны оба CRS (src={src_crs}, dst={dst_crs})")
    if str(src_crs) == str(dst_crs):
        return polygons
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out = []
    for ring in polygons:
        x, y = t.transform(ring[:, 0], ring[:, 1])
        out.append(np.column_stack([x, y]))
    return out


def overlay_polygons(ax, polygons: list[np.ndarray], color="red", lw=1.2) -> None:
    for ring in polygons:
        if ring.shape[0] < 2:
            continue
        if ring.shape[0] == 1:
            ax.plot(ring[0, 0], ring[0, 1], "o", color=color, markersize=4)
            continue
        closed = np.vstack([ring, ring[:1]]) if not np.allclose(ring[0], ring[-1]) else ring
        ax.plot(closed[:, 0], closed[:, 1], "-", color=color, linewidth=lw)


# ---------------------------------------------------------------------------
# Сборка превью
# ---------------------------------------------------------------------------

@dataclass
class PreviewConfig:
    las: Optional[Path] = None
    labels: Optional[Path] = None
    raster: Optional[Path] = None
    las_crs: Optional[str] = None
    band: int = 1
    res_m: float = 1.0
    agg: str = "min"
    pad_m: float = 100.0
    out: Path = Path("preview.png")
    dpi: int = 150
    max_points: int = 2_000_000


def make_preview(cfg: PreviewConfig) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # --- лидар ---
    las_bbox = None
    las_crs = None
    dtm_bbox = None
    if cfg.las is not None:
        xyz, cls, header, las_crs_header = read_las(cfg.las)
        las_bbox = _bbox_from_las_header(header, las_crs_header)

        # разрешение CRS: header → --las-crs → предупреждение
        las_crs = las_crs_header or cfg.las_crs
        if las_crs_header is None and cfg.las_crs is not None:
            print(f"LAS: CRS не в header, используем --las-crs={cfg.las_crs}")
        if las_crs is None:
            print("LAS: CRS неизвестен (нет в header и не задан --las-crs). "
                  "Разметка на облако накладываться не будет.")

        print(f"LAS: {len(xyz):,} точек, CRS={las_crs}, bbox={las_bbox}")

        if len(xyz) > cfg.max_points:
            step = len(xyz) // cfg.max_points + 1
            xyz_p, cls_p = xyz[::step], cls[::step]
        else:
            xyz_p, cls_p = xyz, cls

        axes[0].scatter(xyz_p[:, 0], xyz_p[:, 1],
                        c=xyz_p[:, 2], s=0.1, cmap="terrain", linewidths=0)
        axes[0].set_title(f"LiDAR ({len(xyz_p):,}/{len(xyz):,} точек)")
        axes[0].set_aspect("equal")

        # DTM
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
    raster_crs = None
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

    # --- сравнение покрытий LiDAR и Raster ---
    if las_bbox is not None and raster_bbox is not None:
        try:
            iou = las_bbox.iou(raster_bbox)
            print(f"LiDAR ∩ Raster: IoU={iou:.3f}")
        except ValueError as e:
            print(f"CRS mismatch между LiDAR и Raster: {e}")

    # --- наложение разметки ---
    if polygons_raw and label_crs:
        # на лидар/DTM
        if las_crs is not None:
            polys_utm = reproject_polygons(polygons_raw, label_crs, las_crs)
            overlay_polygons(axes[0], polys_utm, color="red", lw=1.0)
            overlay_polygons(axes[1], polys_utm, color="red", lw=1.0)
        else:
            print("Пропускаю наложение на LiDAR: CRS лидара неизвестен.")

        # на растр
        if raster_crs is not None:
            polys_ras = reproject_polygons(polygons_raw, label_crs, raster_crs)
            overlay_polygons(axes[2], polys_ras, color="red", lw=1.0)
        else:
            print("Пропускаю наложение на Raster: CRS растра неизвестен.")
    elif polygons_raw:
        print("Пропускаю наложение: у GeoJSON нет CRS.")

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
    p.add_argument("--las-crs", type=str, default=None,
                   help="CRS лидара, если в header его нет (напр. EPSG:32636)")
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
        las=a.las, labels=a.labels, raster=a.raster,
        las_crs=a.las_crs, band=a.band,
        res_m=a.res_m, agg=a.agg, pad_m=a.pad_m,
        max_points=a.max_points, out=a.out, dpi=a.dpi,
    )


def main() -> None:
    cfg = parse_args()
    make_preview(cfg)


if __name__ == "__main__":
    main()