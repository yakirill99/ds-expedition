"""Карты вероятностей (2K, H, W) -> Detection-полигоны в crs.internal.

seg-голова: порог -> 8-связные компоненты -> контур по вероятности на уровне порога
(субпиксельно, skimage.find_contours) -> Дуглас-Пекер -> мир. Дыры пока отбрасываются.
Кольца Detection — без замыкающей точки (контракт io.geojson), CCW.
hm-голова: локальные максимумы -> субпиксельное уточнение (парабола по log) -> круг.
Координаты по контракту Grid: row 0 = юг, центр пикселя x = x_min + (col + 0.5) * ps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from expds.features.grid import Grid
from expds.io.geojson import Detection

from .nms import nms_centroids

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class PostConfig:
    """Параметры постобработки (секция `post` YAML), размеры в пикселях."""

    seg_thr: float = 0.5
    min_area_px: float = 16.0
    simplify_px: float = 0.5
    hm_thr: float = 0.3
    peak_window_px: int = 7
    point_radius_px: float = 4.0
    point_vertices: int = 16
    nms_radius_px: float = 6.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> PostConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"post: unknown keys {sorted(unknown)}")
        cfg = cls(**d)
        if not (0.0 < cfg.seg_thr < 1.0 and 0.0 < cfg.hm_thr < 1.0):
            raise ValueError("post: thresholds must be in (0, 1)")
        if cfg.peak_window_px < 3 or cfg.peak_window_px % 2 == 0:
            raise ValueError("post.peak_window_px must be odd and >= 3")
        if cfg.point_vertices < 3:
            raise ValueError("post.point_vertices must be >= 3")
        return cfg


# --- геометрия -----------------------------------------------------------------


def _signed_area(ring: np.ndarray) -> float:
    """Ориентированная площадь; кольцо замкнутое или нет — без разницы."""
    x, y = ring[:, 0] - ring[0, 0], ring[:, 1] - ring[0, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _dp_keep(pts: np.ndarray, tol: float) -> np.ndarray:
    """Маска точек ломаной, оставленных Дугласом-Пекером (концы всегда остаются)."""
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        a, b = pts[s], pts[e]
        seg = b - a
        norm = float(np.hypot(*seg))
        mid = pts[s + 1 : e] - a
        if norm == 0.0:
            dist = np.hypot(mid[:, 0], mid[:, 1])
        else:
            dist = np.abs(seg[0] * mid[:, 1] - seg[1] * mid[:, 0]) / norm
        i = int(np.argmax(dist))
        if dist[i] > tol:
            j = s + 1 + i
            keep[j] = True
            stack += [(s, j), (j, e)]
    return keep


def simplify_ring(ring: np.ndarray, tol: float) -> np.ndarray:
    """Дуглас-Пекер для замкнутого кольца [N, 2] (first == last). Возвращает замкнутое кольцо."""
    ring = np.asarray(ring, dtype=np.float64)
    pts = ring[:-1] if np.array_equal(ring[0], ring[-1]) else ring
    if tol <= 0 or len(pts) < 4:
        return np.vstack([pts, pts[:1]])
    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    a = pts[: far + 1]
    b = np.vstack([pts[far:], pts[:1]])
    out = np.vstack([a[_dp_keep(a, tol)], b[_dp_keep(b, tol)][1:-1]])
    if len(out) < 3:
        return np.vstack([pts, pts[:1]])
    return np.vstack([out, out[:1]])


def _pix_to_world(rc: np.ndarray, grid: Grid) -> np.ndarray:
    """(row, col) дробные индексы пикселей -> (x, y) в crs.internal."""
    ps = grid.pixel_size
    x = grid.x_min + (rc[:, 1] + 0.5) * ps
    y = grid.y_min + (rc[:, 0] + 0.5) * ps
    return np.column_stack([x, y])


def _ccw(ring: np.ndarray) -> np.ndarray:
    return ring[::-1].copy() if _signed_area(ring) < 0 else ring


# --- seg-голова ----------------------------------------------------------------


def seg_to_polygons(prob: np.ndarray, grid: Grid, cls: str, cfg: PostConfig) -> list[Detection]:
    """(H, W) вероятность одного класса -> полигоны компонент (внешний контур, CCW)."""
    h, w = prob.shape
    lab, n = ndimage.label(prob >= cfg.seg_thr, structure=_EIGHT)
    dets: list[Detection] = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        if comp.sum() < cfg.min_area_px:
            continue
        r0, r1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, h)
        c0, c1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, w)
        sub = prob[r0:r1, c0:c1].astype(np.float64)
        lab_sub = lab[r0:r1, c0:c1]
        sub[(lab_sub > 0) & (lab_sub != i)] = 0.0
        contours = find_contours(np.pad(sub, 1), cfg.seg_thr)
        if not contours:
            continue
        outer = max(contours, key=lambda c: abs(_signed_area(c)))
        rc = outer + np.array([r0 - 1, c0 - 1], dtype=np.float64)
        rc = simplify_ring(rc, cfg.simplify_px)
        if len(rc) < 4:
            continue
        ring = _ccw(_pix_to_world(rc[:-1], grid))
        score = float(prob[sl][comp].mean())
        dets.append(Detection(cls=cls, score=score, geom_type="Polygon", geometry=ring))
    return dets


# --- hm-голова -----------------------------------------------------------------


def _refine(v_minus: float, v0: float, v_plus: float) -> float:
    """Смещение вершины параболы по log-значениям (точно для гаусса), в [-0.5, 0.5]."""
    lm, l0, lp = (np.log(max(v, 1e-12)) for v in (v_minus, v0, v_plus))
    den = lm - 2.0 * l0 + lp
    if den >= 0:
        return 0.0
    return float(np.clip(0.5 * (lm - lp) / den, -0.5, 0.5))


def heatmap_to_points(hm: np.ndarray, grid: Grid, cls: str, cfg: PostConfig) -> list[Detection]:
    """(H, W) heatmap одного класса -> круги радиуса point_radius_px вокруг пиков."""
    h, w = hm.shape
    mx = ndimage.maximum_filter(hm, size=cfg.peak_window_px, mode="constant", cval=0.0)
    rows, cols = np.nonzero((hm >= mx) & (hm >= cfg.hm_thr))
    t = np.linspace(0.0, 2.0 * np.pi, cfg.point_vertices, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)]) * cfg.point_radius_px * grid.pixel_size
    dets: list[Detection] = []
    for r, c in zip(rows, cols, strict=True):
        dr = _refine(hm[r - 1, c], hm[r, c], hm[r + 1, c]) if 0 < r < h - 1 else 0.0
        dc = _refine(hm[r, c - 1], hm[r, c], hm[r, c + 1]) if 0 < c < w - 1 else 0.0
        center = _pix_to_world(np.array([[r + dr, c + dc]]), grid)[0]
        ring = center + circle
        dets.append(Detection(cls=cls, score=float(hm[r, c]), geom_type="Polygon", geometry=ring))
    return dets


# --- всё вместе ----------------------------------------------------------------


def postprocess(
    probs: np.ndarray,
    grid: Grid,
    classes: Sequence[str],
    cfg: PostConfig | None = None,
    valid: np.ndarray | None = None,
) -> list[Detection]:
    """(2K, H, W) вероятности [seg_0..K-1, hm_0..K-1] -> детекции после NMS."""
    cfg = cfg or PostConfig()
    k = len(classes)
    if probs.ndim != 3 or probs.shape[0] != 2 * k:
        raise ValueError(f"probs must be ({2 * k}, H, W), got {probs.shape}")
    if probs.shape[1:] != (grid.height, grid.width):
        raise ValueError(f"probs {probs.shape[1:]} vs grid ({grid.height}, {grid.width})")
    if valid is not None:
        probs = probs * valid.astype(probs.dtype)[None]
    dets: list[Detection] = []
    for i, cls in enumerate(classes):
        dets += seg_to_polygons(probs[i], grid, cls, cfg)
        dets += heatmap_to_points(probs[k + i], grid, cls, cfg)
    return nms_centroids(dets, cfg.nms_radius_px * grid.pixel_size)
