"""Карты вероятностей (2K, H, W) -> Detection-полигоны в crs.internal.

seg-голова: порог -> 8-связные компоненты -> контур по вероятности на уровне порога
(субпиксельно, skimage.find_contours) -> Дуглас-Пекер -> мир. Дыры сохраняются:
внутренние контуры (обходят низ значений — знак площади противоположен внешнему)
попадают в Detection.holes. Кольца Detection — без замыкающей точки
(контракт io.geojson): внешнее CCW, дыры CW.
Слитые компоненты (касание/перекрытие курганов) делятся по пикам hm
(_split_component): пики считают объекты, watershed по рельефу hm
проводит границу по седлу между гауссианами; без split_touching —
Voronoi по пикам (старое поведение).
hm-голова: локальные максимумы -> субпиксельное уточнение (парабола по log) -> круг.
Координаты по контракту Grid: row 0 = юг, центр пикселя x = x_min + (col + 0.5) * ps.

Marching squares оставлен в skimage (BSD-3): совместимо по лицензии, выверено,
и даёт субпиксельные уровневые линии из коробки — своя реализация на время
хакатона не даёт выигрыша в качестве (обоснование в PR 02-post-vector).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours
from skimage.segmentation import watershed

from expds.features.grid import Grid
from expds.io.geojson import Detection

from .nms import nms_centroids

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class PostConfig:
    """Параметры постобработки (секция `post` YAML), размеры в пикселях."""

    branches: dict[str, str] | None = None
    split_touching: bool = False
    split_min_distance_px: int = 4
    seg_thr: float = 0.5
    min_area_px: float = 16.0
    simplify_px: float = 0.5
    hm_thr: float = 0.3
    peak_window_px: int = 7
    point_radius_px: float = 4.0
    point_vertices: int = 16
    nms_radius_px: float = 6.0
    # Отдельные параметры heatmap-ветки (точки). None — наследовать общие:
    # старые конфиги ведут себя ровно как раньше. Нужны, потому что общие
    # значения настроены под ДЕЛЕНИЕ полигонов (peak_window 7, NMS 4-6 px),
    # а плотные точечные кластеры (находки через 2-4 м) ими глотаются:
    # синтетика 2026-11-11 — recall 0.636 на шаге 2-4 м.
    point_nms_radius_px: float | None = None
    point_peak_window_px: int | None = None
    max_detections: int = 2000

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> PostConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"post: unknown keys {sorted(unknown)}")
        cfg = cls(**d)
        if not (0.0 < cfg.seg_thr < 1.0 and 0.0 < cfg.hm_thr < 1.0):
            raise ValueError("post: thresholds must be in (0, 1)")
        if cfg.split_min_distance_px < 0:
            raise ValueError("post.split_min_distance_px must be >= 0")
        if cfg.peak_window_px < 3 or cfg.peak_window_px % 2 == 0:
            raise ValueError("post.peak_window_px must be odd and >= 3")
        if cfg.point_peak_window_px is not None and (
            cfg.point_peak_window_px < 3 or cfg.point_peak_window_px % 2 == 0
        ):
            raise ValueError("post.point_peak_window_px must be odd and >= 3")
        if cfg.point_nms_radius_px is not None and cfg.point_nms_radius_px < 0:
            raise ValueError("post.point_nms_radius_px must be >= 0")
        if cfg.point_vertices < 3:
            raise ValueError("post.point_vertices must be >= 3")
        if cfg.max_detections < 1:
            raise ValueError("post.max_detections must be >= 1")
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


def _cw(ring: np.ndarray) -> np.ndarray:
    return ring[::-1].copy() if _signed_area(ring) > 0 else ring


def _component_holes(
    contours: Sequence[np.ndarray],
    outer: np.ndarray,
    r0: int,
    c0: int,
    grid: Grid,
    cfg: PostConfig,
) -> tuple[np.ndarray, ...]:
    """Дыры одной компоненты: контуры с ориентацией, противоположной внешнему.

    В sub обнулены чужие компоненты, поэтому любой внутренний контур обходит
    низ значений внутри этой же компоненты — то есть дыру. Дыры тоже
    упрощаются и переводятся в мир (CW, без замыкающей точки).
    """
    outer_sign = np.sign(_signed_area(outer))
    holes: list[np.ndarray] = []
    for contour in contours:
        if contour is outer or np.sign(_signed_area(contour)) == outer_sign:
            continue
        rc = contour + np.array([r0 - 1, c0 - 1], dtype=np.float64)
        rc = simplify_ring(rc, cfg.simplify_px)
        if len(rc) < 4:
            continue
        holes.append(_cw(_pix_to_world(rc[:-1], grid)))
    return tuple(holes)


def _split_by_peaks(comp: np.ndarray, hm_sub: np.ndarray, cfg: PostConfig) -> np.ndarray:
    """Делит компонент по ближайшему пику тепловой карты внутри него.

    Args:
        comp: (h, w) bool — маска одного компонента.
        hm_sub: (h, w) float — та же область тепловой карты того же класса.
        cfg: параметры постобработки (peak_window_px, hm_thr).

    Returns:
        (h, w) int32: 0 вне компонента, 1..N — номер части по числу пиков.

    Зачем: y_seg растеризует соприкасающиеся полигоны в одну маску, label()
    даёт один компонент на 2-4 кургана. Разделение по пикам hm возвращает
    их число; без него IoU блоба с каждым объектом < 0.5 и все уходят в fn.
    """
    ys, xs = _hm_peaks(comp, hm_sub, cfg)
    return _voronoi_by_peaks(comp, ys, xs)


def _hm_peaks(
    comp: np.ndarray, hm_sub: np.ndarray, cfg: PostConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Координаты пиков hm внутри компонента: локальный максимум >= hm_thr.

    Плато схлопывается в один пик: центр гауссианы часто попадает между
    пикселями (центроид в (r+0.5, c+0.5)), тогда 2x2 пикселя имеют ровно
    одинаковое значение 1.0, и `hm >= maximum_filter(hm)` оставляет ВСЕ
    четыре. Без дедупликации Voronoi делит один объект на сектора
    (IoU ~0.25, все в fn), а watershed получает ложные маркеры.
    Представитель плато — максимум hm; при равенстве минимальный (row, col).
    """
    mx = ndimage.maximum_filter(hm_sub, size=cfg.peak_window_px, mode="constant", cval=0.0)
    peaks = (hm_sub >= mx) & (hm_sub >= cfg.hm_thr) & comp
    if not peaks.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    lab, n = ndimage.label(peaks, structure=_EIGHT)
    ys = np.empty(n, dtype=np.int64)
    xs = np.empty(n, dtype=np.int64)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = np.where(lab[sl] == i, hm_sub[sl], -1.0)
        r, c = np.unravel_index(int(np.argmax(sub)), sub.shape)
        ys[i - 1] = sl[0].start + r
        xs[i - 1] = sl[1].start + c
    return ys, xs


def _voronoi_by_peaks(comp: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Каждый пиксель компонента — к ближайшему пику (1..N). Нулевой пиксель — 0.

    При равных расстояниях побеждает минимальный исходный индекс пика
    (семантика np.argmin, закреплена регрессионным тестом).
    """
    out = np.zeros(comp.shape, dtype=np.int32)
    if len(ys) < 2:
        out[comp] = 1
        return out
    cy, cx = np.nonzero(comp)
    pts = np.column_stack([cy, cx]).astype(np.float64)
    seeds = np.column_stack([ys, xs]).astype(np.float64)
    # Точный поиск ближайшего пика без полной матрицы расстояний.
    from scipy.spatial import cKDTree

    tree = cKDTree(seeds)
    assigned = np.empty(len(pts), dtype=np.int32)

    for start in range(0, len(pts), 4096):
        block = pts[start : start + 4096]
        _, idx = tree.query(block, k=2, eps=0.0)

        delta = block[:, None, :] - seeds[idx]
        d2 = (delta * delta).sum(axis=2)
        best = idx[:, 0].copy()

        # KDTree не гарантирует порядок равноудалённых соседей.
        # Как np.argmin, выбираем минимальный исходный индекс пика.
        for j in np.flatnonzero(d2[:, 0] == d2[:, 1]):
            radius = np.nextafter(np.sqrt(d2[j, 0]), np.inf)
            candidates = np.asarray(
                tree.query_ball_point(block[j], radius, eps=0.0),
                dtype=np.intp,
            )
            delta_j = seeds[candidates] - block[j]
            exact_d2 = (delta_j * delta_j).sum(axis=1)
            best[j] = candidates[exact_d2 == d2[j, 0]].min()

        assigned[start : start + len(block)] = best + 1

    out[cy, cx] = assigned
    return out


def _seed_markers(
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
    hm_sub: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    """Маркеры watershed из пиков; близкие пики схлопываются в сильнейший.

    Без слияния шумная hm модели даёт два маркера в паре пикселей —
    один объект разваливается на части. Порог — split_min_distance_px.
    """
    order = np.argsort(-hm_sub[ys, xs], kind="stable")
    markers = np.zeros(shape, dtype=np.int32)
    kept: list[tuple[int, int]] = []
    for idx in order:
        r, c = int(ys[idx]), int(xs[idx])
        if min_distance > 0 and any(
            (r - kr) ** 2 + (c - kc) ** 2 < min_distance * min_distance for kr, kc in kept
        ):
            continue
        kept.append((r, c))
        markers[r, c] = len(kept)
    return markers


def _watershed_parts(
    comp: np.ndarray,
    hm_sub: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    cfg: PostConfig,
) -> np.ndarray | None:
    """Делит компонент watershed по рельефу heatmap (седло между объектами).

    Маркеры — пики hm (см. _hm_peaks и _seed_markers), террейн — -hm:
    граница проходит по седлу между гауссианами (для равных сигм —
    биссектор между центрами, что и есть граница касания курганов).

    Раньше деление шло по distance transform и запускалось по глобальному
    условию len(peaks) > n_components: пики мелких объектов не проходили
    порог threshold_abs = min_distance/2, и ОДИН мелкий компонент отключал
    деление всех слитых (на rebuilt-данных 78 < 87 — watershed молча
    не запускался ни разу), а для кругов разных радиусов резал вглубь
    малого (IoU 0.46). hm-рельеф лишён обеих проблем.
    """
    markers = _seed_markers(comp.shape, ys, xs, hm_sub, cfg.split_min_distance_px)
    if int(markers.max()) < 2:
        return None
    ws = watershed(-hm_sub.astype(np.float64), markers, mask=comp).astype(np.int32)
    return ws if int(ws.max()) >= 2 else None


def _split_component(comp: np.ndarray, hm_sub: np.ndarray, cfg: PostConfig) -> np.ndarray:
    """Деление слитого компонента на объекты.

    Слой 1: пики hm считают объекты; <= 1 пика — компонент целиком.
    Слой 2 (split_touching): watershed по рельефу hm, маркеры — пики;
    каждый бассейн = один объект.
    Без split_touching — Voronoi по пикам (старое поведение).

    Returns:
        (h, w) int32: 0 вне компонента, 1..N — части.
    """
    ys, xs = _hm_peaks(comp, hm_sub, cfg)
    out = np.zeros(comp.shape, dtype=np.int32)
    if len(ys) < 2:
        out[comp] = 1
        return out
    if not cfg.split_touching:
        return _voronoi_by_peaks(comp, ys, xs)
    ws = _watershed_parts(comp, hm_sub, ys, xs, cfg)
    if ws is None:
        return _voronoi_by_peaks(comp, ys, xs)
    return ws


# --- seg-голова ----------------------------------------------------------------


def seg_to_polygons(
    prob: np.ndarray,
    grid: Grid,
    cls: str,
    cfg: PostConfig,
    hm: np.ndarray | None = None,
) -> list[Detection]:
    """(H, W) вероятность и heatmap одного класса -> полигоны компонент.

    Компонент, внутри которого больше одного пика hm, делится на объекты
    (_split_component): watershed по перешейкам + Voronoi по пикам.
    Растеризация y_seg склеивает соприкасающиеся полигоны, без деления
    каждый блоб даёт один fp и N fn.
    """
    h, w = prob.shape
    if hm is None:
        hm = np.zeros((h, w), dtype=prob.dtype)
    binary = prob >= cfg.seg_thr
    lab, n = ndimage.label(binary, structure=_EIGHT)
    dets: list[Detection] = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        if comp.sum() < cfg.min_area_px:
            continue
        r0, r1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, h)
        c0, c1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, w)
        full = prob[r0:r1, c0:c1].astype(np.float64)
        lab_sub = lab[r0:r1, c0:c1]
        full[(lab_sub > 0) & (lab_sub != i)] = 0.0
        parts = _split_component(lab_sub == i, hm[r0:r1, c0:c1], cfg)
        # Площади всех частей считаем одним проходом.
        # Маску создаём только для частей, прошедших порог площади.
        areas = np.bincount(parts.ravel())
        eligible = np.flatnonzero(~(areas[1:] < cfg.min_area_px)) + 1
        for p in eligible:
            mask = parts == p
            sub = np.where(mask | (lab_sub == 0), full, 0.0)
            contours = find_contours(np.pad(sub, 1), cfg.seg_thr)
            if not contours:
                continue
            outer = max(contours, key=lambda c: abs(_signed_area(c)))
            rc = outer + np.array([r0 - 1, c0 - 1], dtype=np.float64)
            rc = simplify_ring(rc, cfg.simplify_px)
            if len(rc) < 4:
                continue
            ring = _ccw(_pix_to_world(rc[:-1], grid))
            holes = _component_holes(contours, outer, r0, c0, grid, cfg)
            score = float(prob[r0:r1, c0:c1][mask].mean())
            dets.append(
                Detection(cls=cls, score=score, geom_type="Polygon", geometry=ring, holes=holes)
            )
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
    pw = cfg.point_peak_window_px if cfg.point_peak_window_px is not None else cfg.peak_window_px
    mx = ndimage.maximum_filter(hm, size=pw, mode="constant", cval=0.0)
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


# Heatmap-ветка включается только вместе с геофизическими каналами:
# единственный Point-класс (finds_points) в лидарной разметке отсутствует,
# а полигональным классам точечные предсказания дают только FP.
BRANCH_POLYGON = "polygon"
BRANCH_POINT = "point"
BRANCH_BOTH = "both"
BRANCHES = (BRANCH_POLYGON, BRANCH_POINT, BRANCH_BOTH)


def _branch_for(cls: str, cfg: "PostConfig") -> str:
    """Какие ветки включать для класса.

    По умолчанию обе — исходное поведение, на нём держатся тесты и
    весь дальнейший пайплайн. Лидарный конфиг переопределяет: у шести
    классов из семи разметка полигональная, и точка с полигоном не
    проходит IoU>=0.5, то есть точечные предсказания по ним — чистый FP.
    """
    b = (cfg.branches or {}).get(cls, BRANCH_BOTH)
    if b not in BRANCHES:
        raise ValueError(f"post.branches[{cls!r}]={b!r}, ожидалось одно из {BRANCHES}")
    return b


def oracle_postcfg(cfg: PostConfig) -> PostConfig:
    """Конфиг «щедрого потолка» оракула: обе ветки у всех классов.

    Гейт обучения и oracle_diag обязаны мерить ОДНО И ТО ЖЕ — возможности
    post+матчинга без прод-ограничений branches (см. докстринг oracle.py).
    Прод-конфиг может резать ветки — это его право, но тогда гейт и
    диагностика разъезжаются в числах и отладка «одних и тех же ошибок»
    не сходится никогда (night3 2026-11-11: гейт f1 0.872 против diag
    0.910 — circles hm-ветки спасали ~200 GT только в diag).

    Внимание: под coverage-матчингом платформы круги hm-ветки матчатся
    с GT-полигонами (cov = inter/min-площадь >= 0.5), поэтому «точка по
    полигональному классу — чистый FP» верно только для строгого IoU.
    """
    return replace(cfg, branches=None)


def postprocess(
    probs: np.ndarray,
    grid: Grid,
    classes: Sequence[str],
    cfg: PostConfig | None = None,
    valid: np.ndarray | None = None,
) -> list[Detection]:
    """(2K, H, W) вероятности [seg_0..K-1, hm_0..K-1] -> детекции после NMS (топ max_detections)."""
    cfg = cfg or PostConfig()
    k = len(classes)
    if probs.ndim != 3 or probs.shape[0] != 2 * k:
        raise ValueError(f"probs must be ({2 * k}, H, W), got {probs.shape}")
    if probs.shape[1:] != (grid.height, grid.width):
        raise ValueError(f"probs {probs.shape[1:]} vs grid ({grid.height}, {grid.width})")
    if valid is not None:
        # Маска nodata говорит «здесь не считай ошибку», а не «здесь ничего нет».
        # Прямое умножение рвёт объект на куски у границы дыры (010: 94 компоненты).
        # Закрытие 2 px убирает нити, крупные блоки остаются исключёнными.
        m = ndimage.binary_closing(valid, structure=np.ones((3, 3), dtype=bool), iterations=2)
        probs = probs * m.astype(probs.dtype)[None]
    # NMS раздельный: у сег-полигонов и у точек разные радиусы. Общий проход
    # гасил круги находок рядом с центроидом кургана и соседей в плотных
    # кластерах точек (радиус 4-6 м при шаге разметки 2-4 м).
    poly_dets: list[Detection] = []
    hm_dets: list[Detection] = []
    for i, cls in enumerate(classes):
        branch = _branch_for(cls, cfg)
        if branch in (BRANCH_POLYGON, BRANCH_BOTH):
            # hm обязателен для деления слитых компонент: без него
            # соприкасающиеся полигоны дают один блоб = 1 TP и N-1 FN
            # (n3_long: oracle recall 0.244 именно поэтому).
            poly_dets += seg_to_polygons(probs[i], grid, cls, cfg, hm=probs[k + i])
        if branch in (BRANCH_POINT, BRANCH_BOTH):
            hm_dets += heatmap_to_points(probs[k + i], grid, cls, cfg)
    point_r = cfg.point_nms_radius_px if cfg.point_nms_radius_px is not None else cfg.nms_radius_px
    kept = nms_centroids(poly_dets, cfg.nms_radius_px * grid.pixel_size)
    kept += nms_centroids(hm_dets, point_r * grid.pixel_size)
    # Склейка веток BOTH: круг hm и сег-полигон одного объекта не должны
    # выживать оба. Для классов, живущих в одной ветке, проход no-op:
    # свои радиусы там уже отработали, min(nms, point) <= каждого.
    kept = nms_centroids(kept, min(cfg.nms_radius_px, point_r) * grid.pixel_size)
    return sorted(kept, key=lambda d: -d.score)[: cfg.max_detections]
