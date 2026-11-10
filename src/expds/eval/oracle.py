"""Оракульная диагностика: где именно пайплайн теряет эталонные объекты.

GT прогоняется через ту же цепочку, что и модель (растеризация -> post ->
NMS -> матчинг), и каждый объект получает причину потери:

- ``вне сетки``   — полигон не оставил ни одного пикселя в растеризации;
- ``valid/nodata`` — все пиксели объекта вырезаны маской валидности;
- ``min_area``    — растеризованная площадь ниже post.min_area_px;
- ``слияние``     — объект попал в компонент с другими GT и не был выделен
  делением (пиков hm меньше, чем объектов, либо часть не прошла min_area);
- ``nms``         — полигон объекта был построен, но подавлен NMS;
- ``контур``      — полигон построен, но не прошёл матчинг (IoU ниже порога);
- ``ок``          — объект нашёлся (TP).

При eval.ignore_offgrid=true объекты «вне сетки» исключаются из матчинга
(не считаются FN): ни одна модель не может детектировать то, чего нет на
растровых данных. Аудит 2026-11-11 (docs/issues/07-grid-coverage.md): 456
из 461 FN оракула на night3 — именно они; гейт обучения блокировался
объектами вне bbox растра, а не ошибками пайплайна.

Функция :func:`oracle_report` возвращает стадии и причины — её печатают
scripts/oracle_diag.py (локально, на реальных участках) и используют тесты.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from expds.eval.metric import (
    ALL,
    EvalConfig,
    MatchResult,
    evaluate,
    objects_from_detections,
    objects_from_labels,
)
from expds.features.grid import Grid
from expds.io.geojson import LabelFeature
from expds.post.nms import nms_centroids
from expds.post.vectorize import PostConfig, heatmap_to_points, seg_to_polygons
from expds.targets.rasterize import labels_to_targets

EIGHT = np.ones((3, 3), dtype=bool)

# Причины потери GT-объекта оракулом.
OK = "ок"
LOST_GRID = "вне сетки"
LOST_VALID = "valid/nodata"
LOST_MIN_AREA = "min_area"
LOST_MERGE = "слияние"
LOST_NMS = "nms"
LOST_CONTOUR = "контур"


@dataclass
class OracleReport:
    """Стадии потерь по одному участку (или суммарно по нескольким)."""

    gt_total: int = 0
    gt_offgrid: int = 0  # «вне сетки»; при ignore_offgrid исключены из матчинга
    raster_zero: int = 0  # маска пуста после растеризации
    raster_below_min_area: int = 0  # маска есть, но < min_area_px
    valid_killed: int = 0  # вырезаны маской валидности
    merged_components: int = 0  # компонент с > 1 GT
    merged_unresolved: int = 0  # GT в слитых компонентах, не восстановленных делением
    pre_nms_dets: int = 0
    post_nms_dets: int = 0
    nms_suppressed: int = 0
    match: MatchResult = field(default_factory=MatchResult)
    reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "gt_total": self.gt_total,
            "gt_offgrid": self.gt_offgrid,
            "raster_zero": self.raster_zero,
            "raster_below_min_area": self.raster_below_min_area,
            "valid_killed": self.valid_killed,
            "merged_components": self.merged_components,
            "merged_unresolved": self.merged_unresolved,
            "pre_nms_dets": self.pre_nms_dets,
            "post_nms_dets": self.post_nms_dets,
            "nms_suppressed": self.nms_suppressed,
            "match": self.match.to_dict(),
            "reasons": dict(self.reasons),
        }


def _rasterize_detection(det_geom: np.ndarray, grid: Grid, shape: tuple[int, int]) -> np.ndarray:
    """Кольцо детекции (мир) -> булева маска по центрам пикселей Grid."""
    from expds.targets.rasterize import points_in_rings

    ring = np.asarray(det_geom, dtype=np.float64)
    col = (ring[:, 0] - grid.x_min) / grid.pixel_size - 0.5
    row = (ring[:, 1] - grid.y_min) / grid.pixel_size - 0.5
    c0 = max(int(np.floor(col.min())), 0)
    c1 = min(int(np.ceil(col.max())), shape[1] - 1)
    r0 = max(int(np.floor(row.min())), 0)
    r1 = min(int(np.ceil(row.max())), shape[0] - 1)
    mask = np.zeros(shape, dtype=bool)
    if c0 > c1 or r0 > r1:
        return mask
    cols, rows = np.meshgrid(
        np.arange(c0, c1 + 1, dtype=np.float64), np.arange(r0, r1 + 1, dtype=np.float64)
    )
    inside = points_in_rings(cols, rows, (np.column_stack([col, row]),))
    mask[r0 : r1 + 1, c0 : c1 + 1] = inside
    return mask


def oracle_report(
    feats: Sequence[LabelFeature],
    grid: Grid,
    classes: Sequence[str],
    post: PostConfig,
    sigma_px: float,
    valid: np.ndarray | None = None,
    eval_cfg: EvalConfig | None = None,
) -> OracleReport:
    """GT-фичи участка -> отчёт о потерях по стадиям.

    Args:
        feats: объекты разметки (LabelFeature с заполненными pixel-кольцами,
            как их даёт read_labels).
        grid: сетка участка.
        classes: список классов (порядок каналов таргетов).
        post: конфиг постобработки — тот же, с которым будет учиться модель.
        sigma_px: сигма heatmap (targets.heatmap_sigma_px).
        valid: маска валидности участка (SiteData.valid), None — вся валидна.
        eval_cfg: конфиг матчинга (eval-секция), по умолчанию IoU 0.5.
    """
    report = OracleReport()
    classes = list(classes)
    class_set = set(classes)
    poly_feats = [f for f in feats if f.cls in class_set and f.geom_type == "Polygon"]
    point_feats = [f for f in feats if f.cls in class_set and f.geom_type == "Point"]
    gts = objects_from_labels([f for f in feats if f.cls in class_set])
    report.gt_total = len(gts)
    if not gts:
        return report

    y_seg, y_hm = labels_to_targets(poly_feats + point_feats, grid, classes, sigma_px)

    # Маска валидности после того же закрытия, что в postprocess.
    if valid is not None:
        m = ndimage.binary_closing(valid, structure=EIGHT, iterations=2)
    else:
        m = None

    probs = np.concatenate([y_seg, y_hm]).astype(np.float32)
    if m is not None:
        probs = probs * m.astype(probs.dtype)[None]

    # --- стадия: растеризация и valid --------------------------------------
    # Маски в том же порядке, что gts (фильтр по классам, исходный порядок).
    feats_in = [f for f in feats if f.cls in class_set]
    obj_masks: list[np.ndarray | None] = []
    obj_mask_raw: list[np.ndarray | None] = []
    for f in feats_in:
        y1, y_hm1 = labels_to_targets([f], grid, classes, sigma_px)
        ci = classes.index(f.cls)
        if f.geom_type == "Polygon":
            raw = y1[ci] > 0
        else:
            # Точка попадает только в heatmap-канал своего класса: в y_seg
            # у неё пикселей нет (finds_points: 0 px / N hm). Раньше здесь
            # читали y_seg[0] — каждая точка и каждый класс с индексом > 0
            # получали пустую маску и падали в «вне сетки».
            raw = y_hm1[ci] > 0.01  # гауссиана: пиксели со сколь-нибудь значимым сигналом
        om = raw & m if m is not None else raw
        obj_mask_raw.append(raw if raw.any() else None)
        obj_masks.append(om if om.any() else None)

    below_area = 0
    for i, f in enumerate(feats_in):
        om = obj_masks[i]
        if om is None or f.geom_type != "Polygon":
            continue
        if int(om.sum()) < post.min_area_px:
            below_area += 1
    report.raster_below_min_area = below_area

    # --- стадия: компоненты и деление ---------------------------------------
    binary = probs[0] >= post.seg_thr
    lab, n_comp = ndimage.label(binary, structure=EIGHT)
    comps_of: list[set[int]] = []
    for om in obj_masks:
        if om is None:
            comps_of.append(set())
            continue
        comps_of.append(set(np.unique(lab[om]).tolist()) - {0})
    comp_members: dict[int, list[int]] = {}
    for gi, ids in enumerate(comps_of):
        for c in ids:
            comp_members.setdefault(c, []).append(gi)
    report.merged_components = sum(1 for v in comp_members.values() if len(v) > 1)

    # --- детекции: до NMS и после -------------------------------------------
    # Детекции оракула — ЩЕДРЫЙ потолок пайплайна: обе ветки для всех классов
    # (идеальная модель умеет и сег-блоб, и пик). Прод-конфиг может резать
    # ветки — это его право, но гейт должен мерить возможности post+матчинга.
    # NMS раздельный: круги точек не давят друг друга радиусом полигонов.
    dets_poly: list = []
    dets_hm: list = []
    k = len(classes)
    for i, cls in enumerate(classes):
        dets_poly += seg_to_polygons(probs[i], grid, cls, post, hm=probs[k + i])
        dets_hm += heatmap_to_points(probs[k + i], grid, cls, post)
    report.pre_nms_dets = len(dets_poly) + len(dets_hm)
    point_r = (
        post.point_nms_radius_px if post.point_nms_radius_px is not None else post.nms_radius_px
    )
    dets_after = nms_centroids(dets_poly, post.nms_radius_px * grid.pixel_size)
    dets_after += nms_centroids(dets_hm, point_r * grid.pixel_size)
    # Склейка веток BOTH — как в postprocess.
    dets_after = nms_centroids(dets_after, min(post.nms_radius_px, point_r) * grid.pixel_size)
    report.post_nms_dets = len(dets_after)
    report.nms_suppressed = report.pre_nms_dets - report.post_nms_dets
    kept_ids = {id(d) for d in dets_after}
    pre_by_id = {id(d): d for d in [*dets_poly, *dets_hm]}

    # --- матчинг -------------------------------------------------------------
    # GT без единого пикселя в сетке (om_raw is None) недетектируем в принципе.
    # При ignore_offgrid исключаем их из матчинга: они не должны считать FN
    # и топить гейт обучения (аудит 2026-11-11: 456/461 FN — «вне сетки»).
    ignore_offgrid = bool(eval_cfg.ignore_offgrid) if eval_cfg is not None else False
    if ignore_offgrid:
        keep_idx = [gi for gi, raw in enumerate(obj_mask_raw) if raw is not None]
        gts_eval = [gts[gi] for gi in keep_idx]
    else:
        keep_idx = list(range(len(gts)))
        gts_eval = list(gts)
    preds = objects_from_detections(dets_after)
    res = evaluate(preds, gts_eval, eval_cfg)
    report.match = res[ALL]
    # Индексы пар — по gts_eval; возвращаем к индексам полного gts.
    matched_gt = {keep_idx[gi] for _, gi, _ in res[ALL].pairs}

    # --- причины потерь ------------------------------------------------------
    for gi, gt in enumerate(gts):
        if gi in matched_gt:
            report.reasons[OK] = report.reasons.get(OK, 0) + 1
            continue
        om = obj_masks[gi]
        om_raw = obj_mask_raw[gi]
        if om is None:
            if om_raw is not None:
                # пиксели были, но их вырезала маска валидности
                report.reasons[LOST_VALID] = report.reasons.get(LOST_VALID, 0) + 1
                report.valid_killed += 1
            else:
                report.reasons[LOST_GRID] = report.reasons.get(LOST_GRID, 0) + 1
                report.raster_zero += 1
                if ignore_offgrid:
                    # исключены из матчинга выше — отдельно считаем для отчёта
                    report.gt_offgrid += 1
            continue
        is_poly = feats_in[gi].geom_type == "Polygon"
        if is_poly and int(om.sum()) < post.min_area_px:
            report.reasons[LOST_MIN_AREA] = report.reasons.get(LOST_MIN_AREA, 0) + 1
            continue
        # объект в слитой компоненте, деление не выделило
        if any(len(comp_members.get(c, [])) > 1 for c in comps_of[gi]):
            report.reasons[LOST_MERGE] = report.reasons.get(LOST_MERGE, 0) + 1
            report.merged_unresolved += 1
            continue
        # был ли полигон до NMS, которого больше нет после
        suppressed = False
        for d_id, d in pre_by_id.items():
            if d_id in kept_ids or d.cls != gt.cls:
                continue
            dm = _rasterize_detection(d.geometry, grid, lab.shape)
            inter = int((dm & om).sum())
            if inter >= 0.5 * int(om.sum()):
                suppressed = True
                break
        if suppressed:
            report.reasons[LOST_NMS] = report.reasons.get(LOST_NMS, 0) + 1
            continue
        report.reasons[LOST_CONTOUR] = report.reasons.get(LOST_CONTOUR, 0) + 1
    return report
