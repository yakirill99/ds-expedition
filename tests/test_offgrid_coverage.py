"""Платформенный (coverage) матчинг и отсечение GT вне сетки.

Аудит 2026-11-11 (docs/issues/07-grid-coverage.md): oracle gate блокировал
обучение из-за 456/461 FN «вне сетки» — GT, которых нет на растре. Плюс
строгий IoU 0.5 браковал watershed-доли слитых курганов и тонкие стены
фортификаций, которые платформа (ТР КОЗ-3, мягкий cov-матчинг) засчитывает.
"""

import numpy as np
import pytest

from expds.eval.metric import (
    EvalConfig,
    EvalObject,
    match,
    polygon_coverage,
    polygon_iou,
)
from expds.eval.oracle import LOST_GRID, OK, oracle_report
from expds.features.grid import Grid
from expds.io.geojson import LabelFeature
from expds.post.vectorize import PostConfig

PS = 0.5
X0, Y0 = 500000.0, 6200000.0
H, W = 128, 128


def _grid():
    return Grid(crs="EPSG:32637", pixel_size=PS, x_min=X0, y_min=Y0, width=W, height=H)


def _rect(x0, y0, x1, y1):
    """Кольцо-прямоугольник по мировым координатам (CCW, без замыкания)."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _px(grid, ring):
    return np.column_stack(
        [
            (ring[:, 0] - grid.x_min) / grid.pixel_size - 0.5,
            (ring[:, 1] - grid.y_min) / grid.pixel_size - 0.5,
        ]
    )


def _feat(grid, ring, cls="k"):
    return LabelFeature(
        cls=cls, geom_type="Polygon", world=(ring,), pixel=(_px(grid, ring),), properties={}
    )


def _obj(ring, cls="k"):
    return EvalObject(cls, "Polygon", (ring,))


def _annulus(cx, cy, r_out, r_in, n=48):
    """(внешнее кольцо, дыра) —_GT-«вал»."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([cx + r_out * np.cos(t), cy + r_out * np.sin(t)])
    inner = np.column_stack([cx + r_in * np.cos(t), cy + r_in * np.sin(t)])[::-1]
    return outer, inner


# --- coverage-матчинг -----------------------------------------------------------


def test_coverage_matches_half_overlap_that_iou_rejects():
    """Pred = узкая полоса вдоль края GT: IoU < 0.5, cov = 1.0.

    Ровно случай watershed-доли слитого кургана: платформа (cov по
    min-площади) засчитывает, строгий IoU — нет.
    """
    gt = _rect(0.0, 0.0, 20.0, 20.0)
    pred = _rect(0.0, 0.0, 20.0, 9.0)  # 45% площади GT
    assert polygon_iou((pred,), (gt,)) < 0.5
    assert polygon_coverage((pred,), (gt,)) == pytest.approx(1.0)
    iou_res = match([_obj(pred)], [_obj(gt)], EvalConfig(match_mode="iou"))
    cov_res = match([_obj(pred)], [_obj(gt)], EvalConfig(match_mode="coverage"))
    assert iou_res.tp == 0 and iou_res.fn == 1
    assert cov_res.tp == 1 and cov_res.fn == 0


def test_coverage_rep_point_rule():
    """cov >= 0.5 без внутренней точки pred — не матч (правило ТР КОЗ-3).

    GT — кольцо (вал), pred — сплошной диск, затыкающий дыру: пересечение
    большое, но представительская точка диска лежит в дыре, вне GT.
    """
    outer, inner = _annulus(10.0, 10.0, 10.0, 5.0)
    disc = _rect(0.0, 0.0, 20.0, 20.0)
    # cov = площадь кольца / площадь диска >= 0.5
    assert polygon_coverage((disc,), (outer, inner)) >= 0.5
    gt_ring = _obj_rings(outer, inner)
    res = match([_obj(disc)], [gt_ring], EvalConfig())
    assert res.tp == 1  # IoU 0.59 >= 0.5 — строгий режим пропускает
    cov_res = match([_obj(disc)], [gt_ring], EvalConfig(match_mode="coverage"))
    assert cov_res.tp == 0 and cov_res.fn == 1


def _obj_rings(outer, inner, cls="k"):
    return EvalObject(cls, "Polygon", (outer, inner))


# --- выборка пересечения тонких контуров ----------------------------------------


def test_thin_wall_sampling_not_missed():
    """Стена 1 px x 500 px: выборка пересечения не должна «мазать» контур.

    fortifikatsii: изотропная сетка с шагом > ширины стены давала
    inter = 0 (tp 0 из 69 на night3).
    """
    wall = _rect(0.0, 0.0, 250.0, 0.5)  # 500 x 1 px при PS=0.5
    same = _rect(0.0, 0.0, 250.0, 0.5)
    assert polygon_iou((same,), (wall,), samples=128) > 0.9
    # сдвиг на четверть ширины — пересечение всё ещё честное
    shifted = _rect(0.0, 0.125, 250.0, 0.625)
    iou = polygon_iou((shifted,), (wall,), samples=128)
    assert 0.3 < iou <= 1.0
    cov = polygon_coverage((shifted,), (wall,), samples=128)
    assert cov == pytest.approx(iou, rel=0.35)


def test_diagonal_wall_sampling_not_missed():
    """Диагональная стена 45°: bbox-сетка не попадала в контур вовсе.

    Стена шириной 1 px под 45°: вероятность попадания равномерной сетки
    по bbox ~ ширина/сторона bbox < 0.5%; fortifikatsii почти всегда
    неосевые. Сканлайн-выборка лежит внутри по построению.
    """
    c = np.array([100.0, 100.0])
    axis = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    perp = np.array([-axis[1], axis[0]])
    L, w = 283.0, 1.0

    def rot_rect(shift=0.0):
        cc = c + perp * shift
        return np.array(
            [
                cc + axis * L / 2 + perp * w / 2,
                cc - axis * L / 2 + perp * w / 2,
                cc - axis * L / 2 - perp * w / 2,
                cc + axis * L / 2 - perp * w / 2,
            ]
        )

    wall = rot_rect()
    same = rot_rect()
    assert polygon_iou((same,), (wall,), samples=128) > 0.9
    # сдвиг на 20% ширины: cov ~0.8, IoU ~0.67 — обе метрики честные
    shifted = rot_rect(shift=0.2)
    cov = polygon_coverage((shifted,), (wall,), samples=128)
    iou = polygon_iou((shifted,), (wall,), samples=128)
    assert 0.6 < cov <= 1.0
    assert 0.4 < iou < cov


# --- ignore_offgrid --------------------------------------------------------------


def test_eval_config_new_keys():
    cfg = EvalConfig.from_dict({"match_mode": "coverage", "cov_thr": 0.6, "ignore_offgrid": True})
    assert cfg.match_mode == "coverage"
    assert cfg.cov_thr == 0.6
    assert cfg.ignore_offgrid is True
    with pytest.raises(ValueError, match="match_mode"):
        EvalConfig.from_dict({"match_mode": "dice"})
    with pytest.raises(ValueError, match="cov_thr"):
        EvalConfig.from_dict({"cov_thr": 0.0})


def test_oracle_ignore_offgrid_excludes_from_fn():
    """GT вне сетки: без флага — FN, с флагом — исключён из матчинга, но в отчёте."""
    grid = _grid()
    inside = _feat(grid, _rect(X0 + 5.0, Y0 + 5.0, X0 + 25.0, Y0 + 25.0))
    outside = _feat(grid, _rect(X0 - 30.0, Y0 + 5.0, X0 - 10.0, Y0 + 25.0))
    post = PostConfig(min_area_px=6.0, hm_thr=0.15, peak_window_px=7)

    rep_strict = oracle_report([inside, outside], grid, ["k"], post, sigma_px=3.0)
    assert rep_strict.match.tp == 1 and rep_strict.match.fn == 1

    rep = oracle_report(
        [inside, outside],
        grid,
        ["k"],
        post,
        sigma_px=3.0,
        eval_cfg=EvalConfig(ignore_offgrid=True),
    )
    assert rep.gt_total == 2
    assert rep.gt_offgrid == 1
    assert rep.match.tp == 1 and rep.match.fn == 0 and rep.match.fp == 0
    assert rep.reasons.get(LOST_GRID, 0) == 1
    assert rep.reasons.get(OK, 0) == 1


def test_offgrid_keep_mask():
    """Маска пересечения сетки: полный промах отсечён, частичный — остался."""
    from expds.train import _offgrid_keep_mask

    grid = _grid()
    x1, y1 = X0 + W * PS, Y0 + H * PS
    inside = _obj(_rect(X0 + 5, Y0 + 5, X0 + 15, Y0 + 15))
    partial = _obj(_rect(x1 - 5.0, Y0 + 5, x1 + 15.0, Y0 + 15))  # за правым краем наполовину
    outside = _obj(_rect(x1 + 50.0, y1 + 50.0, x1 + 60.0, y1 + 60.0))
    point_in = EvalObject("p", "Point", (np.array([[X0 + 10.0, Y0 + 10.0]]),))
    point_out = EvalObject("p", "Point", (np.array([[x1 + 100.0, y1 + 100.0]]),))
    gts = [inside, partial, outside, point_in, point_out]
    keep = _offgrid_keep_mask(gts, grid, grid.crs, r_tol=5.0, r_tol_units="crs")
    assert keep.tolist() == [True, True, False, True, False]
