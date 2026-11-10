"""Плотные точечные кластеры не должны глотаться NMS/окном пиков.

Регрессия 2026-11-11: общий NMS (радиус под полигоны, 4-6 px) и общее
peak_window (7 px) уничтожали точки разметки, идущие через 2-4 м
(Тимерево: 312 FN «вне сетки», по факту — NMS). Синтетика: кластеры точек
с шагом 2..5 м двух классов + три полигона.

Проверяем:
  1. point-параметры (point_nms_radius_px/point_peak_window_px) спасают
     кластеры с шагом >= 2 px — oracle f1 = 1.0;
  2. без них потеря реальная, но атрибуция честная: точки имеют маску
     (heatmap-канал), причина — подавление, а не «вне сетки».
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from expds.eval.oracle import oracle_report
from expds.features.grid import Grid
from expds.io.geojson import LabelFeature
from expds.post.vectorize import PostConfig

CLASSES = ["kurgany_povrezhdennye", "finds_points"]
SIGMA = 3.0
POST = PostConfig(
    seg_thr=0.5,
    split_touching=True,
    split_min_distance_px=3,
    min_area_px=6,
    hm_thr=0.15,
    peak_window_px=7,
    point_radius_px=4.0,
    point_vertices=16,
    nms_radius_px=4.0,
    max_detections=2000,
    branches={"kurgany_povrezhdennye": "polygon", "finds_points": "point"},
)
GRID = Grid(crs="EPSG:32636", pixel_size=1.0, x_min=0.0, y_min=0.0, width=200, height=200)


def _poly(cls: str, x0: float, y0: float, r: float) -> LabelFeature:
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.column_stack([x0 + r * np.cos(t), y0 + r * np.sin(t)])
    return LabelFeature(cls=cls, geom_type="Polygon", world=(ring,), pixel=(ring.copy(),))


def _point(cls: str, x: float, y: float) -> LabelFeature:
    p = np.array([[x, y]], dtype=np.float64)
    return LabelFeature(cls=cls, geom_type="Point", world=(p,), pixel=(p.copy(),))


def _feats(spacing: float):
    feats = [_poly("kurgany_povrezhdennye", 40, 40, 6), _poly("kurgany_povrezhdennye", 40, 90, 5)]
    for i in range(9):
        feats.append(_point("kurgany_povrezhdennye", 30 + i * spacing, 130.0))
        feats.append(_point("finds_points", 30 + i * spacing, 150.0))
    return feats


def test_point_params_keep_dense_clusters():
    """point_nms 1.5 + окно 3: шаги 2-5 м восстанавливаются полностью."""
    post = replace(POST, point_nms_radius_px=1.5, point_peak_window_px=3)
    for spacing in (2.0, 3.0, 4.0, 5.0):
        rep = oracle_report(_feats(spacing), GRID, CLASSES, post, SIGMA)
        assert rep.match.fn == 0, f"шаг {spacing}: fn {rep.match.fn} ({rep.reasons})"
        assert rep.match.fp == 0
        assert rep.match.tp == rep.gt_total == 20


def test_default_params_lose_but_attribute_honestly():
    """Без point-параметров кластеры 2-4 м теряются, но не как «вне сетки»."""
    post = replace(POST, point_nms_radius_px=None, point_peak_window_px=None)
    for spacing in (2.0, 3.0, 4.0):
        rep = oracle_report(_feats(spacing), GRID, CLASSES, post, SIGMA)
        assert rep.match.fn > 0, f"шаг {spacing}: ожидали потери"
        assert rep.raster_zero == 0, (
            f"шаг {spacing}: точки не должны числиться «вне сетки», причины: {rep.reasons}"
        )
        # «вне сетки» исчезло из причин — точечные маски теперь из heatmap
        assert "вне сетки" not in rep.reasons
