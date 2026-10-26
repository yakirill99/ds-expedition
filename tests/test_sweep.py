"""Тесты свипа постобработки: кэш probs, перебор сетки, таблица и YAML."""

import json

import numpy as np
import pytest

from expds.eval.metric import EvalConfig, EvalObject
from expds.features.grid import Grid
from expds.post.sweep import (
    PostGrid,
    ProbsCache,
    best_post_yaml,
    format_sweep_table,
    load_probs_cache,
    save_probs_cache,
    sweep_postparams,
    sweep_report,
)
from expds.post.vectorize import PostConfig

H = W = 64


def _grid(x_min=0.0, y_min=0.0):
    return Grid(crs="EPSG:32637", pixel_size=1.0, x_min=x_min, y_min=y_min, width=W, height=H)


def _circle_probs(r0=32.0, c0=32.0, radius=10.0, edge=2.0):
    """(2, H, W): seg-канал — круг с 0.5-уровнем ровно на radius, hm — нули."""
    rr, cc = np.mgrid[:H, :W]
    d = np.hypot(rr - r0, cc - c0)
    probs = np.zeros((2, H, W), np.float32)
    probs[0] = np.clip((radius + edge / 2 - d) / edge, 0.0, 1.0)
    return probs


def _gt_square(x0=22.5, y0=22.5, side=20.0, cls="mound"):
    """GT-квадрат в мировых координатах (IoU с кругом радиуса 10 ~ 0.785 > 0.5)."""
    ring = np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]])
    return EvalObject(cls, "Polygon", (ring,))


def _cache(name, grid, probs=None, valid=None):
    """ProbsCache напрямую (без файлов) — постпроцесс и метрика работают без диска."""
    return ProbsCache(
        site=name, probs=probs if probs is not None else _circle_probs(), grid=grid, valid=valid
    )


def test_cache_roundtrip(tmp_path):
    """save/load: probs, grid, valid и тег чекпойнта сохраняются."""
    p = tmp_path / "site.npz"
    grid = _grid(x_min=123.0, y_min=456.0)
    valid = np.ones((H, W), bool)
    valid[:, 32:] = False
    save_probs_cache(p, "site", _circle_probs(), grid, valid, weights_tag="deadbeef1234")
    cache = load_probs_cache(p)
    assert cache.site == "site" and cache.weights_tag == "deadbeef1234"
    assert np.allclose(cache.probs, _circle_probs())
    assert cache.grid == grid
    assert cache.valid is not None and np.array_equal(cache.valid, valid)


def test_cache_roundtrip_without_valid(tmp_path):
    p = tmp_path / "site.npz"
    save_probs_cache(p, "s", _circle_probs(), _grid())
    cache = load_probs_cache(p)
    assert cache.valid is None and cache.weights_tag == ""


def test_sweep_finds_working_params():
    """Комбинация с проходимым min_area даёт F1 1.0 и стоит первой; без неё — F1 0."""
    caches = {"site_a": _cache("site_a", _grid())}
    gts = {"site_a": [_gt_square()]}
    pgrid = PostGrid(
        seg_thr=(0.5,), hm_thr=(0.9,), min_area_px=(16.0, 100000.0), nms_radius_px=(6.0,)
    )
    rows = sweep_postparams(caches, gts, ("mound",), PostConfig(), EvalConfig(), pgrid)
    assert len(rows) == 2
    assert rows[0].f1 == 1.0 and (rows[0].tp, rows[0].fp, rows[0].fn) == (1, 0, 0)
    assert rows[0].params["min_area_px"] == 16.0
    assert rows[0].per_class_f1 == {"mound": 1.0}
    assert rows[1].f1 == 0.0 and rows[1].fn == 1  # круг отфильтрован min_area
    assert [r.f1 for r in rows] == sorted((r.f1 for r in rows), reverse=True)


def test_sweep_sums_counters_over_sites():
    """Счётчики складываются по участкам (как в train): 2 участка -> 2 TP."""
    caches = {
        "a": _cache("a", _grid(x_min=0.0)),
        "b": _cache("b", _grid(x_min=1000.0)),
    }
    gts = {
        "a": [_gt_square()],
        "b": [_gt_square(x0=1022.5)],  # тот же локальный круг, но в сетке участка b
    }
    pgrid = PostGrid(seg_thr=(0.5,), hm_thr=(0.9,), min_area_px=(16.0,), nms_radius_px=(6.0,))
    rows = sweep_postparams(caches, gts, ("mound",), PostConfig(), EvalConfig(), pgrid)
    assert rows[0].tp == 2 and rows[0].f1 == 1.0


def test_sweep_requires_same_sites():
    caches = {"a": _cache("a", _grid())}
    with pytest.raises(ValueError, match="не совпадают"):
        sweep_postparams(caches, {"b": [_gt_square()]}, ("mound",), PostConfig(), EvalConfig())


def test_format_sweep_table_and_yaml():
    caches = {"site_a": _cache("site_a", _grid())}
    gts = {"site_a": [_gt_square()]}
    pgrid = PostGrid(seg_thr=(0.5, 0.7), hm_thr=(0.9,), min_area_px=(16.0,), nms_radius_px=(6.0,))
    rows = sweep_postparams(caches, gts, ("mound",), PostConfig(), EvalConfig(), pgrid)

    table = format_sweep_table(rows, top=1)
    lines = table.splitlines()
    assert "seg_thr" in lines[0] and len(lines) == 3  # шапка, разделитель, одна строка
    assert "1.000" in table

    snippet = best_post_yaml(rows, PostConfig())
    assert "post:" in snippet and "seg_thr: 0.5" in snippet
    assert "simplify_px: 0.5" in snippet  # нескользируемые поля из базового конфига


def test_sweep_report_json_serializable():
    caches = {"site_a": _cache("site_a", _grid())}
    gts = {"site_a": [_gt_square()]}
    pgrid = PostGrid(seg_thr=(0.5,), hm_thr=(0.9,), min_area_px=(16.0,), nms_radius_px=(6.0,))
    rows = sweep_postparams(caches, gts, ("mound",), PostConfig(), EvalConfig(), pgrid)
    report = sweep_report(rows, EvalConfig(), pgrid)
    assert report["n_combos"] == 1 and report["rows"][0]["f1"] == 1.0
    json.dumps(report)  # сериализуемо
