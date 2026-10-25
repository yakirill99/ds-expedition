"""Тесты разбора ошибок: таксономия FP/FN и экспорт в GeoJSON для QGIS."""

import json

import numpy as np

from expds.eval.errors import analyze_errors, errors_to_geojson, format_error_report
from expds.eval.metric import EvalConfig, EvalObject, match

X0, Y0 = 4_200_000.0, 7_500_000.0
S = 20.0  # сторона квадрата; IoU квадрата со сдвигом d: (S-d)² / (2S² - (S-d)²)


def _sq(x, y, s=S):
    return np.array(
        [[X0 + x, Y0 + y], [X0 + x + s, Y0 + y], [X0 + x + s, Y0 + y + s], [X0 + x, Y0 + y + s]]
    )


def _poly(cls, score, x, y, s=S):
    return EvalObject(cls, "Polygon", (_sq(x, y, s),), score)


def _preds_gts():
    """Сцена на все типы FP и FN.

    preds: p0 TP; p1 дубль p0; p2 локализация (IoU ~0.09 с g0);
    p3 фон; p4 чужой класс поверх g1; p5 локализация у g2.
    gts: g0, g1, g2, g3 — g1 покрыт чужим классом, g2 — плохим предсказанием,
    g3 не покрыт ничем.
    """
    preds = [
        _poly("mound", 0.9, 0.0, 0.0),
        _poly("mound", 0.8, 2.0, 0.0),
        _poly("mound", 0.7, 12.0, 0.0),
        _poly("mound", 0.6, 800.0, 800.0),
        _poly("mound", 0.5, 200.0, 0.0),
        _poly("mound", 0.4, 418.0, 0.0),
    ]
    gts = [
        _poly("mound", 1.0, 0.0, 0.0),
        _poly("rampart", 1.0, 200.0, 0.0),
        _poly("mound", 1.0, 400.0, 0.0),
        _poly("mound", 1.0, 600.0, 600.0),
    ]
    return preds, gts


def test_analyze_fp_types():
    """Все четыре типа FP классифицируются по приоритету dup > loc > cls > bg."""
    preds, gts = _preds_gts()
    result = match(preds, gts, EvalConfig())
    assert (result.tp, result.fp, result.fn) == (1, 5, 3)
    report = analyze_errors(preds, gts, result)
    assert report.fp_types == {
        "duplicate": (1,),
        "localization": (2, 5),
        "class_confusion": (4,),
        "background": (3,),
    }
    assert report.counts_fp()["duplicate"] == 1


def test_analyze_fn_types():
    """FN покрытый предсказанием vs FN без предсказаний рядом."""
    preds, gts = _preds_gts()
    result = match(preds, gts, EvalConfig())
    report = analyze_errors(preds, gts, result)
    assert report.fn_types == {"covered": (1, 2), "missed": (3,)}


def test_point_fn_covered_by_pred_nearby():
    """Два GT-точки в одном месте, одно предсказание: второй GT — FN covered."""
    gt1 = EvalObject("mound", "Point", (np.array([[X0, Y0]]),))
    gt2 = EvalObject("mound", "Point", (np.array([[X0 + 0.5, Y0]]),))
    pred = EvalObject("mound", "Point", (np.array([[X0 + 1.0, Y0]]),))
    result = match([pred], [gt1, gt2], EvalConfig())
    assert (result.tp, result.fn) == (1, 1)
    report = analyze_errors([pred], [gt1, gt2], result)
    # жадный матчинг взял ближайший gt2 (индекс 1), FN — gt1 (индекс 0),
    # и она покрыта предсказанием (центроиды в r_tol)
    assert report.fn_types == {"covered": (0,), "missed": ()}


def test_no_preds_or_no_gts_edges():
    """Без предсказаний — все FN missed; без GT — все FP background."""
    preds, gts = _preds_gts()
    r_no_pred = match([], gts, EvalConfig())
    report = analyze_errors([], gts, r_no_pred)
    assert sum(report.counts_fn().values()) == 4
    assert report.counts_fn()["missed"] == 4

    r_no_gt = match(preds, [], EvalConfig())
    report = analyze_errors(preds, [], r_no_gt)
    assert report.counts_fp()["background"] == len(preds)


def test_errors_geojson(tmp_path):
    """GeoJSON: по объекту на каждый FP/FN, kind/error/class/score в properties."""
    preds, gts = _preds_gts()
    result = match(preds, gts, EvalConfig())
    report = analyze_errors(preds, gts, result)
    p = tmp_path / "errors.geojson"
    n = errors_to_geojson(p, preds, gts, report, crs="EPSG:3857")
    assert n == 5 + 3
    fc = json.loads(p.read_text(encoding="utf-8"))
    kinds = [f["properties"]["kind"] for f in fc["features"]]
    assert kinds.count("FP") == 5 and kinds.count("FN") == 3
    dup = next(f for f in fc["features"] if f["properties"]["error"] == "duplicate")
    assert dup["properties"]["class"] == "mound" and dup["properties"]["score"] == 0.8
    assert dup["geometry"]["type"] == "Polygon"
    assert fc["crs"]["properties"]["name"].endswith("EPSG::3857")
    fn_missed = next(f for f in fc["features"] if f["properties"]["error"] == "missed")
    # g3 — полигон: первая вершина внешнего кольца
    assert fn_missed["geometry"]["coordinates"][0][0] == [X0 + 600.0, Y0 + 600.0]


def test_format_error_report_lines():
    """Текстовый отчёт содержит все типы с числами."""
    preds, gts = _preds_gts()
    report = analyze_errors(preds, gts, match(preds, gts, EvalConfig()))
    text = format_error_report(report)
    for kind in ("duplicate", "localization", "class_confusion", "background", "covered", "missed"):
        assert kind in text
    assert "  duplicate: 1" in text
