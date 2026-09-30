import numpy as np
import pytest

from expds.eval import (
    EvalConfig,
    EvalObject,
    evaluate,
    evaluate_files,
    match,
    objects_from_detections,
    polygon_iou,
)
from expds.io.geojson import Detection, write_detections

X0, Y0 = 4_200_000.0, 7_500_000.0  # порядок величин EPSG:3857


def _sq(x, y, s):
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float64) + [X0, Y0]


def _circle(x, y, r, n=32):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([x + r * np.cos(t), y + r * np.sin(t)]) + [X0, Y0]


def _poly(ring, cls="rampart", score=1.0, holes=()):
    return EvalObject(cls, "Polygon", (ring, *holes), score)


def _pt(x, y, cls="mound"):
    return EvalObject(cls, "Point", (np.array([[X0 + x, Y0 + y]]),))


def _gts():
    return [
        _poly(_sq(0, 0, 20)),
        _poly(_sq(100, 0, 30), cls="ditch"),
        _pt(50, 50),
        _pt(200, 200),
    ]


def _perfect_preds():
    return [
        _poly(_sq(0, 0, 20)),
        _poly(_sq(100, 0, 30), cls="ditch"),
        _poly(_circle(50, 50, 2), cls="mound"),
        _poly(_circle(200, 200, 2), cls="mound"),
    ]


def test_config_rejects_bad():
    with pytest.raises(ValueError):
        EvalConfig.from_dict({"iou_thr": 0.5, "typo": 1})
    with pytest.raises(ValueError):
        EvalConfig.from_dict({"r_tol_units": "feet"})


def test_polygon_iou_accuracy():
    a, b = _sq(0, 0, 10), _sq(5, 0, 10)
    assert abs(polygon_iou([a], [b]) - 1 / 3) < 0.01
    assert polygon_iou([a], [a]) > 0.99
    assert polygon_iou([a], [_sq(50, 50, 10)]) == 0.0
    hole = _sq(2, 2, 6)[::-1]
    assert polygon_iou([a, hole], [a, hole]) > 0.98
    assert abs(polygon_iou([a, hole], [a]) - 64 / 100) < 0.02


def test_perfect():
    r = evaluate(_perfect_preds(), _gts())
    assert (r["__all__"].tp, r["__all__"].fp, r["__all__"].fn) == (4, 0, 0)
    assert r["__all__"].f1 == 1.0
    assert r["mound"].tp == 2 and r["ditch"].tp == 1


def test_empty():
    r = match([], _gts())
    assert (r.tp, r.fp, r.fn, r.f1) == (0, 0, 4, 0.0)
    r = match(_perfect_preds(), [])
    assert (r.tp, r.fp, r.fn, r.f1) == (0, 4, 0, 0.0)
    assert match([], []).f1 == 1.0


def test_one_big_polygon():
    big = [_poly(_sq(-10, -10, 300), cls="rampart")]
    r = match(big, _gts(), EvalConfig(class_agnostic=True))
    assert r.tp == 0 and r.fp == 1 and r.fn == 4


@pytest.mark.parametrize(("dx", "ok"), [(4.9, True), (5.1, False)])
def test_point_tolerance(dx, ok):
    pred = [_poly(_circle(50 + dx, 50, 2), cls="mound")]
    r = match(pred, [_pt(50, 50)], EvalConfig(r_tol=5.0))
    assert r.tp == int(ok)


@pytest.mark.parametrize(("dx", "ok"), [(1.0, True), (15.0, False)])
def test_polygon_shift(dx, ok):
    r = match([_poly(_sq(dx, 0, 20))], [_poly(_sq(0, 0, 20))], EvalConfig(iou_thr=0.5))
    assert r.tp == int(ok)


def test_duplicates_and_score_order():
    preds = [_poly(_sq(2, 0, 20), score=0.6), _poly(_sq(0, 0, 20), score=0.9)]
    r = match(preds, [_poly(_sq(0, 0, 20))])
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)
    assert r.pairs[0][0] == 1  # GT забрал pred с большим score


def test_nonconvex():
    c_shape = np.array(
        [[0, 0], [30, 0], [30, 10], [10, 10], [10, 20], [30, 20], [30, 30], [0, 30]], float
    ) + [X0, Y0]
    assert match([_poly(c_shape)], [_poly(c_shape)]).tp == 1
    hull = _sq(0, 0, 30)
    assert polygon_iou([hull], [c_shape]) == pytest.approx(700 / 900, abs=0.01)
    assert match([_poly(hull)], [_poly(c_shape)], EvalConfig(iou_thr=0.8)).tp == 0


def test_class_mismatch():
    pred = [_poly(_sq(0, 0, 20), cls="ditch")]
    assert match(pred, [_poly(_sq(0, 0, 20))]).tp == 0
    assert match(pred, [_poly(_sq(0, 0, 20))], EvalConfig(class_agnostic=True)).tp == 1


def test_r_tol_meters_web_mercator():
    lat = np.deg2rad(60.0)
    y60 = 6378137.0 * np.log(np.tan(np.pi / 4 + lat / 2)) - Y0  # масштаб 3857 на 60° = 2
    gt = [_pt(0, y60)]
    pred = [_poly(_circle(8, y60, 1), cls="mound")]
    assert match(pred, gt, EvalConfig(r_tol=5.0, r_tol_units="crs")).tp == 0
    assert match(pred, gt, EvalConfig(r_tol=5.0, r_tol_units="meters")).tp == 1


def test_evaluate_files_roundtrip(tmp_path):
    dets = [
        Detection(cls="rampart", score=0.9, geom_type="Polygon", geometry=_sq(0, 0, 20)),
        Detection(cls="mound", score=0.8, geom_type="Polygon", geometry=_circle(50, 50, 2)),
    ]
    p, g = tmp_path / "pred.geojson", tmp_path / "gt.geojson"
    write_detections(p, dets, "EPSG:3857", "EPSG:3857")
    write_detections(
        g,
        [
            Detection(cls="rampart", score=1.0, geom_type="Polygon", geometry=_sq(0, 0, 20)),
            Detection(
                cls="mound", score=1.0, geom_type="Point", geometry=np.array([[X0 + 50, Y0 + 50]])
            ),
        ],
        "EPSG:3857",
        "EPSG:3857",
    )
    r = evaluate_files(p, g)
    assert r["__all__"].to_dict()["f1"] == 1.0
    assert objects_from_detections(dets)[0].score == 0.9
