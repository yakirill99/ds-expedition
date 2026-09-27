import numpy as np
import pytest

from expds.features.grid import Grid
from expds.io.geojson import Detection, read_labels, ring_centroid, write_detections
from expds.post import (
    PostConfig,
    heatmap_to_points,
    nms_centroids,
    postprocess,
    seg_to_polygons,
    simplify_ring,
)

PS, X0, Y0 = 0.5, 500000.0, 6200000.0
H, W = 128, 160


def _grid(h=H, w=W):
    return Grid(crs="EPSG:32637", pixel_size=PS, x_min=X0, y_min=Y0, width=w, height=h)


def _world(r, c):
    return np.array([X0 + (c + 0.5) * PS, Y0 + (r + 0.5) * PS])


def _dist(r, c, h=H, w=W):
    rr, cc = np.mgrid[:h, :w]
    return np.hypot(rr - r, cc - c)


def _area(ring):
    x, y = ring[:, 0] - ring[0, 0], ring[:, 1] - ring[0, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def test_config_rejects_bad():
    with pytest.raises(ValueError):
        PostConfig.from_dict({"seg_thr": 0.5, "typo": 1})
    with pytest.raises(ValueError):
        PostConfig.from_dict({"peak_window_px": 4})


@pytest.mark.parametrize(("smooth", "simplify", "tol"), [(False, 0.5, 0.05), (True, 0.05, 0.01)])
def test_disk_polygon(smooth, simplify, tol):
    r0, c0, rad = 60.0, 70.0, 20.0
    d = _dist(r0, c0)
    prob = 1 / (1 + np.exp(-2 * (rad - d))) if smooth else (d <= rad).astype(np.float64)
    dets = seg_to_polygons(
        prob.astype(np.float32), _grid(), "mound", PostConfig(simplify_px=simplify)
    )
    assert len(dets) == 1
    ring = dets[0].geometry
    assert not np.array_equal(ring[0], ring[-1])  # контракт: без замыкающей точки
    area = _area(ring)
    assert area > 0  # CCW
    assert abs(area / (np.pi * (rad * PS) ** 2) - 1) < tol
    assert np.hypot(*(np.asarray(ring_centroid(ring)) - _world(r0, c0))) < 0.5 * PS
    assert dets[0].cls == "mound" and 0.5 < dets[0].score <= 1.0


def test_components_and_min_area():
    prob = ((_dist(30, 30) <= 10) | (_dist(90, 120) <= 12) | (_dist(100, 20) <= 1.5)).astype(
        np.float32
    )
    dets = seg_to_polygons(prob, _grid(), "mound", PostConfig(min_area_px=16))
    assert len(dets) == 2


def test_simplify_square():
    side = np.linspace(0, 10, 11)
    sq = np.vstack(
        [
            np.column_stack([side, np.zeros(11)])[:-1],
            np.column_stack([np.full(11, 10.0), side])[:-1],
            np.column_stack([side[::-1], np.full(11, 10.0)])[:-1],
            np.column_stack([np.zeros(11), side[::-1]])[:-1],
        ]
    )
    ring = np.vstack([sq, sq[:1]])
    s = simplify_ring(ring, 0.1)
    assert len(s) == 5 and np.allclose(s[0], s[-1])
    assert abs(abs(_area(s)) - 100.0) < 1e-9
    assert len(simplify_ring(ring, 0.0)) == len(ring)


def test_heatmap_subpixel_peak():
    r0, c0, sigma = 37.3, 52.7, 3.0
    hm = np.exp(-(_dist(r0, c0) ** 2) / (2 * sigma**2)).astype(np.float32)
    dets = heatmap_to_points(hm, _grid(), "mound", PostConfig())
    assert len(dets) == 1
    assert np.hypot(*(np.asarray(dets[0].centroid) - _world(r0, c0))) < 0.05 * PS
    assert dets[0].score > 0.9
    assert len(dets[0].geometry) == PostConfig().point_vertices
    assert heatmap_to_points(hm * 0.2, _grid(), "mound", PostConfig(hm_thr=0.3)) == []


def _det(cls, score, x, y):
    ring = np.array([[x - 1, y - 1], [x + 1, y - 1], [x + 1, y + 1], [x - 1, y + 1]])
    return Detection(cls=cls, score=score, geom_type="Polygon", geometry=ring)


def test_nms():
    a, b = _det("mound", 0.9, 0, 0), _det("mound", 0.8, 1, 0)
    c, far = _det("ditch", 0.7, 1, 0), _det("mound", 0.6, 100, 0)
    out = nms_centroids([b, a, c, far], radius=3.0)
    assert [(d.cls, d.score) for d in out] == [("mound", 0.9), ("ditch", 0.7), ("mound", 0.6)]
    assert len(nms_centroids([a, b, c], radius=3.0, per_class=False)) == 1


def test_postprocess_heads_and_valid():
    classes = ("mound", "ditch")
    probs = np.zeros((4, H, W), np.float32)
    probs[0] = _dist(40, 40) <= 15
    probs[3] = np.exp(-(_dist(100, 120) ** 2) / 18)
    dets = postprocess(probs, _grid(), classes, PostConfig())
    assert sorted(d.cls for d in dets) == ["ditch", "mound"]
    valid = np.ones((H, W), bool)
    valid[80:, :] = False
    dets = postprocess(probs, _grid(), classes, PostConfig(), valid=valid)
    assert [d.cls for d in dets] == ["mound"]
    with pytest.raises(ValueError):
        postprocess(probs[:3], _grid(), classes)


def test_postprocess_nms_merges_seg_and_point():
    probs = np.zeros((2, H, W), np.float32)
    probs[0] = _dist(60, 60) <= 5
    probs[1] = np.exp(-(_dist(60.5, 60) ** 2) / 18)
    dets = postprocess(probs, _grid(), ("mound",), PostConfig(nms_radius_px=6))
    assert len(dets) == 1


def test_geojson_roundtrip(tmp_path):
    prob = (_dist(60, 70) <= 20).astype(np.float32)
    grid = _grid()
    dets = seg_to_polygons(prob, grid, "mound", PostConfig())
    p = tmp_path / "pred.geojson"
    write_detections(p, dets, grid.crs, "EPSG:3857")
    feats = read_labels(p, grid, "EPSG:3857")
    assert len(feats) == 1 and feats[0].cls == "mound"
    assert np.hypot(*(np.asarray(feats[0].centroid_world) - _world(60, 70))) < 0.5 * PS


def test_postprocess_max_detections():
    probs = np.zeros((2, H, W), np.float32)
    rr, cc = np.mgrid[:H, :W]
    probs[1] = ((rr % 16 == 8) & (cc % 16 == 8)) * np.linspace(0.4, 0.9, W)[None, :]
    all_dets = postprocess(probs, _grid(), ("mound",), PostConfig())
    top = postprocess(probs, _grid(), ("mound",), PostConfig(max_detections=5))
    assert len(all_dets) == 80 and len(top) == 5
    assert [d.score for d in top] == [d.score for d in all_dets[:5]]
