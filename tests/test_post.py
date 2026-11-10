import json

import numpy as np
import pytest

from expds.eval.metric import EvalObject, polygon_iou
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


# --- дыры полигонов --------------------------------------------------------------


def test_seg_polygon_with_hole_smooth():
    """Гладкое кольцо: дыра находится, радиусы колец — по 0.5-уровню вероятности."""
    d = _dist(60.0, 70.0)
    prob = np.clip(1 - np.abs(d - 14.0) / 4.0, 0.0, 1.0).astype(np.float32)
    dets = seg_to_polygons(prob, _grid(), "mound", PostConfig())
    assert len(dets) == 1
    assert len(dets[0].holes) == 1
    ring, hole = dets[0].geometry, dets[0].holes[0]
    assert _area(ring) > 0 and _area(hole) < 0  # внешнее CCW, дыра CW
    assert abs(_area(ring) / (np.pi * (16.0 * PS) ** 2) - 1) < 0.03
    assert abs(abs(_area(hole)) / (np.pi * (12.0 * PS) ** 2) - 1) < 0.03


def test_seg_polygon_with_hole_binary():
    """Бинарное кольцо: у внутренней границы знак площади противоположен внешней."""
    d = _dist(60.0, 70.0)
    prob = ((d <= 20.0) & (d >= 8.0)).astype(np.float32)
    dets = seg_to_polygons(prob, _grid(), "mound", PostConfig())
    assert len(dets) == 1 and len(dets[0].holes) == 1
    assert _area(dets[0].geometry) > 0 and _area(dets[0].holes[0]) < 0


def test_solid_disk_has_no_holes():
    prob = (_dist(60, 70) <= 20).astype(np.float32)
    dets = seg_to_polygons(prob, _grid(), "mound", PostConfig())
    assert len(dets) == 1 and dets[0].holes == ()


def test_detection_rejects_bad_holes():
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="Point"):
        Detection(
            cls="mound", score=1.0, geom_type="Point", geometry=np.zeros((1, 2)), holes=(ring,)
        )
    with pytest.raises(ValueError, match="дыра"):
        Detection(cls="mound", score=1.0, geom_type="Polygon", geometry=ring, holes=(ring[:2],))


def test_holes_geojson_roundtrip(tmp_path):
    """Дыры переживают запись/чтение; в файле внешнее CCW, дыры CW (RFC 7946)."""
    d = _dist(60.0, 70.0)
    prob = np.clip(1 - np.abs(d - 14.0) / 4.0, 0.0, 1.0).astype(np.float32)
    grid = _grid()
    dets = seg_to_polygons(prob, grid, "mound", PostConfig())
    p = tmp_path / "pred.geojson"
    write_detections(p, dets, grid.crs, "EPSG:3857")
    fc = json.loads(p.read_text(encoding="utf-8"))
    rings = fc["features"][0]["geometry"]["coordinates"]
    assert len(rings) == 2
    # в 3857 ориентация сохраняется (нет отражения): внешнее CCW, дыра CW
    areas = [_area(np.asarray(r, dtype=np.float64)) for r in rings]
    assert areas[0] > 0 and areas[1] < 0
    feats = read_labels(p, grid, "EPSG:3857")
    assert len(feats) == 1 and len(feats[0].world) == 2  # внешнее + дыра


def test_iou_subtracts_holes():
    """IoU кольца с диском того же радиуса = (R²-r²)/R² — дыра уменьшает площадь."""
    r_out, r_in = 20.0, 10.0
    ang = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    center = _world(60, 70)

    def circle(r):
        return np.column_stack([np.cos(ang), np.sin(ang)]) * r * PS + center

    donut = EvalObject("mound", "Polygon", (circle(r_out), circle(r_in)))
    disk = EvalObject("mound", "Polygon", (circle(r_out),))
    iou = polygon_iou(donut.rings, disk.rings, 256)
    assert abs(iou - (r_out**2 - r_in**2) / r_out**2) < 0.02


# Проверки эквивалентности ускоренной постобработки.
def test_regression_centroids_and_nms_match_scalar(monkeypatch):
    """Пакетные центроиды и NMS сохраняют результат скалярного пути."""
    from expds.post import nms as nms_module

    origin = np.array([500000.0, 6200000.0])
    square = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    triangle = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 3.0]])
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    angle = np.linspace(0, 2 * np.pi, 129, endpoint=False)
    large = np.column_stack([np.cos(angle), np.sin(angle)])
    rings = [square, square[::-1], triangle, line, large]
    dets = []
    for i, ring in enumerate(rings):
        for offset in (0.0, 1.0, 30.0):
            dets.append(
                Detection(
                    cls="a" if i % 2 else "b",
                    score=1.0 - len(dets) * 0.01,
                    geom_type="Polygon",
                    geometry=ring + origin + offset,
                )
            )
    hole = np.array([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [3.0, 1.0]])
    dets.append(
        Detection(
            cls="b",
            score=0.7,
            geom_type="Polygon",
            geometry=square + origin,
            holes=(hole + origin,),
        )
    )
    dets.append(
        Detection(cls="a", score=0.6, geom_type="Point", geometry=np.array([[500100.0, 6200100.0]]))
    )

    def scalar(ds):
        return np.asarray([d.centroid for d in ds], dtype=np.float64)

    expected = scalar(dets)
    np.testing.assert_allclose(nms_module._centroids_batched(dets), expected, rtol=0, atol=1e-8)
    for per_class in (True, False):
        accelerated = nms_module.nms_centroids(dets, radius=5.0, per_class=per_class)
        with monkeypatch.context() as patch:
            patch.setattr(nms_module, "_centroids_batched", scalar)
            reference = nms_module.nms_centroids(dets, radius=5.0, per_class=per_class)
        assert [id(d) for d in accelerated] == [id(d) for d in reference]


@pytest.mark.parametrize(
    "seeds", [[], [(4, 4)], [(4, 2), (4, 6)], [(2, 2), (2, 6), (6, 2), (6, 6)]]
)
def test_regression_peak_split_matches_dense_argmin(seeds):
    """Равноудалённые пиксели получают минимальный исходный индекс пика."""
    from expds.post.vectorize import _split_by_peaks

    comp = np.ones((9, 9), dtype=bool)
    comp[0, 0] = False
    hm = np.zeros(comp.shape, dtype=np.float32)
    for r, c in seeds:
        hm[r, c] = 1.0
    actual = _split_by_peaks(comp, hm, PostConfig(hm_thr=0.5, peak_window_px=3))
    expected = np.zeros(comp.shape, dtype=np.int32)
    rr, cc = np.nonzero(comp)
    if len(seeds) < 2:
        expected[rr, cc] = 1
    else:
        points = np.column_stack([rr, cc])
        distances = ((points[:, None, :] - np.asarray(seeds)[None, :, :]) ** 2).sum(2)
        expected[rr, cc] = distances.argmin(1) + 1
    np.testing.assert_array_equal(actual, expected)


# --- регрессии деления слитых объектов (n3_long: oracle recall 0.244) ---------


def _feat_poly(grid, ring_m):
    """LabelFeature полигона: мир -> пиксели Grid (как read_labels)."""
    from expds.io.geojson import LabelFeature

    ring = np.asarray(ring_m, dtype=np.float64)
    px = np.column_stack(
        [(ring[:, 0] - grid.x_min) / grid.pixel_size - 0.5,
         (ring[:, 1] - grid.y_min) / grid.pixel_size - 0.5]
    )
    return LabelFeature(cls="k", geom_type="Polygon", world=(ring,), pixel=(px,), properties={})


def _targets(grid, feats, classes=("k",), sigma=3.0):
    from expds.targets.rasterize import labels_to_targets

    return labels_to_targets(feats, grid, list(classes), sigma)


def _pixel_iou(mask_a, mask_b):
    inter = int((mask_a & mask_b).sum())
    union = int(mask_a.sum() + mask_b.sum() - inter)
    return inter / union if union else 0.0


def _rasterize_det(detection, grid, shape):
    from expds.targets.rasterize import points_in_rings

    ring = np.asarray(detection.geometry, dtype=np.float64)
    col = (ring[:, 0] - grid.x_min) / grid.pixel_size - 0.5
    row = (ring[:, 1] - grid.y_min) / grid.pixel_size - 0.5
    c0, c1 = max(int(col.min()), 0), min(int(np.ceil(col.max())), shape[1] - 1)
    r0, r1 = max(int(row.min()), 0), min(int(np.ceil(row.max())), shape[0] - 1)
    mask = np.zeros(shape, dtype=bool)
    cols, rows = np.meshgrid(np.arange(c0, c1 + 1, dtype=float), np.arange(r0, r1 + 1, dtype=float))
    mask[r0 : r1 + 1, c0 : c1 + 1] = points_in_rings(cols, rows, (np.column_stack([col, row]),))
    return mask


def test_plateau_peak_is_single_object():
    """Плато 2x2 на пике гауссианы (центроид между пикселями) — один объект.

    Раньше hm >= maximum_filter(hm) оставлял все пиксели плато, Voronoi делил
    один объект на сектора с IoU ~0.25, и детекция не проходила матчинг.
    """
    grid = _grid()
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    r_m = 6 * PS
    # центр между пикселями: (10.5, 10.5) в пикселях -> плато 2x2
    cx, cy = X0 + 10.5 * PS, Y0 + 10.5 * PS
    ring = np.column_stack([cx + r_m * np.cos(t), cy + r_m * np.sin(t)])
    feat = _feat_poly(grid, ring)
    y_seg, y_hm = _targets(grid, [feat])
    hm = y_hm[0]
    plateau = hm >= ndimage_maximum(hm)
    assert int(plateau.sum()) >= 2  # плато действительно есть

    cfg = PostConfig(min_area_px=1.0, hm_thr=0.15, peak_window_px=7)
    dets = seg_to_polygons(y_seg[0], grid, "k", cfg, hm=hm)
    assert len(dets) == 1
    iou = _pixel_iou(_rasterize_det(dets[0], grid, y_seg[0].shape), y_seg[0] > 0)
    assert iou > 0.8


def ndimage_maximum(a):
    from scipy import ndimage

    return ndimage.maximum_filter(a, size=7, mode="constant", cval=0.0)


def test_split_touching_different_radii():
    """Два касающихся диска разных радиусов -> две детекции, обе с IoU >= 0.5."""
    grid = _grid(h=96, w=96)
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    r1_m, r2_m = 10 * PS, 6 * PS
    c1 = np.array([X0 + 30 * PS, Y0 + 48 * PS])
    c2 = np.array([X0 + (30 + 10 + 6 - 4) * PS, Y0 + 48 * PS])  # перекрытие 4 px
    ring1 = np.column_stack([c1[0] + r1_m * np.cos(t), c1[1] + r1_m * np.sin(t)])
    ring2 = np.column_stack([c2[0] + r2_m * np.cos(t), c2[1] + r2_m * np.sin(t)])
    feats = [_feat_poly(grid, ring1), _feat_poly(grid, ring2)]
    y_seg, y_hm = _targets(grid, feats)

    cfg = PostConfig(split_touching=True, split_min_distance_px=4,
                     min_area_px=1.0, hm_thr=0.15, peak_window_px=7)
    dets = seg_to_polygons(y_seg[0], grid, "k", cfg, hm=y_hm[0])
    assert len(dets) == 2
    for feat in feats:
        target, _ = _targets(grid, [feat])
        best = max(_pixel_iou(_rasterize_det(d, grid, y_seg[0].shape), target[0] > 0) for d in dets)
        assert best >= 0.5


def test_tiny_component_does_not_disable_split():
    """Мелкий одиночный компонент не отключает деление слитых (глобальный гейт).

    Раньше watershed запускался по условию len(peaks) > n_components: пики
    мелких объектов не проходили threshold_abs, и один мелкий компонент
    выключал деление всех слитых.
    """
    grid = _grid(h=96, w=128)
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    # слитая пара
    r1_m = r2_m = 6 * PS
    c1 = np.array([X0 + 30 * PS, Y0 + 48 * PS])
    c2 = np.array([X0 + (30 + 8) * PS, Y0 + 48 * PS])  # перекрытие 4 px
    pair = [
        _feat_poly(grid, np.column_stack([c1[0] + r1_m * np.cos(t), c1[1] + r1_m * np.sin(t)])),
        _feat_poly(grid, np.column_stack([c2[0] + r2_m * np.cos(t), c2[1] + r2_m * np.sin(t)])),
    ]
    # мелкий одиночный (r = 2 px < split_min_radius)
    c3 = np.array([X0 + 90 * PS, Y0 + 48 * PS])
    tiny = _feat_poly(
        grid, np.column_stack([c3[0] + 2 * PS * np.cos(t), c3[1] + 2 * PS * np.sin(t)])
    )
    y_seg, y_hm = _targets(grid, pair + [tiny])

    cfg = PostConfig(split_touching=True, split_min_distance_px=4,
                     min_area_px=1.0, hm_thr=0.15, peak_window_px=7)
    dets = seg_to_polygons(y_seg[0], grid, "k", cfg, hm=y_hm[0])
    # пара делится на 2 + мелкий = 3 детекции
    assert len(dets) == 3
