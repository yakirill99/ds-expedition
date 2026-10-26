"""Тесты валидатора формата платформы."""

import json

import pytest

from expds.eval.validate import validate_submission

X0, Y0 = 4_200_000.0, 7_500_000.0


def _write(tmp_path, obj, name="s.geojson"):
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def _point(x=X0, y=Y0, cls="mound", score=0.9):
    return {
        "type": "Feature",
        "properties": {"class": cls, "score": score},
        "geometry": {"type": "Point", "coordinates": [x, y]},
    }


def _polygon(cls="mound", hole=False):
    rings = [[[X0, Y0], [X0 + 20, Y0], [X0 + 20, Y0 + 20], [X0, Y0 + 20], [X0, Y0]]]
    if hole:
        rings.append(
            [
                [X0 + 4, Y0 + 4],
                [X0 + 8, Y0 + 4],
                [X0 + 8, Y0 + 8],
                [X0 + 4, Y0 + 8],
                [X0 + 4, Y0 + 4],
            ]
        )
    return {
        "type": "Feature",
        "properties": {"class": cls, "score": 1.0},
        "geometry": {"type": "Polygon", "coordinates": rings},
    }


def _fc(features, crs="urn:ogc:def:crs:EPSG::3857"):
    fc = {"type": "FeatureCollection", "features": features}
    if crs:
        fc["crs"] = {"type": "name", "properties": {"name": crs}}
    return fc


def test_valid_file_passes(tmp_path):
    p = _write(tmp_path, _fc([_point(), _polygon(hole=True)]))
    assert validate_submission(p) == []


def test_no_crs_member_is_ok(tmp_path):
    """RFC 7946 не требует члена crs — отсутствие не ошибка."""
    p = _write(tmp_path, {"type": "FeatureCollection", "features": [_point()]})
    assert validate_submission(p) == []


def test_not_a_feature_collection(tmp_path):
    p = _write(tmp_path, {"type": "Feature"})
    problems = validate_submission(p)
    assert len(problems) == 1 and "FeatureCollection" in problems[0]


def test_features_not_a_list(tmp_path):
    p = _write(tmp_path, {"type": "FeatureCollection", "features": {"a": 1}})
    assert any("списком" in s for s in validate_submission(p))


def test_missing_class_key(tmp_path):
    feat = _point()
    feat["properties"] = {"score": 0.9}
    p = _write(tmp_path, _fc([feat]))
    problems = validate_submission(p)
    assert any("class" in s and "feature 0" in s for s in problems)


def test_bad_geometry_type(tmp_path):
    p = _write(
        tmp_path,
        _fc([{**_point(), "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}]),
    )
    problems = validate_submission(p)
    assert any("LineString" in s for s in problems)


def test_unclosed_and_short_ring(tmp_path):
    unclosed = _polygon()
    unclosed["geometry"]["coordinates"] = [c[:-1] for c in unclosed["geometry"]["coordinates"]]
    short = _polygon()
    short["geometry"]["coordinates"] = [[[X0, Y0], [X0 + 1, Y0], [X0, Y0]]]
    problems = validate_submission(_write(tmp_path, _fc([unclosed, short])))
    assert any("не замкнуто" in s for s in problems)
    assert any(">= 4 вершин" in s for s in problems)


def test_wrong_declared_crs(tmp_path):
    p = _write(tmp_path, _fc([_point()], crs="urn:ogc:def:crs:EPSG::4326"))
    problems = validate_submission(p)
    assert any("4326" in s and "3857" in s for s in problems)


@pytest.mark.parametrize("score", [1.5, "0.5", True])
def test_bad_score(tmp_path, score):
    p = _write(tmp_path, _fc([_point(score=score)]))
    assert any("score" in s for s in validate_submission(p))


def test_multipolygon_and_null_geometry(tmp_path):
    multi = _polygon()
    multi["geometry"] = {"type": "MultiPolygon", "coordinates": [multi["geometry"]["coordinates"]]}
    null_geom = {**_point(), "geometry": None}
    assert validate_submission(_write(tmp_path, _fc([multi, null_geom]))) == []


def test_non_numeric_coordinates(tmp_path):
    feat = _point()
    feat["geometry"]["coordinates"] = [X0, "Y"]
    assert any("2-3 числа" in s for s in validate_submission(_write(tmp_path, _fc([feat]))))
