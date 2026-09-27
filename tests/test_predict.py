import json

import pytest

from expds.predict import find_sites, merge_geojson


def test_find_sites(tmp_path):
    for name in ("site_2", "site_1"):
        (tmp_path / name / "layers").mkdir(parents=True)
    (tmp_path / "junk").mkdir()
    assert [p.name for p in find_sites(tmp_path)] == ["site_1", "site_2"]
    assert find_sites(tmp_path / "site_1") == [tmp_path / "site_1"]
    with pytest.raises(FileNotFoundError, match="build_layers"):
        find_sites(tmp_path / "junk")


def test_merge_geojson(tmp_path):
    crs = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}
    parts = []
    for i in range(2):
        p = tmp_path / f"p{i}.geojson"
        feats = [{"type": "Feature", "properties": {"class": "mound"}, "geometry": None}] * (i + 1)
        p.write_text(json.dumps({"type": "FeatureCollection", "crs": crs, "features": feats}))
        parts.append(p)
    out = tmp_path / "out.geojson"
    merge_geojson(parts, out)
    fc = json.loads(out.read_text())
    assert len(fc["features"]) == 3 and fc["crs"] == crs
