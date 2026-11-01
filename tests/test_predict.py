"""Инференс: find_sites, merge_geojson и контракт платформы на пустой ответ."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import expds.predict as predict
from expds.data.build_site import RawSiteUnsupportedError
from expds.predict import find_sites, merge_geojson


def test_find_sites(tmp_path: Path) -> None:
    for name in ("site_2", "site_1"):
        (tmp_path / name / "layers").mkdir(parents=True)
    (tmp_path / "junk").mkdir()
    assert [p.name for p in find_sites(tmp_path)] == ["site_1", "site_2"]
    assert find_sites(tmp_path / "site_1") == [tmp_path / "site_1"]
    with pytest.raises(FileNotFoundError, match="build_layers"):
        find_sites(tmp_path / "junk")


def test_merge_geojson(tmp_path: Path) -> None:
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


def test_main_writes_empty_and_exits_zero(tmp_path: Path, monkeypatch) -> None:
    """Регион, который не собрать: валидный пустой GeoJSON и код 0.

    Код != 0 на платформе сжигает одну из трёх попыток запуска в сутки,
    а отсутствие файла даёт ещё и ошибку разбора ответа.
    """

    def _boom(*args: object, **kwargs: object) -> dict:
        raise RawSiteUnsupportedError("нет ни одного участка")

    monkeypatch.setattr(predict, "run", _boom)
    monkeypatch.setattr(
        predict,
        "load_pipeline_config",
        lambda _p: SimpleNamespace(data=SimpleNamespace(labels_crs="EPSG:3857")),
    )

    out = tmp_path / "out.geojson"
    rc = predict.main(
        [
            "--data",
            str(tmp_path / "data"),
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--weights",
            str(tmp_path / "w.pt"),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert fc["features"] == []
