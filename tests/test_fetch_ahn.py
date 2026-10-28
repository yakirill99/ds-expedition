"""Тесты fetch_ahn: резолв сервиса, разбор бокса и URL-билдер (без сети)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_ahn", Path(__file__).resolve().parents[1] / "scripts" / "fetch_ahn.py"
)
assert spec is not None and spec.loader is not None
fetch_ahn = importlib.util.module_from_spec(spec)
sys.modules.setdefault("fetch_ahn", fetch_ahn)
spec.loader.exec_module(fetch_ahn)

_VAALS_512 = (180000.0, 318000.0, 180512.0, 318512.0)


def _ns(**kw):
    import argparse

    return argparse.Namespace(bbox=kw.get("bbox"), center=kw.get("center"), size_m=kw.get("size_m"))


def test_resolve_service() -> None:
    assert fetch_ahn.resolve_service("5", "dtm") == "AHN5_DTM_50cm"
    assert fetch_ahn.resolve_service("5", "dsm") == "AHN5_DSM_50cm"
    assert fetch_ahn.resolve_service("4", "dtm") == "AHN4_DTM_50cm"
    with pytest.raises(ValueError, match="версия AHN"):
        fetch_ahn.resolve_service("3", "dtm")
    with pytest.raises(ValueError, match="dataset"):
        fetch_ahn.resolve_service("5", "ndvi")


def test_parse_bbox_explicit() -> None:
    assert fetch_ahn.parse_bbox(_ns(bbox=["180000", "318000", "180512", "318512"])) == _VAALS_512


def test_parse_bbox_center_size() -> None:
    bbox = fetch_ahn.parse_bbox(_ns(center=["180256", "318256"], size_m=512))
    assert bbox == _VAALS_512


def test_parse_bbox_rejects_ambiguous() -> None:
    with pytest.raises(ValueError, match="ровно один"):
        fetch_ahn.parse_bbox(_ns(bbox=["1", "2", "3", "4"], center=["1", "2"], size_m=10))
    with pytest.raises(ValueError, match="ровно один"):
        fetch_ahn.parse_bbox(_ns())
    with pytest.raises(ValueError, match="size-m"):
        fetch_ahn.parse_bbox(_ns(center=["1", "2"]))
    with pytest.raises(ValueError, match="вырожден"):
        fetch_ahn.parse_bbox(_ns(bbox=["10", "20", "10", "30"]))


def test_build_export_url() -> None:
    url, w, h = fetch_ahn.build_export_url("AHN5_DTM_50cm", _VAALS_512)
    assert (w, h) == (1024, 1024)
    assert url.startswith(fetch_ahn.AHN_BASE + "/AHN5_DTM_50cm/ImageServer/exportImage?")
    assert "bbox=180000.0%2C318000.0%2C180512.0%2C318512.0" in url
    assert "size=1024%2C1024" in url
    assert "bboxSR=28992" in url and "imageSR=28992" in url
    assert "format=tiff" in url and "pixelType=F32" in url and "f=image" in url


def test_build_export_url_rejects_too_big() -> None:
    big = (10000.0, 306250.0, 11255.5, 318780.5)  # 5000 x 5000 px
    with pytest.raises(ValueError, match="лимита"):
        fetch_ahn.build_export_url("AHN5_DTM_50cm", big)
    with pytest.raises(ValueError, match="меньше пикселя"):
        fetch_ahn.build_export_url("AHN5_DTM_50cm", (180000.0, 318000.0, 180000.2, 318000.4))
