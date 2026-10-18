"""Дымовой тест inventory.py: обход, классификация, запись отчётов."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import inventory  # noqa: E402


def test_classify_urls():
    assert inventory.classify(Path("a/b/lidar.las")) == "lidar"
    assert inventory.classify(Path("x/dtm.tif")) == "relief"
    assert inventory.classify(Path("x/hillshade.tif")) == "relief"
    assert inventory.classify(Path("x/rgb.tif")) == "optic"
    assert inventory.classify(Path("x/mag.tif")) == "geo"
    assert inventory.classify(Path("x/obj.zip")) == "archive"
    assert inventory.classify(Path("x/labels.geojson")) == "labels"


def test_walk_and_reports(tmp_path):
    root = tmp_path / "01-source"
    (root / "site_1").mkdir(parents=True)
    (root / "site_1" / "dtm.tif").write_bytes(b"not-a-real-tif")
    (root / "site_1" / "labels.geojson").write_text("{}", encoding="utf-8")
    (root / "notes.txt").write_text("ignore me", encoding="utf-8")  # не в suffixes

    records = inventory.walk(root)
    rel = {r.rel_path.split("/")[-1] for r in records}
    assert "dtm.tif" in rel
    assert "labels.geojson" in rel
    assert "notes.txt" not in rel  # не попал по суффиксу

    out_json = tmp_path / "inv.json"
    out_md = tmp_path / "inv.md"
    inventory.write_json(records, out_json)
    inventory.write_report(records, inventory.find_gaps(records), out_md)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_files"] == len(records)
    assert out_md.read_text(encoding="utf-8").startswith("# Инвентаризация")
