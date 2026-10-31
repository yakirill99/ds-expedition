"""Манифест участков: диапазоны номеров, атомарность групп, стабильность сплита."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expds.data.site_sets import parse_refs, scan, select


def _site(root: Path, name: str, code: str = "04", objects: int = 3) -> None:
    d = root / name
    (d / f"{code}_x").mkdir(parents=True)
    feat = {
        "type": "Feature",
        "properties": {"class": "kurgany_tselye"},
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }
    (d / "labels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [feat] * objects}), encoding="utf-8"
    )


def _manifest() -> dict:
    return {
        "groups": ["Гнёздово"],
        "split": {"seed": 0, "fractions": {"train": 0.6, "val": 0.2, "test": 0.2}},
    }


def test_parse_refs_ranges_and_names() -> None:
    """Диапазон номеров, одиночный номер и полное имя в одной строке."""
    ids, names = parse_refs("014-016, 030,200_Судьбищи_2")
    assert ids == {14, 15, 16, 30}
    assert names == {"200_Судьбищи_2"}
    assert parse_refs(["024-022"])[0] == {22, 23, 24}  # обратный диапазон
    assert parse_refs(None) == (set(), set())


def test_parse_refs_rejects_bare_numbers() -> None:
    """YAML читает 014 как восьмеричное 12 — числа только в кавычках."""
    with pytest.raises(ValueError, match="без кавычек"):
        parse_refs([14])


def test_group_is_atomic(tmp_path: Path) -> None:
    """Группа ландшафта целиком в одной части; прочие участки — сами себе группы."""
    for i in range(1, 9):
        _site(tmp_path, f"{i:03d}_Гнёздово_{i}")
    for i in range(10, 40):
        _site(tmp_path, f"{i:03d}_Курганы_{i}")
    sites = scan(tmp_path, _manifest())
    assert len({s.split for s in sites if s.group == "Гнёздово"}) == 1
    assert len({s.group for s in sites}) == 31  # Гнёздово + Курганы_10..39


def test_split_stable_when_sites_added(tmp_path: Path) -> None:
    """Докопирование новых участков не перетасовывает уже распределённые."""
    for i in range(1, 30):
        _site(tmp_path, f"{i:03d}_Участок_{i}")
    before = {s.name: s.split for s in scan(tmp_path, _manifest())}
    for i in range(30, 60):
        _site(tmp_path, f"{i:03d}_Участок_{i}")
    after = {s.name: s.split for s in scan(tmp_path, _manifest())}

    assert set(before) < set(after), "старые участки пропали из обхода"
    changed = {n for n in before if before[n] != after[n]}
    assert not changed, f"сплит перетасован: {sorted(changed)[:5]}"


def test_exclude_removes_from_scan(tmp_path: Path) -> None:
    """Глобальный exclude убирает участок из всех наборов сразу."""
    _site(tmp_path, "001_Гнёздово_Нивленское")
    _site(tmp_path, "216_Гнёздово_Селище_2")
    m = _manifest()
    m["exclude"] = ["216"]
    assert {s.name for s in scan(tmp_path, m)} == {"001_Гнёздово_Нивленское"}


def test_select_only_and_require(tmp_path: Path) -> None:
    """only — белый список; require/missing смотрят на реальные файлы."""
    _site(tmp_path, "001_Гнёздово_Нивленское", code="01")
    _site(tmp_path, "002_Тешилово", code="04")
    _site(tmp_path, "003_Лубно", code="04")
    (tmp_path / "003_Лубно" / "labels.geojson").unlink()
    (tmp_path / "002_Тешилово" / "layers").mkdir()
    (tmp_path / "002_Тешилово" / "layers" / "rgb_r.tif").write_bytes(b"")

    sites = scan(tmp_path, _manifest())
    picked = select(sites, {"only": "002-003", "kind": "optic", "require": ["built", "labels"]})
    assert [s.name for s in picked] == ["002_Тешилово"]
    assert [s.name for s in select(sites, {"missing": ["labels"]})] == ["003_Лубно"]


def test_select_kind_detection(tmp_path: Path) -> None:
    """Тип участка — по папкам модальностей, как в survey_dataset."""
    _site(tmp_path, "001_a", code="01")
    _site(tmp_path, "002_b", code="04")
    _site(tmp_path, "003_c", code="07")
    _site(tmp_path, "004_d", code="06")
    kinds = {s.name: s.kind for s in scan(tmp_path, _manifest())}
    assert kinds == {"001_a": "lidar", "002_b": "optic", "003_c": "geo", "004_d": "none"}
