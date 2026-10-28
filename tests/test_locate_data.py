"""Тесты локатора данных: scripts/locate_data.py (структура, фильтры, отчёты)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import locate_data as ld  # noqa: E402


def make_site(
    root: Path,
    num: str,
    name: str,
    *,
    las: int = 1,
    tif: int = 2,
    layers: bool = False,
    root_zip: bool = False,
) -> Path:
    """Собирает участок в структуре train-koz-3: 01_*_La_точки + 02_*_Li_карты."""
    site = root / f"{num}_{name}"
    las_dir = site / f"01_{name}_La_точки"
    las_dir.mkdir(parents=True)
    for i in range(las):
        (las_dir / f"{name}_part{i}.las").write_bytes(b"las")
    if tif:
        raster_dir = site / f"02_{name}_Li_карты"
        raster_dir.mkdir()
        for i in range(tif):
            (raster_dir / f"{name}_map{i}.tif").write_bytes(b"tif")
    if layers:
        cache = las_dir / "layers"
        cache.mkdir()
        (cache / "hillshade_az_315.tif").write_bytes(b"cache")
    if root_zip:
        (root / f"{num}_{name}.zip").write_bytes(b"zip")
    return site


def test_counts_real_files_not_just_names(tmp_path):
    make_site(tmp_path, "001", "Гнёздово", las=2, tif=3, root_zip=True)
    make_site(tmp_path, "002", "Нора", las=1, tif=0)

    first = ld.scan_site(tmp_path / "001_Гнёздово")
    assert (first.las_count, first.raster_count) == (2, 3)
    assert first.site_id == "001" and first.ok

    second = ld.scan_site(tmp_path / "002_Нора")
    assert second.las_count == 1 and second.raster_count == 0
    assert not second.ok


def test_layers_cache_not_counted(tmp_path):
    make_site(tmp_path, "003", "С_кэшем", las=1, tif=1, layers=True)
    info = ld.scan_site(tmp_path / "003_С_кэшем")
    assert info.las_count == 1 and info.raster_count == 1


def test_direct_sites_root(tmp_path):
    make_site(tmp_path, "001", "А", las=1, tif=1)
    result = ld.scan_candidate(tmp_path, with_sizes=False, deadline=ld.Deadline(None))
    assert len(result.groups) == 1
    assert result.groups[0].sites[0].name == "001_А"
    assert result.exists and not result.truncated


def test_data_root_nested_groups(tmp_path):
    data = tmp_path / "04-data"
    make_site(data / "train-koz-3", "001", "А", las=1, tif=1)
    make_site(data / "test", "013", "Б", las=1, tif=1)
    (data / "archive").mkdir()

    result = ld.scan_candidate(data, with_sizes=False, deadline=ld.Deadline(None))
    names = {g.name for g in result.groups}
    assert names == {"train-koz-3", "test"}
    assert sum(len(g.sites) for g in result.groups) == 2


def test_missing_path_is_reported_not_raised(tmp_path):
    result = ld.scan_candidate(
        tmp_path / "нет-такого", with_sizes=False, deadline=ld.Deadline(None)
    )
    assert not result.exists and result.groups == []


def test_full_mode_collects_sizes(tmp_path):
    make_site(tmp_path, "001", "А", las=1, tif=2)
    info = ld.scan_site(tmp_path / "001_А", with_sizes=True)
    assert info.las_bytes == 3 and info.raster_bytes == 6


class _Expired:
    expired = True


def test_deadline_stops_scan_gracefully(tmp_path):
    for i in range(5):
        make_site(tmp_path, f"{i + 1:03d}", f"С{i}", las=1, tif=1)
    sites, truncated = ld.scan_sites_root(tmp_path, with_sizes=False, deadline=_Expired())
    assert truncated and sites == []


def test_interrupt_yields_partial_report(tmp_path, monkeypatch, capsys):
    for i in range(3):
        make_site(tmp_path, f"{i + 1:03d}", f"С{i}", las=1, tif=1)
    calls = {"n": 0}
    real_scan = ld.scan_site

    def flaky(path, with_sizes=False):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        return real_scan(path, with_sizes=with_sizes)

    monkeypatch.setattr(ld, "scan_site", flaky)
    rc = ld.main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 130
    assert "ЧАСТИЧНО" in out
    assert "полных 1" in out


def test_expect_min_soft_vs_strict(tmp_path, capsys):
    make_site(tmp_path, "001", "А", las=1, tif=1)
    assert ld.main(["--root", str(tmp_path), "--expect-min", "192"]) == 0
    assert "НЕ ДОСТИГНУТ" in capsys.readouterr().out
    rc = ld.main(["--root", str(tmp_path), "--expect-min", "192", "--strict"])
    assert rc == 2


def test_expect_min_ok_when_enough(tmp_path, capsys):
    make_site(tmp_path, "001", "А", las=1, tif=1)
    rc = ld.main(["--root", str(tmp_path), "--expect-min", "1", "--strict"])
    assert rc == 0
    assert "Минимум полных сайтов 1: OK" in capsys.readouterr().out


def test_json_report(tmp_path):
    make_site(tmp_path, "001", "А", las=1, tif=1)
    out_json = tmp_path / "report.json"
    rc = ld.main(["--root", str(tmp_path), "--json", str(out_json)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["totals"]["sites"] == 1
    assert payload["candidates"][0]["groups"][0]["sites"][0]["ok"] is True
    assert payload["mode"] == "quick"


def test_early_stop_after_first_hit(tmp_path, capsys):
    hit = tmp_path / "hit"
    miss = tmp_path / "miss"
    make_site(hit, "001", "А", las=1, tif=1)
    miss.mkdir()
    rc = ld.main(["--root", str(hit), "--root", str(miss)])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(miss) not in out


def test_default_candidates_no_crash(capsys):
    rc = ld.main(["--timeout", "0.001", "--quiet"])
    out = capsys.readouterr().out
    assert rc in (0, 130)
    assert "Локатор данных КОЗ-3" in out
