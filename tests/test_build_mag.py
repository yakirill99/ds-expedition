"""Регрессии build_mag: какие файлы 07_ становятся каналами mag_*."""

from pathlib import Path

from expds.data.build_mag import _mag_sources


def test_regression_mag_skips_colour_map(tmp_path: Path) -> None:
    """Из пары серый/цвет в 07_ берётся только серая карта."""
    mag = tmp_path / "07_site_MagGr_карты"
    mag.mkdir()
    (mag / "01_site_MagGr_серый.tif").write_bytes(b"")
    (mag / "02_site_MagGr_цвет.tif").write_bytes(b"")
    assert [p.name for p in _mag_sources(tmp_path)] == ["01_site_MagGr_серый.tif"]
