"""Тесты management-команды import_location."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from web.apps.viz.models import Location, MapLayer


@pytest.fixture
def location_dir(tmp_path: Path) -> Path:
    """Временная папка с одной локацией."""
    return tmp_path / "001-gnezdovo-nivlenskoe"


@pytest.fixture
def tiles_root(tmp_path: Path) -> Path:
    """Временный корень с локациями."""
    return tmp_path


@pytest.mark.django_db
def test_import_from_manifest(tiles_root: Path, location_dir: Path) -> None:
    """Команда импортирует локацию по manifest.json."""
    location_dir.mkdir()
    manifest = {
        "slug": "001-gnezdovo-nivlenskoe",
        "name": "001 Гнёздово Нивленское",
        "center": {"lat": 54.8794, "lon": 31.9719},
        "zoom": 14,
    }
    (location_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Структура с подпапкой tiles/
    (location_dir / "tiles" / "satellite" / "14" / "9900").mkdir(parents=True)
    (location_dir / "tiles" / "lidar" / "14" / "9900").mkdir(parents=True)

    out = StringIO()
    call_command("import_location", tiles_dir=str(tiles_root), stdout=out)

    location = Location.objects.get(slug="001-gnezdovo-nivlenskoe")
    assert location.name == "001 Гнёздово Нивленское"
    assert location.center_lat == pytest.approx(54.8794)
    assert location.center_lon == pytest.approx(31.9719)
    assert location.zoom == 14

    layers = list(location.layers.values_list("slug", flat=True))
    assert "lidar" in layers
    assert "satellite" in layers

    satellite = location.layers.get(slug="satellite")
    assert satellite.min_zoom == 14
    assert satellite.max_zoom == 14


@pytest.mark.django_db
def test_import_detects_zoom_range(tiles_root: Path, location_dir: Path) -> None:
    """Команда определяет min/max zoom по папкам {z}/ внутри слоя."""
    location_dir.mkdir()
    manifest = {
        "slug": "001-gnezdovo-nivlenskoe",
        "name": "001 Гнёздово Нивленское",
        "center": {"lat": 54.8794, "lon": 31.9719},
        "zoom": 14,
    }
    (location_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    (location_dir / "tiles" / "satellite" / "13" / "0").mkdir(parents=True)
    (location_dir / "tiles" / "satellite" / "14" / "0").mkdir(parents=True)
    (location_dir / "tiles" / "satellite" / "16" / "0").mkdir(parents=True)

    call_command("import_location", tiles_dir=str(tiles_root), stdout=StringIO())

    location = Location.objects.get(slug="001-gnezdovo-nivlenskoe")
    satellite = location.layers.get(slug="satellite")
    assert satellite.min_zoom == 13
    assert satellite.max_zoom == 16


@pytest.mark.django_db
def test_import_without_manifest(location_dir: Path, tiles_root: Path) -> None:
    """Команда импортирует локацию по аргументам, если manifest отсутствует."""
    location_dir.mkdir()
    (location_dir / "satellite" / "14" / "9900").mkdir(parents=True)

    out = StringIO()
    call_command(
        "import_location",
        tiles_dir=str(tiles_root),
        slug="001-gnezdovo-nivlenskoe",
        name="Гнёздово",
        lat=54.0,
        lon=31.0,
        zoom=13,
        stdout=out,
    )

    location = Location.objects.get(slug="001-gnezdovo-nivlenskoe")
    assert location.name == "Гнёздово"
    assert location.center_lat == pytest.approx(54.0)


@pytest.mark.django_db
def test_dry_run_does_not_create_records(tiles_root: Path, location_dir: Path) -> None:
    """--dry-run не создаёт записи в БД."""
    location_dir.mkdir()
    manifest = {
        "slug": "001-gnezdovo-nivlenskoe",
        "name": "001 Гнёздово Нивленское",
        "center": {"lat": 54.8794, "lon": 31.9719},
        "zoom": 14,
    }
    (location_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (location_dir / "tiles" / "satellite" / "14" / "9900").mkdir(parents=True)

    out = StringIO()
    call_command("import_location", tiles_dir=str(tiles_root), dry_run=True, stdout=out)

    assert Location.objects.count() == 0
    assert MapLayer.objects.count() == 0
    assert "Будет импортирована локация" in out.getvalue()


@pytest.mark.django_db
def test_missing_center_raises_error(location_dir: Path, tiles_root: Path) -> None:
    """Без центра и зума команда падает с понятной ошибкой."""
    location_dir.mkdir()
    (location_dir / "satellite" / "14" / "9900").mkdir(parents=True)

    with pytest.raises(CommandError, match="не задан центр/зум"):
        call_command(
            "import_location",
            tiles_dir=str(tiles_root),
            slug="001-gnezdovo-nivlenskoe",
        )


@pytest.mark.django_db
def test_import_geojson_layer(tiles_root: Path, location_dir: Path) -> None:
    """GeoJSON-файлы в корне локации импортируются как векторные слои."""
    location_dir.mkdir()
    manifest = {
        "slug": "001-gnezdovo-nivlenskoe",
        "name": "001 Гнёздово Нивленское",
        "center": {"lat": 54.8794, "lon": 31.9719},
        "zoom": 14,
    }
    (location_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (location_dir / "satellite" / "14" / "9900").mkdir(parents=True)
    (location_dir / "annotation_kurgans.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )

    call_command("import_location", tiles_dir=str(tiles_root), stdout=StringIO())

    location = Location.objects.get(slug="001-gnezdovo-nivlenskoe")
    geojson_layer = location.layers.get(slug="annotation_1")
    assert geojson_layer.layer_type == "geojson"
    assert geojson_layer.geojson_url.endswith("annotation_kurgans.geojson")
