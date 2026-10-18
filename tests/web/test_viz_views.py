"""Тесты представлений приложения viz."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from django.test import Client
from django.urls import reverse

from web.apps.viz.models import Location, MapLayer


@pytest.fixture
def client() -> Client:
    return Client()


@pytest.mark.django_db
def test_map_page_renders(client: Client) -> None:
    """Главная страница viz рендерит HTML с Leaflet."""
    response = client.get(reverse("viz-index"))
    assert response.status_code == 200
    content = response.content.decode()
    assert "leaflet" in content.lower()
    assert "Локация" in content


@pytest.mark.django_db
def test_map_page_has_valid_leaflet_script(client: Client) -> None:
    """JS-код инициализации карты не содержит случайных закрывающих тегов script."""
    response = client.get(reverse("viz-index"))
    assert response.status_code == 200
    content = response.content.decode()
    assert "L.map('map')" in content
    assert "L.map</script>" not in content


@pytest.mark.django_db
def test_map_page_uses_radio_for_raster_layers(client: Client) -> None:
    """Растровые слои переключаются радиокнопками, GeoJSON — чекбоксами."""
    response = client.get(reverse("viz-index"))
    assert response.status_code == 200
    content = response.content.decode()
    assert "input.type = 'radio'" in content
    assert "input.name = 'raster-layer'" in content
    assert "input.type = 'checkbox'" in content


@pytest.mark.django_db
def test_map_page_assigns_colors_to_geojson_layers(client: Client) -> None:
    """GeoJSON-слои получают разные цвета из палитры."""
    response = client.get(reverse("viz-index"))
    assert response.status_code == 200
    content = response.content.decode()
    assert "GEOJSON_COLORS" in content
    assert "const layerColor = GEOJSON_COLORS[geoJsonIndex % GEOJSON_COLORS.length]" in content


@pytest.mark.django_db
def test_location_list_empty(client: Client) -> None:
    """API локаций возвращает пустой список, если данных нет."""
    response = client.get(reverse("viz-location-list"))
    assert response.status_code == 200
    assert response.json() == {"locations": []}


@pytest.mark.django_db
def test_location_layers(client: Client) -> None:
    """API слоёв возвращает шаблоны тайлов для локации."""
    location = Location.objects.create(
        name="Тест",
        slug="test-location",
        center_lat=55.0,
        center_lon=37.0,
        zoom=10,
    )
    MapLayer.objects.create(
        location=location,
        name="Спутник",
        slug="satellite",
        layer_type="satellite",
        order=0,
        min_zoom=13,
        max_zoom=16,
    )

    response = client.get(reverse("viz-location-layers", args=[location.slug]))
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["slug"] == "test-location"
    assert len(data["layers"]) == 1
    layer = data["layers"][0]
    assert "/viz/tiles/test-location/satellite/{z}/{x}/{y}.png" in layer["tile_url_template"]
    assert layer["min_zoom"] == 13
    assert layer["max_zoom"] == 16


@pytest.mark.django_db
def test_tile_serve_404_for_missing_tile(client: Client) -> None:
    """Запрос несуществующего тайла возвращает 404."""
    response = client.get("/viz/tiles/missing/layer/0/0/0.png")
    assert response.status_code == 404


@pytest.mark.django_db
def test_geojson_serve_reprojects_to_wgs84(client: Client, settings, tmp_path: Path) -> None:
    """GeoJSON endpoint перепроецирует координаты из CRS локации в WGS84."""
    settings.MAP_STORAGE_ROOT = tmp_path
    settings.MAP_TILE_URL_PREFIX = "/viz/tiles/"

    location = Location.objects.create(
        name="Тест",
        slug="test-geojson",
        center_lat=55.0,
        center_lon=37.0,
        zoom=10,
    )
    filename = "test.geojson"
    MapLayer.objects.create(
        location=location,
        name="Разметка",
        slug="annotation_1",
        layer_type="geojson",
        geojson_url=f"/viz/geojson/{location.slug}/{filename}",
    )

    location_dir = tmp_path / location.slug
    location_dir.mkdir()
    (location_dir / "manifest.json").write_text(json.dumps({"crs": "EPSG:3857"}), encoding="utf-8")

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [3540420.0, 7316830.0],
                },
                "properties": {"id": 1},
            }
        ],
    }
    (location_dir / filename).write_text(json.dumps(geojson), encoding="utf-8")

    response = client.get(reverse("viz-geojson", args=[location.slug, filename]))
    assert response.status_code == 200
    data = response.json()
    coords = data["features"][0]["geometry"]["coordinates"]
    # Примерно Смоленская область
    assert 30 < coords[0] < 35
    assert 54 < coords[1] < 56
