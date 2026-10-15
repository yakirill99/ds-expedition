"""Тесты представлений приложения viz."""

from __future__ import annotations

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
    )

    response = client.get(reverse("viz-location-layers", args=[location.slug]))
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["slug"] == "test-location"
    assert len(data["layers"]) == 1
    assert (
        "/viz/tiles/test-location/satellite/{z}/{x}/{y}.png"
        in data["layers"][0]["tile_url_template"]
    )


@pytest.mark.django_db
def test_tile_serve_404_for_missing_tile(client: Client) -> None:
    """Запрос несуществующего тайла возвращает 404."""
    response = client.get("/viz/tiles/missing/layer/0/0/0.png")
    assert response.status_code == 404
