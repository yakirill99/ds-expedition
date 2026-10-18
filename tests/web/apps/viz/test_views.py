"""Тесты визуализации."""

from django.test import Client


def test_viz_index():
    client = Client()
    response = client.get("/viz/")
    assert response.status_code == 200
    assert "leaflet" in response.content.decode().lower()
