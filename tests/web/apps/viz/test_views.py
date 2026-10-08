"""Тесты визуализации."""

from rest_framework.test import APIClient


def test_viz_index():
    client = APIClient()
    response = client.get("/viz/")
    assert response.status_code == 200
    assert response.json()["detail"] == "viz index"
