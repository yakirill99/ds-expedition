"""Тесты API-представлений."""

from rest_framework.test import APIClient


def test_health():
    client = APIClient()
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
