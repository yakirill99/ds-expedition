"""Тесты API-представлений."""

from rest_framework.test import APIClient


def test_health():
    client = APIClient()
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_swagger_ui():
    client = APIClient()
    response = client.get("/api/docs/")
    assert response.status_code == 200


def test_redoc():
    client = APIClient()
    response = client.get("/api/redoc/")
    assert response.status_code == 200


def test_schema():
    client = APIClient()
    response = client.get("/api/schema/")
    assert response.status_code == 200
    assert "openapi" in response.json()
