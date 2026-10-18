"""Тесты API-представлений."""

import yaml
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
    # drf-spectacular отдаёт схему в YAML с Content-Type
    # application/vnd.oai.openapi, поэтому парсим вручную.
    schema = yaml.safe_load(response.content)
    assert "openapi" in schema
