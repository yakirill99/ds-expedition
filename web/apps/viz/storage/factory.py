"""Фабрика хранилища карт: выбор реализации по настройкам Django."""

from __future__ import annotations

from django.conf import settings

from .base import MapStorage
from .filesystem import FileSystemMapStorage
from .s3 import S3MapStorage

_BACKENDS: dict[str, type[MapStorage]] = {
    "filesystem": FileSystemMapStorage,
    "s3": S3MapStorage,
}


def get_map_storage() -> MapStorage:
    """Вернуть активное хранилище карт по настройке ``MAP_STORAGE_BACKEND``."""
    backend_name = getattr(settings, "MAP_STORAGE_BACKEND", "filesystem")
    try:
        backend_cls = _BACKENDS[backend_name]
    except KeyError as exc:
        raise ValueError(f"Неизвестный бэкенд хранилища карт: {backend_name}") from exc
    return backend_cls()
