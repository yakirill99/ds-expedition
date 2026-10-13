"""Модуль хранилища карт: абстракция над файловой системой и S3."""

from .base import MapStorage, TilePath
from .factory import get_map_storage
from .filesystem import FileSystemMapStorage

__all__ = ["MapStorage", "TilePath", "get_map_storage", "FileSystemMapStorage"]
