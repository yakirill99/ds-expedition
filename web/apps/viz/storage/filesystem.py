"""Файловая реализация хранилища тайлов."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings

from .base import TilePath


class FileSystemMapStorage:
    """Хранилище тайлов в локальной файловой системе.

    Тайлы лежат в ``MAP_STORAGE_ROOT`` по схеме:
    ``<location_slug>/<layer_slug>/<z>/<x>/<y>.<ext>``.
    Публичный URL формируется относительно ``MAP_TILE_URL_PREFIX``.
    """

    def __init__(self, root: Path | str | None = None, url_prefix: str | None = None) -> None:
        self.root = Path(root or settings.MAP_STORAGE_ROOT)
        self.url_prefix = url_prefix or getattr(settings, "MAP_TILE_URL_PREFIX", "/tiles/")

    def tile_url(self, tile_path: TilePath) -> str:
        """Собрать публичный URL для тайла."""
        prefix = self.url_prefix.rstrip("/")
        return f"{prefix}/{tile_path.relative}"

    def local_path(self, tile_path: TilePath) -> Path:
        """Локальный путь к тайлу."""
        return self.root / tile_path.relative

    def exists(self, tile_path: TilePath) -> bool:
        """Проверить, что файл тайла существует на диске."""
        return self.local_path(tile_path).is_file()

    def list_layers(self, location_slug: str) -> list[str]:
        """Вернуть slug-ы слоёв, для которых есть папки в локации."""
        location_dir = self.root / location_slug
        if not location_dir.is_dir():
            return []
        return sorted(item.name for item in location_dir.iterdir() if item.is_dir())
