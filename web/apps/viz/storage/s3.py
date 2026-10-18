"""S3-реализация хранилища тайлов (заглушка для будущей миграции)."""

from __future__ import annotations

from django.conf import settings

from .base import TilePath


class S3MapStorage:
    """Хранилище тайлов в S3-совместимом объектном хранилище.

    Пока не реализовано: служит контрактом для перехода с файловой
    системы на S3 без изменений в бизнес-логике приложения.
    """

    def __init__(self, bucket: str | None = None, endpoint_url: str | None = None) -> None:
        self.bucket = bucket or getattr(settings, "MAP_S3_BUCKET", "")
        self.endpoint_url = endpoint_url or getattr(settings, "MAP_S3_ENDPOINT_URL", "")

    def tile_url(self, tile_path: TilePath) -> str:
        """Вернуть прямой URL к объекту в S3."""
        raise NotImplementedError("S3MapStorage.tile_url будет реализован при переходе на S3")

    def local_path(self, tile_path: TilePath) -> None:
        """Для S3 локального пути нет."""
        return None

    def exists(self, tile_path: TilePath) -> bool:
        """Проверить наличие объекта в бакете."""
        raise NotImplementedError("S3MapStorage.exists будет реализован при переходе на S3")

    def list_layers(self, location_slug: str) -> list[str]:
        """Вернуть префиксы слоёв в бакете."""
        raise NotImplementedError("S3MapStorage.list_layers будет реализован при переходе на S3")
