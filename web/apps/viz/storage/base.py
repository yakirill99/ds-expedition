"""Базовый интерфейс хранилища геопространственных тайлов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TilePath:
    """Путь к тайлу внутри локации и слоя."""

    location_slug: str
    layer_slug: str
    z: int
    x: int
    y: int
    extension: str = "png"

    @property
    def relative(self) -> str:
        """Относительный путь в хранилище: location/layer/z/x/y.ext."""
        return f"{self.location_slug}/{self.layer_slug}/{self.z}/{self.x}/{self.y}.{self.extension}"


class MapStorage(Protocol):
    """Абстракция хранилища тайлов карт.

    Реализации должны уметь отдавать публичный URL для тайла и,
    при необходимости, путь на диске для раздачи через веб-сервер.
    """

    def tile_url(self, tile_path: TilePath) -> str:
        """Вернуть URL, по которому клиент запросит тайл."""
        ...

    def exists(self, tile_path: TilePath) -> bool:
        """Проверить наличие тайла в хранилище."""
        ...

    def list_layers(self, location_slug: str) -> list[str]:
        """Вернуть список slug-ов слоёв для локации."""
        ...

    def local_path(self, tile_path: TilePath) -> Path | None:
        """Локальный путь к тайлу или None, если хранилице удалённое."""
        ...
