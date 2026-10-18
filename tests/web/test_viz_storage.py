"""Тесты хранилища тайлов карт."""

from __future__ import annotations

from pathlib import Path

import pytest

from web.apps.viz.storage import FileSystemMapStorage, TilePath, get_map_storage
from web.apps.viz.storage.factory import _BACKENDS


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    """Временная корневая папка для тайлов."""
    return tmp_path / "tiles"


@pytest.fixture
def sample_tile() -> TilePath:
    """Пример пути к тайлу."""
    return TilePath(
        location_slug="uchastok-1",
        layer_slug="satellite",
        z=14,
        x=9900,
        y=5000,
        extension="png",
    )


class TestTilePath:
    def test_relative_path(self, sample_tile: TilePath) -> None:
        assert sample_tile.relative == "uchastok-1/satellite/14/9900/5000.png"


class TestFileSystemMapStorage:
    def test_tile_url_uses_prefix(self, storage_root: Path, sample_tile: TilePath) -> None:
        storage = FileSystemMapStorage(root=storage_root, url_prefix="/tiles/")
        assert storage.tile_url(sample_tile) == "/tiles/uchastok-1/satellite/14/9900/5000.png"

    def test_local_path(self, storage_root: Path, sample_tile: TilePath) -> None:
        storage = FileSystemMapStorage(root=storage_root, url_prefix="/tiles/")
        assert storage.local_path(sample_tile) == storage_root / sample_tile.relative

    def test_exists_when_file_present(self, storage_root: Path, sample_tile: TilePath) -> None:
        storage = FileSystemMapStorage(root=storage_root, url_prefix="/tiles/")
        local = storage.local_path(sample_tile)
        local.parent.mkdir(parents=True)
        local.write_bytes(b"fake tile")
        assert storage.exists(sample_tile) is True

    def test_exists_when_file_missing(self, storage_root: Path, sample_tile: TilePath) -> None:
        storage = FileSystemMapStorage(root=storage_root, url_prefix="/tiles/")
        assert storage.exists(sample_tile) is False

    def test_list_layers(self, storage_root: Path) -> None:
        storage = FileSystemMapStorage(root=storage_root, url_prefix="/tiles/")
        (storage_root / "uchastok-1" / "satellite").mkdir(parents=True)
        (storage_root / "uchastok-1" / "lidar").mkdir(parents=True)
        assert storage.list_layers("uchastok-1") == ["lidar", "satellite"]

    def test_list_layers_for_missing_location(self, storage_root: Path) -> None:
        storage = FileSystemMapStorage(root=storage_root, url_prefix="/tiles/")
        assert storage.list_layers("missing") == []


class TestStorageFactory:
    def test_default_backend_is_filesystem(self) -> None:
        storage = get_map_storage()
        assert isinstance(storage, FileSystemMapStorage)

    def test_unknown_backend_raises(self, settings) -> None:
        settings.MAP_STORAGE_BACKEND = "unknown"
        with pytest.raises(ValueError, match="Неизвестный бэкенд"):
            get_map_storage()

    def test_backend_registry_contains_expected_backends(self) -> None:
        assert "filesystem" in _BACKENDS
        assert "s3" in _BACKENDS
