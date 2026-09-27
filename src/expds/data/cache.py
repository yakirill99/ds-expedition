"""Кэш промежуточных растровых слоёв на диск.

Идея: дорогие вычисления (DTM, производные рельефа, геофизика) кэшируются
по ключу, который зависит от имени слоя, параметров и сетки. Если параметры
не менялись — слой читается с диска. Если менялись — кэш инвалидируется
автоматически.

Хранилище: <cache_dir>/<key>.tif (данные) + <cache_dir>/<key>.json (метаданные).
Ключ — SHA256 от сериализованных параметров и версии формата кэша.

Атомарность: запись во временный файл + os.replace. Так параллельные
прогоны и падения не оставляют битых файлов.

Кэш не знает про семантику слоёв: он сериализует параметры и сравнивает.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from expds.data.raster import read_raster, write_raster
from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)

# Версия формата кэша. Меняем при breaking changes — старый кэш игнорируется.
_CACHE_VERSION = "v1"

# Константы: имена файлов и суффиксы.
_DATA_SUFFIX = ".tif"
_META_SUFFIX = ".json"
_TMP_SUFFIX = ".tmp"

# Константы: имена полей в метаданных.
_META_FIELD_VERSION = "cache_version"
_META_FIELD_KEY = "key"
_META_FIELD_NAME = "name"
_META_FIELD_GRID = "grid"
_META_FIELD_NODATA = "nodata"
_META_FIELD_SOURCE = "source"
_META_FIELD_LICENSE = "license"
_META_FIELD_UNIT = "unit"


@dataclass(frozen=True)
class CacheEntry:
    """Запись о файле в кэше."""

    key: str
    name: str
    data_path: Path
    meta_path: Path


def _canonical_json(value: Any) -> str:
    """Сериализует значение в канонический JSON.

    Ключи сортируются, пробелы убираются. Это гарантирует, что одинаковые
    по смыслу параметры дадут одинаковую строку.

    Args:
        value: произвольное JSON-сериализуемое значение.

    Returns:
        Строка канонического JSON.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _grid_to_dict(grid: Grid) -> dict[str, Any]:
    """Представление Grid для сериализации.

    Args:
        grid: сетка.

    Returns:
        Словарь с полями сетки.
    """
    return {
        "crs": grid.crs,
        "pixel_size": grid.pixel_size,
        "x_min": grid.x_min,
        "y_min": grid.y_min,
        "width": grid.width,
        "height": grid.height,
    }


def _dict_to_grid(data: dict[str, Any]) -> Grid:
    """Восстанавливает Grid из словаря.

    Args:
        data: словарь с полями сетки.

    Returns:
        Grid.
    """
    return Grid(
        crs=str(data["crs"]),
        pixel_size=float(data["pixel_size"]),
        x_min=float(data["x_min"]),
        y_min=float(data["y_min"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )


def compute_cache_key(
    name: str,
    params: dict[str, Any],
    grid: Grid,
) -> str:
    """Считает ключ кэша.

    Ключ зависит от имени слоя, параметров, сетки и версии формата.
    Изменение любого из них даёт новый ключ.

    Args:
        name: имя слоя (например, "dtm").
        params: параметры вычисления (например, {"method": "idw", "k": 8}).
        grid: сетка слоя.

    Returns:
        Строка SHA256 (64 hex-символа).
    """
    payload = {
        "version": _CACHE_VERSION,
        "name": name,
        "params": params,
        "grid": _grid_to_dict(grid),
    }
    serialized = _canonical_json(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class LayerCache:
    """Кэш растровых слоёв на диске.

    Использование:
        cache = LayerCache(root=Path("data/cache"))
        layer = cache.get_or_compute(
            name="dtm",
            params={"method": "idw"},
            grid=grid,
            compute_fn=lambda: build_dtm(...),
        )
    """

    def __init__(self, root: Path) -> None:
        """Инициализирует кэш.

        Args:
            root: корневая папка кэша. Создаётся, если её нет.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        logger.debug("Кэш инициализирован: %s", self._root)

    @property
    def root(self) -> Path:
        """Корневая папка кэша."""
        return self._root

    def _paths(self, key: str) -> tuple[Path, Path]:
        """Возвращает пути к файлам данных и метаданных.

        Args:
            key: ключ кэша.

        Returns:
            Кортеж (data_path, meta_path).
        """
        return (
            self._root / f"{key}{_DATA_SUFFIX}",
            self._root / f"{key}{_META_SUFFIX}",
        )

    def _write_meta(self, meta_path: Path, meta: dict[str, Any]) -> None:
        """Атомарно пишет JSON-метаданные.

        Args:
            meta_path: путь к файлу метаданных.
            meta: словарь метаданных.
        """
        tmp_path = meta_path.with_suffix(meta_path.suffix + _TMP_SUFFIX)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, meta_path)

    def _read_meta(self, meta_path: Path) -> dict[str, Any]:
        """Читает JSON-метаданные.

        Args:
            meta_path: путь к файлу метаданных.

        Returns:
            Словарь метаданных.
        """
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _is_valid(self, data_path: Path, meta_path: Path, key: str) -> bool:
        """Проверяет, что кэш-файлы на месте и версия совпадает.

        Args:
            data_path: путь к файлу данных.
            meta_path: путь к файлу метаданных.
            key: ожидаемый ключ.

        Returns:
            True, если кэш валиден.
        """
        if not data_path.exists() or not meta_path.exists():
            return False
        try:
            meta = self._read_meta(meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Битые метаданные кэша %s: %s", meta_path.name, exc)
            return False
        if meta.get(_META_FIELD_VERSION) != _CACHE_VERSION:
            logger.debug(
                "Версия кэша не совпадает: %s != %s",
                meta.get(_META_FIELD_VERSION),
                _CACHE_VERSION,
            )
            return False
        if meta.get(_META_FIELD_KEY) != key:
            logger.debug("Ключ в метаданных не совпадает с ожидаемым")
            return False
        return True

    def _load(self, key: str) -> Layer:
        """Читает слой из кэша.

        Args:
            key: ключ кэша.

        Returns:
            Layer, восстановленный из кэша.
        """
        data_path, meta_path = self._paths(key)
        meta = self._read_meta(meta_path)
        grid = _dict_to_grid(meta[_META_FIELD_GRID])
        layer = read_raster(path=data_path, grid=grid)
        # Пересобираем Layer с метаданными из кэша.
        return Layer(
            name=str(meta[_META_FIELD_NAME]),
            data=layer.data,
            grid=layer.grid,
            nodata=float(meta[_META_FIELD_NODATA]),
            source=str(meta[_META_FIELD_SOURCE]),
            license=str(meta[_META_FIELD_LICENSE]),
            unit=str(meta[_META_FIELD_UNIT]),
        )

    def _save(self, key: str, layer: Layer) -> None:
        """Сохраняет слой в кэш атомарно.

        Сначала пишутся данные во временный файл, потом метаданные,
        потом оба переименовываются через os.replace.

        Args:
            key: ключ кэша.
            layer: слой для сохранения.
        """
        data_path, meta_path = self._paths(key)
        tmp_data = data_path.with_suffix(data_path.suffix + _TMP_SUFFIX)

        # 1. Пишем данные во временный файл.
        write_raster(layer=layer, path=tmp_data)
        os.replace(tmp_data, data_path)

        # 2. Пишем метаданные.
        meta = {
            _META_FIELD_VERSION: _CACHE_VERSION,
            _META_FIELD_KEY: key,
            _META_FIELD_NAME: layer.name,
            _META_FIELD_GRID: _grid_to_dict(layer.grid),
            _META_FIELD_NODATA: layer.nodata,
            _META_FIELD_SOURCE: layer.source,
            _META_FIELD_LICENSE: layer.license,
            _META_FIELD_UNIT: layer.unit,
        }
        self._write_meta(meta_path=meta_path, meta=meta)

    def get_or_compute(
        self,
        name: str,
        params: dict[str, Any],
        grid: Grid,
        compute_fn: Callable[[], Layer],
        use_cache: bool = True,
    ) -> Layer:
        """Возвращает слой из кэша или вычисляет его.

        Args:
            name: имя слоя.
            params: параметры вычисления.
            grid: целевая сетка. compute_fn обязан вернуть слой в этой сетке.
            compute_fn: функция без аргументов, возвращающая Layer.
            use_cache: если False — всегда пересчитывать и перезаписывать.

        Returns:
            Layer (из кэша или свежевычисленный).

        Raises:
            ValueError: если compute_fn вернул слой с другой сеткой.
        """
        key = compute_cache_key(name=name, params=params, grid=grid)
        data_path, meta_path = self._paths(key)

        if use_cache and self._is_valid(data_path=data_path, meta_path=meta_path, key=key):
            logger.info("Кэш hit: %s (%s)", name, key[:12])
            return self._load(key=key)

        logger.info("Кэш miss: %s (%s), вычисляем", name, key[:12])
        layer = compute_fn()

        if layer.grid != grid:
            raise ValueError(
                f"compute_fn вернул слой '{name}' с grid, отличным от запрошенного. "
                f"Ожидалось {grid}, получено {layer.grid}"
            )
        if layer.name != name:
            logger.warning(
                "compute_fn вернул слой с именем '%s', ожидалось '%s'. Переименовываем в '%s'.",
                layer.name,
                name,
                name,
            )
            layer = Layer(
                name=name,
                data=layer.data,
                grid=layer.grid,
                nodata=layer.nodata,
                source=layer.source,
                license=layer.license,
                unit=layer.unit,
            )

        if use_cache:
            self._save(key=key, layer=layer)
            logger.info("Кэш записан: %s (%s)", name, key[:12])

        return layer

    def invalidate(self, name: str | None = None) -> int:
        """Удаляет записи кэша.

        Если name задан — удаляет только записи с этим именем.
        Если None — удаляет весь кэш.

        Args:
            name: имя слоя или None.

        Returns:
            Число удалённых записей (пар файлов).
        """
        removed = 0
        for entry in self.list_entries():
            if name is not None and entry.name != name:
                continue
            entry.data_path.unlink(missing_ok=True)
            entry.meta_path.unlink(missing_ok=True)
            removed += 1

        logger.info("Кэш очищен: удалено %d записей (name=%s)", removed, name)
        return removed

    def clear(self) -> None:
        """Полностью очищает кэш (удаляет папку и создаёт заново)."""
        if self._root.exists():
            shutil.rmtree(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("Кэш полностью очищен: %s", self._root)

    def list_entries(self) -> list[CacheEntry]:
        """Возвращает список валидных записей кэша.

        Returns:
            Список CacheEntry, отсортированный по имени слоя.
        """
        entries: list[CacheEntry] = []
        for meta_path in self._root.glob(f"*{_META_SUFFIX}"):
            key = meta_path.stem
            data_path = self._root / f"{key}{_DATA_SUFFIX}"
            if not data_path.exists():
                continue
            try:
                meta = self._read_meta(meta_path)
            except (OSError, json.JSONDecodeError):
                continue
            entries.append(
                CacheEntry(
                    key=key,
                    name=str(meta.get(_META_FIELD_NAME, "unknown")),
                    data_path=data_path,
                    meta_path=meta_path,
                )
            )
        entries.sort(key=lambda e: (e.name, e.key))
        return entries

    def size_bytes(self) -> int:
        """Суммарный размер файлов кэша в байтах.

        Returns:
            Размер в байтах.
        """
        total = 0
        for entry in self.list_entries():
            total += entry.data_path.stat().st_size
            total += entry.meta_path.stat().st_size
        return total
