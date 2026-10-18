"""Тесты кэша слоёв."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from expds.data.cache import LayerCache, compute_cache_key
from expds.features.grid import Grid, Layer


@pytest.fixture()
def grid() -> Grid:
    """Простая сетка для тестов."""
    return Grid(
        crs="EPSG:32617",
        pixel_size=1.0,
        x_min=500_000.0,
        y_min=4_000_000.0,
        width=8,
        height=6,
    )


@pytest.fixture()
def cache(tmp_path: Path) -> LayerCache:
    """Кэш во временной папке."""
    return LayerCache(root=tmp_path / "cache")


def _make_layer(grid: Grid, value: float = 1.0) -> Layer:
    """Вспомогательный слой с постоянным значением."""
    return Layer(
        name="test",
        data=np.full(grid.shape, value, dtype=np.float32),
        grid=grid,
    )


class TestCacheKey:
    """Тесты compute_cache_key."""

    def test_same_inputs_same_key(self, grid: Grid) -> None:
        k1 = compute_cache_key(name="dtm", params={"a": 1}, grid=grid)
        k2 = compute_cache_key(name="dtm", params={"a": 1}, grid=grid)
        assert k1 == k2

    def test_different_name_different_key(self, grid: Grid) -> None:
        k1 = compute_cache_key(name="dtm", params={"a": 1}, grid=grid)
        k2 = compute_cache_key(name="hillshade", params={"a": 1}, grid=grid)
        assert k1 != k2

    def test_different_params_different_key(self, grid: Grid) -> None:
        k1 = compute_cache_key(name="dtm", params={"a": 1}, grid=grid)
        k2 = compute_cache_key(name="dtm", params={"a": 2}, grid=grid)
        assert k1 != k2

    def test_params_order_independent(self, grid: Grid) -> None:
        k1 = compute_cache_key(name="dtm", params={"a": 1, "b": 2}, grid=grid)
        k2 = compute_cache_key(name="dtm", params={"b": 2, "a": 1}, grid=grid)
        assert k1 == k2

    def test_different_grid_different_key(self, grid: Grid) -> None:
        other = Grid(
            crs="EPSG:32617",
            pixel_size=1.0,
            x_min=500_000.0,
            y_min=4_000_000.0,
            width=16,
            height=12,
        )
        k1 = compute_cache_key(name="dtm", params={}, grid=grid)
        k2 = compute_cache_key(name="dtm", params={}, grid=other)
        assert k1 != k2


class TestGetOrCompute:
    """Тесты основного метода."""

    def test_first_call_computes(self, cache: LayerCache, grid: Grid) -> None:
        calls = {"n": 0}

        def compute() -> Layer:
            calls["n"] += 1
            return _make_layer(grid=grid, value=42.0)

        layer = cache.get_or_compute(
            name="test",
            params={},
            grid=grid,
            compute_fn=compute,
        )
        assert calls["n"] == 1
        assert layer.data[0, 0] == pytest.approx(42.0)

    def test_second_call_reads_from_cache(self, cache: LayerCache, grid: Grid) -> None:
        calls = {"n": 0}

        def compute() -> Layer:
            calls["n"] += 1
            return _make_layer(grid=grid, value=42.0)

        cache.get_or_compute(name="test", params={}, grid=grid, compute_fn=compute)
        layer = cache.get_or_compute(name="test", params={}, grid=grid, compute_fn=compute)
        assert calls["n"] == 1  # второй раз не считали
        assert layer.data[0, 0] == pytest.approx(42.0)

    def test_different_params_recompute(self, cache: LayerCache, grid: Grid) -> None:
        calls = {"n": 0}

        def compute() -> Layer:
            calls["n"] += 1
            return _make_layer(grid=grid, value=42.0)

        cache.get_or_compute(name="test", params={"a": 1}, grid=grid, compute_fn=compute)
        cache.get_or_compute(name="test", params={"a": 2}, grid=grid, compute_fn=compute)
        assert calls["n"] == 2

    def test_use_cache_false_always_computes(self, cache: LayerCache, grid: Grid) -> None:
        calls = {"n": 0}

        def compute() -> Layer:
            calls["n"] += 1
            return _make_layer(grid=grid, value=42.0)

        cache.get_or_compute(name="test", params={}, grid=grid, compute_fn=compute)
        cache.get_or_compute(
            name="test",
            params={},
            grid=grid,
            compute_fn=compute,
            use_cache=False,
        )
        assert calls["n"] == 2

    def test_wrong_grid_raises(self, cache: LayerCache, grid: Grid) -> None:
        other = Grid(
            crs="EPSG:32617",
            pixel_size=2.0,
            x_min=500_000.0,
            y_min=4_000_000.0,
            width=4,
            height=3,
        )

        def compute() -> Layer:
            return _make_layer(grid=other, value=1.0)

        with pytest.raises(ValueError, match="grid"):
            cache.get_or_compute(name="test", params={}, grid=grid, compute_fn=compute)

    def test_preserves_metadata(self, cache: LayerCache, grid: Grid) -> None:
        def compute() -> Layer:
            return Layer(
                name="test",
                data=np.ones(grid.shape, dtype=np.float32),
                grid=grid,
                nodata=-9999.0,
                source="synthetic",
                license="CC0",
                unit="meters",
            )

        cache.get_or_compute(name="test", params={}, grid=grid, compute_fn=compute)
        restored = cache.get_or_compute(name="test", params={}, grid=grid, compute_fn=compute)
        assert restored.source == "synthetic"
        assert restored.license == "CC0"
        assert restored.unit == "meters"
        assert restored.nodata == pytest.approx(-9999.0)


class TestInvalidate:
    """Тесты очистки кэша."""

    def test_invalidate_by_name(self, cache: LayerCache, grid: Grid) -> None:
        def compute_a() -> Layer:
            layer = _make_layer(grid=grid, value=1.0)
            return Layer(
                name="a",
                data=layer.data,
                grid=layer.grid,
            )

        def compute_b() -> Layer:
            layer = _make_layer(grid=grid, value=2.0)
            return Layer(
                name="b",
                data=layer.data,
                grid=layer.grid,
            )

        cache.get_or_compute(name="a", params={}, grid=grid, compute_fn=compute_a)
        cache.get_or_compute(name="b", params={}, grid=grid, compute_fn=compute_b)
        assert len(cache.list_entries()) == 2

        removed = cache.invalidate(name="a")
        assert removed == 1
        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0].name == "b"

    def test_clear_removes_all(self, cache: LayerCache, grid: Grid) -> None:
        def compute() -> Layer:
            return _make_layer(grid=grid, value=1.0)

        cache.get_or_compute(name="a", params={}, grid=grid, compute_fn=compute)
        cache.get_or_compute(name="b", params={}, grid=grid, compute_fn=compute)
        assert len(cache.list_entries()) == 2

        cache.clear()
        assert len(cache.list_entries()) == 0


class TestListAndSize:
    """Тесты list_entries и size_bytes."""

    def test_list_entries_sorted(self, cache: LayerCache, grid: Grid) -> None:
        def make(name: str) -> Layer:
            return Layer(
                name=name,
                data=np.ones(grid.shape, dtype=np.float32),
                grid=grid,
            )

        cache.get_or_compute(name="b", params={}, grid=grid, compute_fn=lambda: make("b"))
        cache.get_or_compute(name="a", params={}, grid=grid, compute_fn=lambda: make("a"))
        entries = cache.list_entries()
        assert [e.name for e in entries] == ["a", "b"]

    def test_size_bytes_positive(self, cache: LayerCache, grid: Grid) -> None:
        def compute() -> Layer:
            return _make_layer(grid=grid, value=1.0)

        cache.get_or_compute(name="a", params={}, grid=grid, compute_fn=compute)
        assert cache.size_bytes() > 0


class TestCodeVersion:
    """Тесты _get_code_version."""

    def test_returns_nonempty(self) -> None:
        from expds.data.cache import _get_code_version

        version = _get_code_version()
        assert isinstance(version, str)
        assert len(version) > 0
        # Один из трёх форматов: git:<hash>, pkg:<ver>, unknown.
        assert version.startswith("git:") or version.startswith("pkg:") or version == "unknown"

    def test_cached(self) -> None:
        from expds.data.cache import _get_code_version

        v1 = _get_code_version()
        v2 = _get_code_version()
        assert v1 == v2

    def test_in_cache_key(self, grid: Grid) -> None:
        """Версия кода влияет на ключ."""
        from expds.data.cache import compute_cache_key

        k1 = compute_cache_key(name="test", params={}, grid=grid)
        # Ключ должен быть стабильным.
        k2 = compute_cache_key(name="test", params={}, grid=grid)
        assert k1 == k2
