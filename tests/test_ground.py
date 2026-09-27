"""Тесты фильтра земли на синтетике."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.config import GroundConfig
from expds.features.ground import filter_ground


def _make_config(preset: str = "soft") -> GroundConfig:
    """Конфиг для тестов."""
    return GroundConfig(
        cell_size=1.0,
        preset=preset,
        window_sizes=(3, 5),
        height_threshold=0.5,
        slope_threshold=0.2,
        max_iterations=3,
        min_change_ratio=0.001,
    )


def _make_flat_ground(
    n: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Плоская земля: Z = 0 + шум."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    z = rng.normal(0, 0.05, n)
    return x, y, z


def _make_ground_with_trees(
    n_ground: int = 10_000,
    n_trees: int = 1_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Земля + деревья высотой 10 м. Возвращает x, y, z, is_ground."""
    rng = np.random.default_rng(seed)
    x_g = rng.uniform(0, 100, n_ground)
    y_g = rng.uniform(0, 100, n_ground)
    z_g = rng.normal(0, 0.05, n_ground)

    x_t = rng.uniform(0, 100, n_trees)
    y_t = rng.uniform(0, 100, n_trees)
    z_t = 10.0 + rng.normal(0, 1.0, n_trees)

    x = np.concatenate([x_g, x_t])
    y = np.concatenate([y_g, y_t])
    z = np.concatenate([z_g, z_t])
    is_ground = np.concatenate([np.ones(n_ground, dtype=bool), np.zeros(n_trees, dtype=bool)])
    return x, y, z, is_ground


def _make_ground_with_mound(
    n: int = 20_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Земля + курган (гауссов холм высотой 2 м).

    Возвращает x, y, z, is_ground. is_ground=True для точек,
    которые НЕ на кургане (плоская земля), False — на кургане.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)

    # Курган в центре (50, 50), радиус ~10 м, высота 2 м.
    dx = x - 50.0
    dy = y - 50.0
    mound = 2.0 * np.exp(-(dx * dx + dy * dy) / (2 * 10.0 * 10.0))

    z = mound + rng.normal(0, 0.05, n)
    is_ground = mound < 0.1  # вне кургана
    return x, y, z, is_ground


class TestFlatGround:
    """Плоская земля: почти всё должно быть землёй."""

    def test_flat_mostly_ground(self) -> None:
        x, y, z = _make_flat_ground()
        result = filter_ground(x=x, y=y, z=z, config=_make_config(), crs="EPSG:32617")
        ratio = result.ground_mask.mean()
        assert ratio > 0.9, f"Ожидали >90% земли, получили {ratio:.2%}"

    def test_returns_correct_shapes(self) -> None:
        x, y, z = _make_flat_ground(n=5000)
        result = filter_ground(x=x, y=y, z=z, config=_make_config(), crs="EPSG:32617")
        assert result.ground_mask.shape == (5000,)
        assert result.surface.ndim == 2
        assert result.slope.ndim == 2
        assert result.surface.shape == result.slope.shape


class TestTrees:
    """Деревья должны быть отброшены, земля — сохранена."""

    def test_trees_removed(self) -> None:
        x, y, z, is_ground = _make_ground_with_trees()
        result = filter_ground(
            x=x, y=y, z=z, config=_make_config(preset="strict"), crs="EPSG:32617"
        )
        # Земля: >80% сохранена.
        ground_kept = result.ground_mask[is_ground].mean()
        assert ground_kept > 0.8, f"Земля сохранена на {ground_kept:.2%}"
        # Деревья: >80% отброшено.
        trees_removed = (~result.ground_mask[~is_ground]).mean()
        assert trees_removed > 0.8, f"Деревья отброшены на {trees_removed:.2%}"

    def test_soft_vs_strict(self) -> None:
        x, y, z, _ = _make_ground_with_trees()
        soft = filter_ground(x=x, y=y, z=z, config=_make_config(preset="soft"), crs="EPSG:32617")
        strict = filter_ground(
            x=x, y=y, z=z, config=_make_config(preset="strict"), crs="EPSG:32617"
        )
        # Soft должен оставить больше точек, чем strict.
        assert soft.ground_mask.sum() >= strict.ground_mask.sum()


class TestMound:
    """Курган: мягкий режим должен его частично сохранить."""

    def test_soft_preserves_mound(self) -> None:
        x, y, z, is_ground = _make_ground_with_mound()
        result = filter_ground(x=x, y=y, z=z, config=_make_config(preset="soft"), crs="EPSG:32617")
        # Хотя бы часть точек кургана должна остаться в земле.
        mound_kept = result.ground_mask[~is_ground].mean()
        assert mound_kept > 0.1, f"Курган сохранён на {mound_kept:.2%}"

    def test_ground_still_preserved(self) -> None:
        x, y, z, is_ground = _make_ground_with_mound()
        result = filter_ground(x=x, y=y, z=z, config=_make_config(preset="soft"), crs="EPSG:32617")
        ground_kept = result.ground_mask[is_ground].mean()
        assert ground_kept > 0.8, f"Земля сохранена на {ground_kept:.2%}"


class TestErrors:
    """Ошибочные сценарии."""

    def test_mismatched_lengths(self) -> None:
        x = np.zeros(10)
        y = np.zeros(5)
        z = np.zeros(10)
        with pytest.raises(ValueError, match="Длины"):
            filter_ground(x=x, y=y, z=z, config=_make_config(), crs="EPSG:32617")

    def test_empty_cloud(self) -> None:
        x = np.zeros(0)
        y = np.zeros(0)
        z = np.zeros(0)
        with pytest.raises(ValueError, match="Пустое"):
            filter_ground(x=x, y=y, z=z, config=_make_config(), crs="EPSG:32617")

    def test_unknown_preset(self) -> None:
        config = GroundConfig(
            cell_size=1.0,
            preset="unknown",
            window_sizes=(3,),
            height_threshold=0.5,
            slope_threshold=0.2,
            max_iterations=2,
            min_change_ratio=0.001,
        )
        x, y, z = _make_flat_ground(n=100)
        with pytest.raises(ValueError, match="preset"):
            filter_ground(x=x, y=y, z=z, config=config, crs="EPSG:32617")
