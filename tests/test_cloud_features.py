"""Тесты растровых признаков из облака."""

from __future__ import annotations

import numpy as np
import pytest

from expds.features.cloud_features import (
    FEATURE_MEAN_INTENSITY,
    FEATURE_MEAN_RETURN_NUMBER,
    FEATURE_MEAN_Z,
    FEATURE_NON_FIRST_RETURN_RATIO,
    FEATURE_POINT_DENSITY,
    FEATURE_STD_INTENSITY,
    FEATURE_Z_STD,
    build_cloud_features,
    compute_mean_intensity,
    compute_mean_return_number,
    compute_mean_z,
    compute_non_first_return_ratio,
    compute_point_density,
    compute_std_intensity,
    compute_z_std,
)
from expds.features.grid import Grid


@pytest.fixture()
def grid() -> Grid:
    """Сетка 10x10, pixel_size=1.0, покрывает [0..10] x [0..10]."""
    return Grid(
        crs="EPSG:32617",
        pixel_size=1.0,
        x_min=0.0,
        y_min=0.0,
        width=10,
        height=10,
    )


def _uniform_cloud(
    grid: Grid,
    n_per_cell: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Равномерное облако: n_per_cell точек в каждой ячейке."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for row in range(grid.height):
        for col in range(grid.width):
            x_center = grid.x_min + (col + 0.5) * grid.pixel_size
            y_center = grid.y_min + (row + 0.5) * grid.pixel_size
            xs.append(rng.normal(x_center, 0.1, n_per_cell))
            ys.append(rng.normal(y_center, 0.1, n_per_cell))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = rng.normal(0.0, 1.0, len(x)).astype(np.float32)
    intensity = rng.uniform(0, 100, len(x)).astype(np.float32)
    return_number = rng.integers(1, 4, len(x)).astype(np.uint8)
    return {
        "x": x,
        "y": y,
        "z": z,
        "intensity": intensity,
        "return_number": return_number,
    }


class TestPointDensity:
    """Тесты плотности."""

    def test_uniform_density(self, grid: Grid) -> None:
        cloud = _uniform_cloud(grid=grid, n_per_cell=5)
        layer = compute_point_density(x=cloud["x"], y=cloud["y"], grid=grid)
        # Большинство ячеек заполнено (краевые могут промахнуться).
        valid = layer.data != layer.nodata
        assert valid.mean() > 0.9
        # Средняя плотность по непустым близка к 5 точек/м².
        assert layer.data[valid].mean() == pytest.approx(5.0, abs=1.0)

    def test_empty_cells_nodata(self, grid: Grid) -> None:
        # Точки только в одной ячейке.
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        layer = compute_point_density(x=x, y=y, grid=grid)
        assert layer.data[0, 0] == pytest.approx(3.0)
        assert layer.data[5, 5] == layer.nodata

    def test_wrong_unit(self, grid: Grid) -> None:
        cloud = _uniform_cloud(grid=grid, n_per_cell=5)
        layer = compute_point_density(x=cloud["x"], y=cloud["y"], grid=grid)
        assert layer.unit == "points/m^2"


class TestMeanIntensity:
    """Тесты средней интенсивности."""

    def test_constant_intensity(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        intensity = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        layer = compute_mean_intensity(x=x, y=y, intensity=intensity, grid=grid)
        assert layer.data[0, 0] == pytest.approx(50.0)

    def test_mean_of_values(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        layer = compute_mean_intensity(x=x, y=y, intensity=intensity, grid=grid)
        assert layer.data[0, 0] == pytest.approx(20.0)


class TestStdIntensity:
    """Тесты std интенсивности."""

    def test_constant_zero_std(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        intensity = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        layer = compute_std_intensity(x=x, y=y, intensity=intensity, grid=grid)
        assert layer.data[0, 0] == pytest.approx(0.0)

    def test_known_std(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5, 0.5])
        intensity = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float32)
        layer = compute_std_intensity(x=x, y=y, intensity=intensity, grid=grid)
        # std = 5.
        assert layer.data[0, 0] == pytest.approx(5.0, abs=1e-3)


class TestNonFirstReturnRatio:
    """Тесты доли непервых отражений."""

    def test_all_first(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        return_number = np.array([1, 1, 1], dtype=np.uint8)
        layer = compute_non_first_return_ratio(x=x, y=y, return_number=return_number, grid=grid)
        assert layer.data[0, 0] == pytest.approx(0.0)

    def test_half_non_first(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5, 0.5])
        return_number = np.array([1, 2, 1, 2], dtype=np.uint8)
        layer = compute_non_first_return_ratio(x=x, y=y, return_number=return_number, grid=grid)
        assert layer.data[0, 0] == pytest.approx(0.5)

    def test_all_non_first(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5])
        y = np.array([0.5, 0.5])
        return_number = np.array([2, 3], dtype=np.uint8)
        layer = compute_non_first_return_ratio(x=x, y=y, return_number=return_number, grid=grid)
        assert layer.data[0, 0] == pytest.approx(1.0)


class TestMeanReturnNumber:
    """Тесты среднего номера отражения."""

    def test_mean(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        return_number = np.array([1, 2, 3], dtype=np.uint8)
        layer = compute_mean_return_number(x=x, y=y, return_number=return_number, grid=grid)
        assert layer.data[0, 0] == pytest.approx(2.0)


class TestMeanZ:
    """Тесты средней высоты."""

    def test_mean_z(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        z = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        layer = compute_mean_z(x=x, y=y, z=z, grid=grid)
        assert layer.data[0, 0] == pytest.approx(20.0)

    def test_constant_z(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5])
        y = np.array([0.5, 0.5])
        z = np.array([5.0, 5.0], dtype=np.float32)
        layer = compute_mean_z(x=x, y=y, z=z, grid=grid)
        assert layer.data[0, 0] == pytest.approx(5.0)


class TestZStd:
    """Тесты разброса высот."""

    def test_constant_zero_std(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        z = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        layer = compute_z_std(x=x, y=y, z=z, grid=grid)
        assert layer.data[0, 0] == pytest.approx(0.0)

    def test_known_std(self, grid: Grid) -> None:
        x = np.array([0.5, 0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5, 0.5])
        z = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float32)
        layer = compute_z_std(x=x, y=y, z=z, grid=grid)
        assert layer.data[0, 0] == pytest.approx(5.0, abs=1e-3)


class TestBuildCloudFeatures:
    """Тесты сборки всех признаков."""

    def test_all_features(self, grid: Grid) -> None:
        cloud = _uniform_cloud(grid=grid, n_per_cell=5)
        result = build_cloud_features(
            x=cloud["x"],
            y=cloud["y"],
            z=cloud["z"],
            intensity=cloud["intensity"],
            return_number=cloud["return_number"],
            grid=grid,
            enabled=(
                FEATURE_POINT_DENSITY,
                FEATURE_MEAN_INTENSITY,
                FEATURE_STD_INTENSITY,
                FEATURE_NON_FIRST_RETURN_RATIO,
                FEATURE_MEAN_RETURN_NUMBER,
                FEATURE_MEAN_Z,
                FEATURE_Z_STD,
            ),
        )
        assert len(result.layers) == 7
        for _name, layer in result.layers.items():
            assert layer.grid == grid
            assert layer.data.shape == grid.shape
            assert layer.data.dtype == np.float32

    def test_subset_of_features(self, grid: Grid) -> None:
        cloud = _uniform_cloud(grid=grid, n_per_cell=5)
        result = build_cloud_features(
            x=cloud["x"],
            y=cloud["y"],
            z=cloud["z"],
            intensity=cloud["intensity"],
            return_number=cloud["return_number"],
            grid=grid,
            enabled=(FEATURE_POINT_DENSITY, FEATURE_MEAN_INTENSITY),
        )
        assert set(result.layers.keys()) == {
            FEATURE_POINT_DENSITY,
            FEATURE_MEAN_INTENSITY,
        }

    def test_mismatched_lengths(self, grid: Grid) -> None:
        with pytest.raises(ValueError, match="Длины"):
            build_cloud_features(
                x=np.zeros(10),
                y=np.zeros(5),
                z=np.zeros(10),
                intensity=np.zeros(10),
                return_number=np.zeros(10, dtype=np.uint8),
                grid=grid,
                enabled=(FEATURE_POINT_DENSITY,),
            )

    def test_empty_cloud(self, grid: Grid) -> None:
        with pytest.raises(ValueError, match="Пустое"):
            build_cloud_features(
                x=np.zeros(0),
                y=np.zeros(0),
                z=np.zeros(0),
                intensity=np.zeros(0),
                return_number=np.zeros(0, dtype=np.uint8),
                grid=grid,
                enabled=(FEATURE_POINT_DENSITY,),
            )

    def test_unknown_feature(self, grid: Grid) -> None:
        cloud = _uniform_cloud(grid=grid, n_per_cell=1)
        with pytest.raises(ValueError, match="Неизвестные"):
            build_cloud_features(
                x=cloud["x"],
                y=cloud["y"],
                z=cloud["z"],
                intensity=cloud["intensity"],
                return_number=cloud["return_number"],
                grid=grid,
                enabled=("nonexistent",),
            )

    def test_empty_enabled_list(self, grid: Grid) -> None:
        cloud = _uniform_cloud(grid=grid, n_per_cell=1)
        result = build_cloud_features(
            x=cloud["x"],
            y=cloud["y"],
            z=cloud["z"],
            intensity=cloud["intensity"],
            return_number=cloud["return_number"],
            grid=grid,
            enabled=(),
        )
        assert result.layers == {}
