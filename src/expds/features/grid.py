"""Контракты данных: Grid, Layer, Sample, DataSpec.

Grid — общая растровая сетка (CRS, pixel_size, extent, shape).
Layer — один растровый слой в Grid с метаданными.
Sample — тензор [C, H, W] + маска каналов, готовый для модели.
DataSpec — описание датасета: сетка, список каналов, метаданные.

Все dataclass'ы frozen: неизменяемость контрактов — часть дисциплины.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Константы: значения по умолчанию и имена.
_DEFAULT_NODATA = -9999.0
_DEFAULT_PIXEL_SIZE = 1.0

# Допуск для проверки целочисленности пиксельных координат.
_PIXEL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Grid:
    """Общая растровая сетка.

    Определяет, в каком CRS, с каким разрешением и в каких границах
    живут все растровые слои проекта. Pixel_size — в метрах (или в
    единицах CRS, если CRS не метрическая).

    Пиксель (col, row) — центр ячейки:
        world_x = x_min + (col + 0.5) * pixel_size
        world_y = y_min + (row + 0.5) * pixel_size

    Левый нижний угол сетки — (x_min, y_min). Row 0 — снизу.
    """

    crs: str
    pixel_size: float
    x_min: float
    y_min: float
    width: int
    height: int

    def __post_init__(self) -> None:
        """Валидация параметров сетки."""
        if self.pixel_size <= 0:
            raise ValueError(f"pixel_size должен быть > 0, получено {self.pixel_size}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width/height должны быть > 0, получено {self.width}x{self.height}")
        if not self.crs:
            raise ValueError("crs не может быть пустым")

    @property
    def shape(self) -> tuple[int, int]:
        """Форма массива: (height, width)."""
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Границы: (x_min, y_min, x_max, y_max)."""
        return (
            self.x_min,
            self.y_min,
            self.x_min + self.width * self.pixel_size,
            self.y_min + self.height * self.pixel_size,
        )

    @property
    def x_max(self) -> float:
        """Правый край сетки."""
        return self.x_min + self.width * self.pixel_size

    @property
    def y_max(self) -> float:
        """Верхний край сетки."""
        return self.y_min + self.height * self.pixel_size

    def transform_to_pixel(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Переводит мировые координаты в пиксельные (центр ячейки).

        Args:
            x: мировая координата X или массив.
            y: мировая координата Y или массив.

        Returns:
            Кортеж (col, row) — пиксельные координаты в виде float.
            Целочисленная часть — индекс ячейки, дробная — положение внутри.
        """
        col = (np.asarray(x) - self.x_min) / self.pixel_size - 0.5
        row = (np.asarray(y) - self.y_min) / self.pixel_size - 0.5
        return col, row

    def transform_to_world(
        self,
        col: np.ndarray | float,
        row: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Переводит пиксельные координаты (центр ячейки) в мировые.

        Args:
            col: пиксельная координата X (столбец) или массив.
            row: пиксельная координата Y (строка) или массив.

        Returns:
            Кортеж (x, y) — мировые координаты.
        """
        x = self.x_min + (np.asarray(col) + 0.5) * self.pixel_size
        y = self.y_min + (np.asarray(row) + 0.5) * self.pixel_size
        return x, y

    @classmethod
    def from_bounds(
        cls,
        crs: str,
        pixel_size: float,
        bounds: tuple[float, float, float, float],
    ) -> "Grid":
        """Строит сетку по границам.

        Границы расширяются вниз-влево, чтобы вместить весь extent
        при заданном pixel_size. width/height округляются вверх.

        Args:
            crs: CRS сетки.
            pixel_size: размер пикселя в единицах CRS.
            bounds: (x_min, y_min, x_max, y_max).

        Returns:
            Grid, покрывающий bounds.
        """
        x_min, y_min, x_max, y_max = bounds
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Некорректные bounds: {bounds}")

        width = int(np.ceil((x_max - x_min) / pixel_size))
        height = int(np.ceil((y_max - y_min) / pixel_size))

        logger.debug(
            "Grid.from_bounds: bounds=%s, pixel_size=%.3f -> %dx%d",
            bounds,
            pixel_size,
            width,
            height,
        )
        return cls(
            crs=crs,
            pixel_size=pixel_size,
            x_min=x_min,
            y_min=y_min,
            width=width,
            height=height,
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Проверяет, попадает ли точка в границы сетки.

        Args:
            x: мировая координата X.
            y: мировая координата Y.

        Returns:
            True, если точка внутри [x_min, x_max] × [y_min, y_max].
        """
        bx_min, by_min, bx_max, by_max = self.bounds
        return (bx_min <= x <= bx_max) and (by_min <= y <= by_max)


@dataclass(frozen=True)
class Layer:
    """Один растровый слой в общей сетке.

    data — 2D массив (height, width) типа float32. Значения "нет данных"
    кодируются nodata. Метаданные source/license/unit нужны для реестра
    лицензий и для отладки: откуда слой, в каких единицах.
    """

    name: str
    data: np.ndarray
    grid: Grid
    nodata: float = _DEFAULT_NODATA
    source: str = "unknown"
    license: str = "unknown"
    unit: str = "unitless"

    def __post_init__(self) -> None:
        """Валидация: форма массива совпадает с сеткой, тип float32."""
        if self.data.ndim != 2:
            raise ValueError(
                f"Layer '{self.name}': data должен быть 2D, получено {self.data.ndim}D"
            )
        if self.data.shape != self.grid.shape:
            raise ValueError(
                f"Layer '{self.name}': shape {self.data.shape} "
                f"не совпадает с grid {self.grid.shape}"
            )
        if self.data.dtype != np.float32:
            raise ValueError(
                f"Layer '{self.name}': dtype должен быть float32, получено {self.data.dtype}"
            )

    @property
    def valid_mask(self) -> np.ndarray:
        """Булева маска валидных пикселей (не nodata)."""
        return self.data != self.nodata


@dataclass(frozen=True)
class Sample:
    """Готовый для модели пример.

    tensor — [C, H, W] float32. channel_names — имена каналов в том же
    порядке. channel_mask — [C] bool: какие каналы реально есть (для
    отсутствующих модальностей — False). target — [H, W] float32 или None.
    """

    tensor: np.ndarray
    channel_names: tuple[str, ...]
    channel_mask: np.ndarray
    grid: Grid
    target: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Валидация размерностей и согласованности."""
        if self.tensor.ndim != 3:
            raise ValueError(f"Sample: tensor должен быть 3D [C,H,W], получено {self.tensor.ndim}D")
        n_channels = self.tensor.shape[0]
        if len(self.channel_names) != n_channels:
            raise ValueError(
                f"Sample: channel_names ({len(self.channel_names)}) не совпадает с C ({n_channels})"
            )
        if self.channel_mask.shape != (n_channels,):
            raise ValueError(
                f"Sample: channel_mask shape {self.channel_mask.shape} "
                f"не совпадает с ({n_channels},)"
            )
        if self.tensor.shape[1:] != self.grid.shape:
            raise ValueError(
                f"Sample: tensor HxW {self.tensor.shape[1:]} не совпадает с grid {self.grid.shape}"
            )
        if self.target is not None and self.target.shape != self.grid.shape:
            raise ValueError(
                f"Sample: target shape {self.target.shape} не совпадает с grid {self.grid.shape}"
            )

    @property
    def n_channels(self) -> int:
        """Число каналов в тензоре."""
        return self.tensor.shape[0]

    @property
    def n_available_channels(self) -> int:
        """Сколько каналов реально доступно."""
        return int(self.channel_mask.sum())


@dataclass(frozen=True)
class DataSpec:
    """Описание датасета: сетка, ожидаемые каналы, метаданные.

    Используется как контракт между препроцессингом и моделью: модель
    знает, сколько каналов ждать и в каком порядке, но не знает, откуда
    они взялись. Смена датасета = смена DataSpec через YAML.
    """

    grid: Grid
    channel_names: tuple[str, ...]
    nodata: float = _DEFAULT_NODATA
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация: имена каналов уникальны."""
        if len(set(self.channel_names)) != len(self.channel_names):
            duplicates = [n for n in self.channel_names if self.channel_names.count(n) > 1]
            raise ValueError(f"DataSpec: дублирующиеся имена каналов: {set(duplicates)}")

    def channel_index(self, name: str) -> int:
        """Возвращает индекс канала по имени.

        Args:
            name: имя канала.

        Returns:
            Индекс в channel_names.

        Raises:
            KeyError: если канала нет.
        """
        try:
            return self.channel_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Канал '{name}' не найден. Доступны: {self.channel_names}") from exc


def stack_layers(
    layers: Iterable[Layer],
    spec: DataSpec,
) -> Sample:
    """Собирает Sample из набора слоёв по спецификации.

    Каждый слой из layers кладётся в свой канал. Если какого-то канала
    из spec нет в layers — он заполняется нулями, а в channel_mask
    ставится False. Лишние слои (не из spec) игнорируются с warning.

    Args:
        layers: итерируемое по Layer.
        spec: DataSpec с ожидаемым порядком каналов.

    Returns:
        Sample с тензором [C, H, W], именами каналов, маской и target=None.
    """
    layers_by_name = {layer.name: layer for layer in layers}
    n_channels = len(spec.channel_names)
    height, width = spec.grid.shape

    tensor = np.zeros((n_channels, height, width), dtype=np.float32)
    channel_mask = np.zeros((n_channels,), dtype=bool)

    for idx, name in enumerate(spec.channel_names):
        layer = layers_by_name.get(name)
        if layer is None:
            logger.warning("Канал '%s' отсутствует, заполняем нулями (mask=False)", name)
            continue

        if layer.grid != spec.grid:
            raise ValueError(
                f"Layer '{name}': grid не совпадает с DataSpec.grid. "
                f"Сначала приведи слой к общей сетке через prep/align."
            )

        # nodata -> 0, валидные -> как есть.
        data = layer.data.copy()
        data[~layer.valid_mask] = 0.0
        tensor[idx] = data
        channel_mask[idx] = True

    extra = set(layers_by_name) - set(spec.channel_names)
    if extra:
        logger.warning("Лишние слои не из spec, игнорируем: %s", sorted(extra))

    logger.info(
        "Sample собран: C=%d, доступно=%d, shape=%dx%d",
        n_channels,
        int(channel_mask.sum()),
        height,
        width,
    )

    return Sample(
        tensor=tensor,
        channel_names=spec.channel_names,
        channel_mask=channel_mask,
        grid=spec.grid,
    )
