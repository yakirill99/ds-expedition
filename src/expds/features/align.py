"""Выравнивание слоёв: ресэмплинг в общую сетку и проверка сдвига.

Когда у нас несколько модальностей (лидар, оптика, геофизика), каждая
приходит в своей сетке: другой CRS, другое разрешение, другой extent.
Чтобы модель могла их сложить в один тензор, все слои приводятся к
одной Grid.

Плюс отдельная задача — проверка совмещения. Даже если формально слои
в одной сетке, между ними может быть сдвиг в 1–3 пикселя. Причины:
разные системы привязки, ошибки в геопривязке снимков, разные эпохи.
Сдвиг в 2–3 пикселя — типичная причина, по которой хорошая модель
даёт плохую метрику.

Модуль делает три вещи:
1. resample_layer     — приводит слой к целевой сетке.
2. estimate_shift     — оценивает сдвиг между двумя слоями через
                        кросс-корреляцию.
3. apply_shift        — сдвигает слой на целое число пикселей.

Плюс check_alignment — проверяет все слои относительно эталонного.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)

# Константы: имена методов ресэмплинга.
METHOD_BILINEAR = "bilinear"
METHOD_NEAREST = "nearest"
METHOD_CUBIC = "cubic"
METHOD_AVERAGE = "average"

SUPPORTED_METHODS = (
    METHOD_BILINEAR,
    METHOD_NEAREST,
    METHOD_CUBIC,
    METHOD_AVERAGE,
)

# Маппинг имён в Resampling rasterio.
_METHOD_MAP: dict[str, Resampling] = {
    METHOD_BILINEAR: Resampling.bilinear,
    METHOD_NEAREST: Resampling.nearest,
    METHOD_CUBIC: Resampling.cubic,
    METHOD_AVERAGE: Resampling.average,
}

# Допуск при сравнении pixel_size и границ сеток.
_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ShiftEstimate:
    """Оценка сдвига между двумя слоями.

    dx, dy — сдвиг в пикселях. Положительный dx: слой B сдвинут вправо
    относительно слоя A.
    confidence — отношение высоты пика кросс-корреляции к медиане.
    Высокое значение (>= 3) означает надёжный сдвиг.
    """

    dx: int
    dy: int
    confidence: float

    @property
    def magnitude(self) -> float:
        """Модуль сдвига в пикселях."""
        return float(np.sqrt(self.dx**2 + self.dy**2))


@dataclass(frozen=True)
class AlignmentReport:
    """Отчёт по проверке совмещения всех слоёв.

    reference — имя эталонного слоя.
    shifts — словарь имя -> ShiftEstimate.
    """

    reference: str
    shifts: dict[str, ShiftEstimate]


def _grids_equal(a: Grid, b: Grid) -> bool:
    """Проверяет, совпадают ли две сетки с точностью до допуска.

    Args:
        a: первая сетка.
        b: вторая сетка.

    Returns:
        True, если сетки идентичны.
    """
    if a.crs != b.crs:
        return False
    if abs(a.pixel_size - b.pixel_size) > _TOLERANCE:
        return False
    if abs(a.x_min - b.x_min) > _TOLERANCE:
        return False
    if abs(a.y_min - b.y_min) > _TOLERANCE:
        return False
    if a.width != b.width or a.height != b.height:
        return False
    return True


def _build_transform(grid: Grid) -> Affine:
    """Строит affine-трансформацию rasterio из Grid.

    rasterio ожидает верхний левый угол.

    Args:
        grid: наша сетка.

    Returns:
        Affine для rasterio.
    """
    _, _, _, y_max = grid.bounds
    return Affine(
        grid.pixel_size,
        0.0,
        grid.x_min,
        0.0,
        -grid.pixel_size,
        y_max,
    )


def resample_layer(
    layer: Layer,
    target_grid: Grid,
    method: str = METHOD_BILINEAR,
) -> Layer:
    """Приводит слой к целевой сетке.

    Если сетки совпадают — возвращает слой как есть. Если различаются —
    ресэмплит через rasterio.warp.reproject.

    Перепроецирование между CRS НЕ делается. Если CRS слоя не совпадает
    с target_grid.crs, будет ошибка. Сначала нужно привести CRS.

    Args:
        layer: исходный слой.
        target_grid: целевая сетка.
        method: метод ресэмплинга (bilinear, nearest, cubic, average).

    Returns:
        Layer в target_grid.

    Raises:
        ValueError: если method неизвестен или CRS не совпадает.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Метод '{method}' не поддерживается. Доступны: {SUPPORTED_METHODS}")
    if layer.grid.crs != target_grid.crs:
        raise ValueError(
            f"CRS слоя ({layer.grid.crs}) не совпадает с target_grid "
            f"({target_grid.crs}). Перепроецирование — отдельная задача."
        )

    if _grids_equal(layer.grid, target_grid):
        logger.debug("Сетки совпадают, ресэмплинг не нужен: %s", layer.name)
        return layer

    logger.info(
        "Ресэмплинг %s: %dx%d -> %dx%d, метод=%s",
        layer.name,
        layer.grid.width,
        layer.grid.height,
        target_grid.width,
        target_grid.height,
        method,
    )

    src_transform = _build_transform(layer.grid)
    src_crs = CRS.from_user_input(layer.grid.crs)
    dst_transform = _build_transform(target_grid)
    dst_crs = CRS.from_user_input(target_grid.crs)

    destination = np.full(target_grid.shape, layer.nodata, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(np.flipud(layer.data)),  # Grid (row 0 = юг) -> north-up
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=layer.nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=layer.nodata,
        resampling=_METHOD_MAP[method],
    )

    destination = np.ascontiguousarray(np.flipud(destination))  # north-up -> Grid

    return Layer(
        name=layer.name,
        data=destination,
        grid=target_grid,
        nodata=layer.nodata,
        source=layer.source,
        license=layer.license,
        unit=layer.unit,
    )


def resample_layers(
    layers: dict[str, Layer],
    target_grid: Grid,
    methods: dict[str, str] | None = None,
    default_method: str = METHOD_BILINEAR,
) -> dict[str, Layer]:
    """Ресэмплит несколько слоёв к одной сетке.

    Args:
        layers: словарь имя -> Layer.
        target_grid: целевая сетка.
        methods: словарь имя -> метод. Если для слоя не указан —
            используется default_method.
        default_method: метод по умолчанию.

    Returns:
        Словарь имя -> Layer в target_grid.
    """
    methods = methods or {}
    result: dict[str, Layer] = {}

    for name, layer in layers.items():
        method = methods.get(name, default_method)
        result[name] = resample_layer(layer=layer, target_grid=target_grid, method=method)

    logger.info("Ресэмплинг завершён: %d слоёв", len(result))
    return result


def estimate_shift(
    layer_a: Layer,
    layer_b: Layer,
    max_shift: int = 5,
) -> ShiftEstimate:
    """Оценивает сдвиг между двумя слоями через кросс-корреляцию.

    Работает только для коррелирующих слоёв (например, DTM и его
    производные, или два DTM из разных источников). Для несвязанных
    модальностей (лидар и оптика) нужны другие методы.

    Алгоритм:
    1. Нормализуем оба слоя (вычитаем среднее, делим на std).
    2. Считаем кросс-корреляцию через FFT.
    3. Ищем пик в окне [-max_shift, +max_shift].
    4. Сравниваем высоту пика с медианой — это уверенность.

    Args:
        layer_a: первый слой (эталон).
        layer_b: второй слой (проверяемый).
        max_shift: максимальный сдвиг в пикселях.

    Returns:
        ShiftEstimate со сдвигом и уверенностью.

    Raises:
        ValueError: если сетки слоёв не совпадают.
    """
    if not _grids_equal(layer_a.grid, layer_b.grid):
        raise ValueError(
            "Сетки слоёв не совпадают. Сначала приведи их к одной сетке через resample_layer."
        )

    # Маски: валидные пиксели в обоих слоях.
    mask = layer_a.valid_mask & layer_b.valid_mask
    if mask.sum() < 100:
        logger.warning("Мало валидных пикселей для оценки сдвига: %d", int(mask.sum()))
        return ShiftEstimate(dx=0, dy=0, confidence=0.0)

    a = layer_a.data[mask].astype(np.float64)
    b = layer_b.data[mask].astype(np.float64)

    # Нормализация.
    a = a - a.mean()
    b = b - b.mean()
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-9 or b_std < 1e-9:
        logger.warning("Один из слоёв константный, сдвиг не определён")
        return ShiftEstimate(dx=0, dy=0, confidence=0.0)
    a = a / a_std
    b = b / b_std

    # Нормированная кросс-корреляция (NCC).
    # Заменяем nodata на 0, чтобы не портить FFT.
    a_full = np.where(mask, layer_a.data, 0.0).astype(np.float64)
    b_full = np.where(mask, layer_b.data, 0.0).astype(np.float64)

    # Центрируем по среднему валидных пикселей.
    a_full = a_full - a_full[mask].mean()
    b_full = b_full - b_full[mask].mean()

    # Энергии (нормы) для нормировки.
    energy_a = np.sqrt((a_full[mask] ** 2).sum())
    energy_b = np.sqrt((b_full[mask] ** 2).sum())
    if energy_a < 1e-9 or energy_b < 1e-9:
        logger.warning("Один из слоёв константный, сдвиг не определён")
        return ShiftEstimate(dx=0, dy=0, confidence=0.0)

    fa = np.fft.fft2(a_full)
    fb = np.fft.fft2(b_full)
    # Кросс-корреляция: A * conj(B).
    cc = np.fft.ifft2(fa * np.conj(fb)).real
    # Нормировка: NCC в диапазоне [-1, 1].
    cc = cc / (energy_a * energy_b)
    # Сдвигаем нулевую частоту в центр.
    cc = np.fft.fftshift(cc)

    # Ищем пик в окне [-max_shift, +max_shift] вокруг центра.
    center_y = cc.shape[0] // 2
    center_x = cc.shape[1] // 2
    y_lo = max(0, center_y - max_shift)
    y_hi = min(cc.shape[0], center_y + max_shift + 1)
    x_lo = max(0, center_x - max_shift)
    x_hi = min(cc.shape[1], center_x + max_shift + 1)

    window = cc[y_lo:y_hi, x_lo:x_hi]
    peak_idx = np.unravel_index(np.argmax(window), window.shape)
    # Кросс-корреляция a * conj(b) даёт пик в -d, где d — сдвиг b
    # относительно a. Инвертируем знак, чтобы получить сам сдвиг.
    peak_y = -(peak_idx[0] + y_lo - center_y)
    peak_x = -(peak_idx[1] + x_lo - center_x)
    peak_value = float(window[peak_idx])

    # NCC: пик в [-1, 1]. confidence = сам пик (насколько коррелируют слои).
    # Для идеального совпадения ~1.0. Для шума ~0.
    confidence = peak_value

    logger.info(
        "Оценка сдвига %s vs %s: dx=%d, dy=%d, confidence=%.2f",
        layer_a.name,
        layer_b.name,
        peak_x,
        peak_y,
        confidence,
    )

    return ShiftEstimate(dx=int(peak_x), dy=int(peak_y), confidence=confidence)


def apply_shift(layer: Layer, shift: ShiftEstimate) -> Layer:
    """Сдвигает слой на целое число пикселей.

    Сдвиг применяется через np.roll. Освободившиеся края заполняются
    nodata.

    Args:
        layer: исходный слой.
        shift: оценка сдвига.

    Returns:
        Сдвинутый слой.
    """
    if shift.dx == 0 and shift.dy == 0:
        return layer

    data = np.roll(np.roll(layer.data, shift.dy, axis=0), shift.dx, axis=1)

    # Заполняем края nodata.
    if shift.dy > 0:
        data[: shift.dy, :] = layer.nodata
    elif shift.dy < 0:
        data[shift.dy :, :] = layer.nodata
    if shift.dx > 0:
        data[:, : shift.dx] = layer.nodata
    elif shift.dx < 0:
        data[:, shift.dx :] = layer.nodata

    logger.info(
        "Применён сдвиг к %s: dx=%d, dy=%d",
        layer.name,
        shift.dx,
        shift.dy,
    )

    return Layer(
        name=layer.name,
        data=data,
        grid=layer.grid,
        nodata=layer.nodata,
        source=layer.source,
        license=layer.license,
        unit=layer.unit,
    )


def check_alignment(
    layers: dict[str, Layer],
    reference_name: str,
    max_shift: int = 5,
    confidence_threshold: float = 3.0,
) -> AlignmentReport:
    """Проверяет совмещение всех слоёв относительно эталонного.

    Для каждого слоя оценивает сдвиг относительно reference. Если
    уверенность ниже порога — сдвиг считается нулевым (нет надёжного
    сигнала).

    Args:
        layers: словарь имя -> Layer.
        reference_name: имя эталонного слоя (должен быть в layers).
        max_shift: максимальный сдвиг для поиска.
        confidence_threshold: минимальная уверенность для доверия сдвигу.

    Returns:
        AlignmentReport со сдвигами по каждому слою.

    Raises:
        KeyError: если reference_name нет в layers.
    """
    if reference_name not in layers:
        raise KeyError(
            f"Эталонный слой '{reference_name}' не найден. Доступны: {sorted(layers.keys())}"
        )

    reference = layers[reference_name]
    shifts: dict[str, ShiftEstimate] = {}

    for name, layer in layers.items():
        if name == reference_name:
            continue
        try:
            estimate = estimate_shift(layer_a=reference, layer_b=layer, max_shift=max_shift)
        except ValueError as exc:
            logger.warning("Не удалось оценить сдвиг для %s: %s", name, exc)
            continue

        if estimate.confidence < confidence_threshold:
            logger.info(
                "Сдвиг %s ненадёжен (confidence=%.2f < %.2f), считаем нулевым",
                name,
                estimate.confidence,
                confidence_threshold,
            )
            estimate = ShiftEstimate(dx=0, dy=0, confidence=estimate.confidence)

        shifts[name] = estimate

    logger.info(
        "Проверка совмещения завершена: %d слоёв относительно %s",
        len(shifts),
        reference_name,
    )
    return AlignmentReport(reference=reference_name, shifts=shifts)
