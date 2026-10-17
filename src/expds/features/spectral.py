"""Спектральные индексы из мультиканальных снимков (NDVI, SR, ARI, ...).

Зачем: NDVI/SR/ARI — классические индексы для выявления угнетённой
растительности над скрытыми археологическими объектами (см. статью
о применении МО в археологии). Они дают модели дополнительный сигнал,
которого нет в grayscale и в лидаре.

Модуль работает на массивах каналов (уже прочитанных и приведённых
к общей сетке), а не на файлах. Чтение GeoTIFF и ресэмплинг — снаружи
(data/raster.py + features/align.py). Это позволяет тестировать
индексы без I/O.

Индексы каналов — 0-based. Если нужного канала нет (nir_channel=None),
индекс не строится: возвращается пустой словарь, слой не попадает
в Sample, channel_mask=False. Это graceful skip для grayscale-снимков,
где NIR физически отсутствует.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from expds.features.config import SpectralConfig
from expds.features.grid import Grid, Layer

logger = logging.getLogger(__name__)

# Константы: имена индексов.
INDEX_NDVI = "ndvi"
INDEX_SR = "sr"
INDEX_ARI = "ari"
INDEX_GNDVI = "gndvi"
INDEX_NDWI = "ndwi"

SUPPORTED_INDICES = (
    INDEX_NDVI,
    INDEX_SR,
    INDEX_ARI,
    INDEX_GNDVI,
    INDEX_NDWI,
)

# Константы: защита от деления на ноль.
_EPSILON = 1e-6

# Константы: диапазоны нормированных разностей.
_NDVI_MIN = -1.0
_NDVI_MAX = 1.0


@dataclass(frozen=True)
class SpectralResult:
    """Результат построения спектральных индексов.

    layers — словарь имя -> Layer, только для построенных индексов.
    skipped — имена индексов, которые не построены (нет каналов).
    """

    layers: dict[str, Layer]
    skipped: tuple[str, ...]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Делит массивы с защитой от нуля.

    Там, где |denominator| < epsilon, результат = 0.

    Args:
        numerator: числитель.
        denominator: знаменатель.

    Returns:
        np.ndarray float64.
    """
    safe_denom = np.where(np.abs(denominator) < _EPSILON, 1.0, denominator)
    return np.where(np.abs(denominator) < _EPSILON, 0.0, numerator / safe_denom)


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Нормированная разность (a - b) / (a + b).

    Args:
        a: первый канал.
        b: второй канал.

    Returns:
        np.ndarray float64 в диапазоне [-1, 1] (там, где a+b != 0).
    """
    return _safe_ratio(a - b, a + b)


def _clip_normalized(values: np.ndarray) -> np.ndarray:
    """Обрезает значения нормированной разности до [-1, 1].

    Численные погрешности могут дать -1.0000001. Для модели это шум.
    """
    return np.clip(values, _NDVI_MIN, _NDVI_MAX)


def _apply_valid_mask(
    data: np.ndarray,
    valid: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """Ставит nodata там, где входные каналы были невалидны.

    Args:
        data: массив индекса.
        valid: булева маска валидных пикселей (True = валидно).
        nodata: значение nodata.

    Returns:
        np.ndarray float32 с nodata в невалидных пикселях.
    """
    out = data.astype(np.float32, copy=True)
    out[~valid] = nodata
    return out


def compute_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    valid: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red).

    Классический индекс здоровья растительности. На участках с угнетённой
    растительностью над скрытыми объектами NDVI аномально низкий.

    Args:
        red: канал красного.
        nir: канал ближнего инфракрасного.
        valid: маска валидных пикселей.
        nodata: значение nodata.

    Returns:
        2D float32 массив NDVI в [-1, 1], nodata в невалидных пикселях.
    """
    ndvi = _clip_normalized(_normalized_difference(nir.astype(np.float64), red.astype(np.float64)))
    return _apply_valid_mask(ndvi, valid, nodata)


def compute_sr(
    red: np.ndarray,
    nir: np.ndarray,
    valid: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """SR = NIR / Red. Простое отношение, чувствительное к биомассе.

    Args:
        red: канал красного.
        nir: канал ближнего инфракрасного.
        valid: маска валидных пикселей.
        nodata: значение nodata.

    Returns:
        2D float32 массив SR (>= 0), nodata в невалидных пикселях.
    """
    sr = _safe_ratio(nir.astype(np.float64), red.astype(np.float64))
    # SR не нормирован, но отрицательным быть не может (физически).
    sr = np.maximum(sr, 0.0)
    return _apply_valid_mask(sr, valid, nodata)


def compute_ari(
    green: np.ndarray,
    red: np.ndarray,
    valid: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """ARI = (1 / Green) - (1 / Red). Anthocyanin Reflectance Index.

    Антоцианы указывают на стресс растений. Для археологии — косвенный
    признак нарушенного грунта/скрытых объектов.

    Args:
        green: канал зелёного.
        red: канал красного.
        valid: маска валидных пикселей.
        nodata: значение nodata.

    Returns:
        2D float32 массив ARI, nodata в невалидных пикселях.
    """
    inv_green = _safe_ratio(np.ones_like(green, dtype=np.float64), green.astype(np.float64))
    inv_red = _safe_ratio(np.ones_like(red, dtype=np.float64), red.astype(np.float64))
    ari = inv_green - inv_red
    return _apply_valid_mask(ari, valid, nodata)


def compute_gndvi(
    green: np.ndarray,
    nir: np.ndarray,
    valid: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """GNDVI = (NIR - Green) / (NIR + Green). Аналог NDVI для зелёного.

    Args:
        green: канал зелёного.
        nir: канал ближнего инфракрасного.
        valid: маска валидных пикселей.
        nodata: значение nodata.

    Returns:
        2D float32 массив GNDVI в [-1, 1], nodata в невалидных пикселях.
    """
    gndvi = _clip_normalized(
        _normalized_difference(nir.astype(np.float64), green.astype(np.float64))
    )
    return _apply_valid_mask(gndvi, valid, nodata)


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    valid: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR). Water Index (McFeeters).

    Помогает отделить воду/влаголюбивую растительность (рвы, заболоченные
    участки) от суши.

    Args:
        green: канал зелёного.
        nir: канал ближнего инфракрасного.
        valid: маска валидных пикселей.
        nodata: значение nodata.

    Returns:
        2D float32 массив NDWI в [-1, 1], nodata в невалидных пикселях.
    """
    ndwi = _clip_normalized(
        _normalized_difference(green.astype(np.float64), nir.astype(np.float64))
    )
    return _apply_valid_mask(ndwi, valid, nodata)


def _get_channel(
    channels: dict[int, np.ndarray],
    index: int | None,
    name: str,
) -> np.ndarray | None:
    """Достаёт канал по индексу, логирует отсутствие.

    Args:
        channels: словарь index -> массив.
        index: 0-based индекс канала или None.
        name: имя канала для лога.

    Returns:
        Массив канала или None.
    """
    if index is None:
        return None
    if index not in channels:
        logger.warning("Канал '%s' (index=%d) отсутствует в снимке", name, index)
        return None
    return channels[index]


def build_spectral_indices(
    channels: dict[int, np.ndarray],
    valid_mask: np.ndarray,
    grid: Grid,
    config: SpectralConfig,
    source: str = "unknown",
) -> SpectralResult:
    """Строит запрошенные спектральные индексы.

    Args:
        channels: словарь index -> 2D массив канала (0-based).
        valid_mask: булева маска валидных пикселей (общая для всех каналов).
            Обычно это маска nodata/alpha из исходного снимка.
        grid: общая сетка (все каналы уже приведены к ней).
        config: параметры спектральных индексов.
        source: описание источника для метаданных Layer.

    Returns:
        SpectralResult с построенными слоями и списком пропущенных индексов.

    Raises:
        ValueError: если valid_mask не совпадает по форме с каналами.
    """
    if valid_mask.shape != grid.shape:
        raise ValueError(f"valid_mask shape {valid_mask.shape} не совпадает с grid {grid.shape}")
    for idx, ch in channels.items():
        if ch.shape != grid.shape:
            raise ValueError(f"Канал {idx} shape {ch.shape} не совпадает с grid {grid.shape}")

    enabled = set(config.enabled)
    unknown = enabled - set(SUPPORTED_INDICES)
    if unknown:
        logger.warning("Неизвестные индексы в конфиге, игнорируем: %s", sorted(unknown))
        enabled &= set(SUPPORTED_INDICES)

    if not enabled:
        logger.info("Спектральные индексы отключены или не запрошены")
        return SpectralResult(layers={}, skipped=())

    red = _get_channel(channels, config.red_channel, "red")
    nir = _get_channel(channels, config.nir_channel, "nir")
    green = _get_channel(channels, config.green_channel, "green")
    # blue пока не используется в индексах, но оставлен в конфиге на будущее.

    layers: dict[str, Layer] = {}
    skipped: list[str] = []

    def _need(*arrays: np.ndarray | None) -> bool:
        return all(a is not None for a in arrays)

    if INDEX_NDVI in enabled:
        if _need(red, nir):
            data = compute_ndvi(red=red, nir=nir, valid=valid_mask, nodata=config.nodata)  # type: ignore[arg-type]
            layers[INDEX_NDVI] = Layer(
                name=INDEX_NDVI,
                data=data,
                grid=grid,
                nodata=config.nodata,
                source=source,
                license="unknown",
                unit="index",
            )
            logger.info("Построен индекс: %s", INDEX_NDVI)
        else:
            skipped.append(INDEX_NDVI)

    if INDEX_SR in enabled:
        if _need(red, nir):
            data = compute_sr(red=red, nir=nir, valid=valid_mask, nodata=config.nodata)  # type: ignore[arg-type]
            layers[INDEX_SR] = Layer(
                name=INDEX_SR,
                data=data,
                grid=grid,
                nodata=config.nodata,
                source=source,
                license="unknown",
                unit="ratio",
            )
            logger.info("Построен индекс: %s", INDEX_SR)
        else:
            skipped.append(INDEX_SR)

    if INDEX_ARI in enabled:
        if _need(green, red):
            data = compute_ari(green=green, red=red, valid=valid_mask, nodata=config.nodata)  # type: ignore[arg-type]
            layers[INDEX_ARI] = Layer(
                name=INDEX_ARI,
                data=data,
                grid=grid,
                nodata=config.nodata,
                source=source,
                license="unknown",
                unit="index",
            )
            logger.info("Построен индекс: %s", INDEX_ARI)
        else:
            skipped.append(INDEX_ARI)

    if INDEX_GNDVI in enabled:
        if _need(green, nir):
            data = compute_gndvi(green=green, nir=nir, valid=valid_mask, nodata=config.nodata)  # type: ignore[arg-type]
            layers[INDEX_GNDVI] = Layer(
                name=INDEX_GNDVI,
                data=data,
                grid=grid,
                nodata=config.nodata,
                source=source,
                license="unknown",
                unit="index",
            )
            logger.info("Построен индекс: %s", INDEX_GNDVI)
        else:
            skipped.append(INDEX_GNDVI)

    if INDEX_NDWI in enabled:
        if _need(green, nir):
            data = compute_ndwi(green=green, nir=nir, valid=valid_mask, nodata=config.nodata)  # type: ignore[arg-type]
            layers[INDEX_NDWI] = Layer(
                name=INDEX_NDWI,
                data=data,
                grid=grid,
                nodata=config.nodata,
                source=source,
                license="unknown",
                unit="index",
            )
            logger.info("Построен индекс: %s", INDEX_NDWI)
        else:
            skipped.append(INDEX_NDWI)

    if skipped:
        logger.info("Пропущены индексы (нет каналов): %s", skipped)

    return SpectralResult(layers=layers, skipped=tuple(skipped))
