"""Спектральные индексы из мультиканальных снимков (NDVI, SR, ARI, ...).

Зачем: NDVI/SR/ARI/SAVI — классические индексы для выявления угнетённой
растительности и нарушенного грунта над скрытыми археологическими
объектами (см. статью о применении МО в археологии). Они дают модели
дополнительный сигнал, которого нет в grayscale и в лидаре.

Модуль работает на массивах каналов (уже прочитанных и приведённых
к общей сетке), а не на файлах. Чтение GeoTIFF и ресэмплинг — снаружи
(data/raster.py + features/align.py). Это позволяет тестировать
индексы без I/O.

Индексы каналов — 0-based. Если нужного канала нет (nir_channel=None),
индекс не строится: возвращается пустой словарь, слой не попадает
в Sample, channel_mask=False. Это graceful skip для grayscale-снимков,
где NIR физически отсутствует.

Дополнительно: каналы с std по валидным пикселям ниже channel_std_min
считаются константными (alpha-маска, nodata-заливка) и в индексы не идут.
Иначе NDVI/ARI на константном канале дают мусор, который модель примет
за сигнал.

Используемые каналы:
- NDVI, SR, GNDVI, NDWI — Red и/или NIR и/или Green.
- ARI — Green и Red.
- SAVI — Red и NIR (Soil-Adjusted Vegetation Index, устойчив к почвенному
  фону — важно для распаханных курганов).
- NDBI (Normalized Difference Built-up Index) — требует SWIR, которого
  в наших данных нет, поэтому НЕ реализован. blue_channel в конфиге
  зарезервирован на будущее.
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
INDEX_SAVI = "savi"

SUPPORTED_INDICES = (
    INDEX_NDVI,
    INDEX_SR,
    INDEX_ARI,
    INDEX_GNDVI,
    INDEX_NDWI,
    INDEX_SAVI,
)

# Константы: защита от деления на ноль.
_EPSILON = 1e-6

# Константы: диапазоны нормированных разностей.
_NDVI_MIN = -1.0
_NDVI_MAX = 1.0

# Константы: верхний предел SR (физически NIR/Red редко > 20).
_SR_MAX = 20.0

# Константы: верхний предел |ARI| (1/green - 1/red, при uint8-яркостях
# разумный предел — единицы; выбросы обрезаем).
_ARI_ABS_MAX = 10.0

# Константы: SAVI. L — параметр адаптации к плотности растительности.
# L=0.5 — рекомендованный стандарт (Huete, 1988).
_SAVI_L = 0.5
_SAVI_MIN = -1.5
_SAVI_MAX = 1.5


@dataclass(frozen=True)
class SpectralResult:
    """Результат построения спектральных индексов.

    layers — словарь имя -> Layer, только для построенных индексов.
    skipped — имена индексов, которые не построены (нет каналов
        или канал константный).
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

    SR ограничен сверху _SR_MAX: при Red→0 отношение взрывается, а для
    модели это выброс. Клиппинг в [0, _SR_MAX].

    Args:
        red: канал красного.
        nir: канал ближнего инфракрасного.
        valid: маска валидных пикселей.
        nodata: значение nodata.

    Returns:
        2D float32 массив SR в [0, _SR_MAX], nodata в невалидных пикселях.
    """
    sr = _safe_ratio(nir.astype(np.float64), red.astype(np.float64))
    sr = np.clip(sr, 0.0, _SR_MAX)
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

    ARI не нормирован: при яркостях, близких к 0, он взрывается.
    Клиппинг в [-_ARI_ABS_MAX, +_ARI_ABS_MAX].

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
    ari = np.clip(ari, -_ARI_ABS_MAX, _ARI_ABS_MAX)
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


def compute_savi(
    red: np.ndarray,
    nir: np.ndarray,
    valid: np.ndarray,
    nodata: float,
    l_factor: float = _SAVI_L,
) -> np.ndarray:
    """SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L).

    Soil-Adjusted Vegetation Index (Huete, 1988). Устойчив к почвенному
    фону — важно для распаханных курганов и открытого грунта, где NDVI
    занижен из-за яркой почвы.

    Args:
        red: канал красного.
        nir: канал ближнего инфракрасного.
        valid: маска валидных пикселей.
        nodata: значение nodata.
        l_factor: параметр L (0.5 — стандарт Huete).

    Returns:
        2D float32 массив SAVI, nodata в невалидных пикселях.
    """
    nir_f = nir.astype(np.float64)
    red_f = red.astype(np.float64)
    numer = (nir_f - red_f) * (1.0 + l_factor)
    denom = nir_f + red_f + l_factor
    savi = _safe_ratio(numer, denom)
    savi = np.clip(savi, _SAVI_MIN, _SAVI_MAX)
    return _apply_valid_mask(savi, valid, nodata)


def _is_informative(
    channel: np.ndarray | None,
    valid_mask: np.ndarray,
    std_min: float,
) -> bool:
    """Проверяет, что канал не None и не константа по валидным пикселям.

    Константные каналы — это alpha-маски (все 255), nodata-заливки
    (все 0/-9999). Считать по ним NDVI/ARI — мусор.

    Args:
        channel: 2D массив канала или None.
        valid_mask: булева маска валидных пикселей.
        std_min: порог std по валидным пикселям.

    Returns:
        True, если канал информативен.
    """
    if channel is None:
        return False
    valid = valid_mask & np.isfinite(channel)
    if not valid.any():
        return False
    std = float(np.std(channel[valid].astype(np.float64)))
    return std >= std_min


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
    # blue_channel в текущих индексах не используется: NDBI требует SWIR,
    # которого в наших данных нет. Поле оставлено на будущее.

    std_min = config.channel_std_min
    red_ok = _is_informative(red, valid_mask, std_min)
    nir_ok = _is_informative(nir, valid_mask, std_min)
    green_ok = _is_informative(green, valid_mask, std_min)

    if not red_ok:
        logger.info("Канал red неинформативен (None или константа), red-индексы пропущены")
    if not nir_ok:
        logger.info("Канал nir неинформативен (None или константа), NIR-индексы пропущены")
    if not green_ok:
        logger.info("Канал green неинформативен (None или константа), green-индексы пропущены")

    layers: dict[str, Layer] = {}
    skipped: list[str] = []

    def _need(*flags: bool) -> bool:
        return all(flags)

    if INDEX_NDVI in enabled:
        if _need(red_ok, nir_ok):
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
        if _need(red_ok, nir_ok):
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
        if _need(green_ok, red_ok):
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
        if _need(green_ok, nir_ok):
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
        if _need(green_ok, nir_ok):
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

    if INDEX_SAVI in enabled:
        if _need(red_ok, nir_ok):
            data = compute_savi(red=red, nir=nir, valid=valid_mask, nodata=config.nodata)  # type: ignore[arg-type]
            layers[INDEX_SAVI] = Layer(
                name=INDEX_SAVI,
                data=data,
                grid=grid,
                nodata=config.nodata,
                source=source,
                license="unknown",
                unit="index",
            )
            logger.info("Построен индекс: %s", INDEX_SAVI)
        else:
            skipped.append(INDEX_SAVI)

    if skipped:
        logger.info("Пропущены индексы (нет каналов или канал константный): %s", skipped)

    return SpectralResult(layers=layers, skipped=tuple(skipped))
