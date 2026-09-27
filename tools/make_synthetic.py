"""Генератор синтетических участков в формате выхода build_layers.

Структура участка <out_dir>/site_<i>/:
    layers/<name>.tif         — dtm, производные рельефа (build_relief_layers роли 1),
                                rgb_r, rgb_g, rgb_b, mag;
    samples/sample.npz        — тензор + channel_mask (_save_sample из build_layers);
    samples/sample_meta.json  — имена каналов, Grid, nodata;
    labels.geojson            — разметка в crs.output (EPSG:3857).

Массивы строятся в конвенции Grid (row 0 = юг), запись — через write_raster.
Детерминировано по (seed, номер участка). Параметры — секция `synthetic` YAML.

Запуск:
    uv run python tools/make_synthetic.py --n-sites 3 --size 1024 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from expds.data.raster import write_raster
from expds.features.config import AppConfig, load_config, setup_logging
from expds.features.grid import Grid, Layer
from expds.features.relief import build_relief_layers

logger = logging.getLogger("make_synthetic")

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _ROOT / "configs" / "dataset_synthetic.yaml"
_BUILD_LAYERS_PATH = _ROOT / "scripts" / "build_layers.py"

_KEY_SYNTHETIC = "synthetic"
_SOURCE = "synthetic"
_LICENSE = "generated"
_CIRCLE_VERTICES = 32
_PLACEMENT_TRIES_PER_OBJECT = 200
_EPS = 1e-12

CLASS_MOUND = "mound"
CLASS_RAMPART = "rampart"
CLASS_DITCH = "ditch"


# ------------------------------------------------------------
# Конфиг генератора
# ------------------------------------------------------------
@dataclass(frozen=True)
class IntRange:
    """Целочисленный диапазон [lo, hi] включительно."""

    lo: int
    hi: int

    def sample(self, rng: np.random.Generator) -> int:
        """Случайное целое из диапазона."""
        return int(rng.integers(self.lo, self.hi + 1))


@dataclass(frozen=True)
class FloatRange:
    """Вещественный диапазон [lo, hi)."""

    lo: float
    hi: float

    def sample(self, rng: np.random.Generator) -> float:
        """Случайное число из диапазона."""
        return float(rng.uniform(self.lo, self.hi))


@dataclass(frozen=True)
class LinearSpec:
    """Параметры линейных объектов (валы, рвы)."""

    count: IntRange
    length_m: FloatRange
    width_m: FloatRange
    amplitude_m: FloatRange


@dataclass(frozen=True)
class SyntheticConfig:
    """Параметры генератора (секция `synthetic`)."""

    out_dir: Path
    n_sites: int
    size_px: int
    seed: int
    origin_x: float
    origin_y: float
    site_spacing_m: float
    edge_margin_m: float
    base_elevation_m: float
    bg_amplitude_m: float
    bg_sigma_px: float
    bg_max_tilt: float
    mound_count: IntRange
    mound_radius_m: FloatRange
    mound_height_m: FloatRange
    mound_min_gap_m: float
    point_max_radius_m: float
    rampart: LinearSpec
    ditch: LinearSpec
    rgb_noise_std: float
    rgb_field_sigma_px: float
    mag_dipole_count: IntRange
    mag_dipole_amplitude_nt: FloatRange
    mag_dipole_sigma_m: float
    mag_dipole_offset_m: float
    mag_stripe_amplitude_nt: float
    mag_stripe_period_m: float
    mag_noise_std_nt: float
    mag_coverage: float


def _irange(value: Any) -> IntRange:
    """[lo, hi] из YAML -> IntRange."""
    lo, hi = value
    return IntRange(int(lo), int(hi))


def _frange(value: Any) -> FloatRange:
    """[lo, hi] из YAML -> FloatRange."""
    lo, hi = value
    return FloatRange(float(lo), float(hi))


def _linear(section: dict[str, Any]) -> LinearSpec:
    """Секция линейных объектов -> LinearSpec."""
    return LinearSpec(
        count=_irange(section["count"]),
        length_m=_frange(section["length_m"]),
        width_m=_frange(section["width_m"]),
        amplitude_m=_frange(section["amplitude_m"]),
    )


def load_synthetic_config(path: Path) -> SyntheticConfig:
    """Читает секцию `synthetic` из YAML.

    Args:
        path: путь к YAML (тот же, что читает load_config роли 1).

    Returns:
        SyntheticConfig.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if _KEY_SYNTHETIC not in raw:
        raise KeyError(f"В конфиге {path} нет секции '{_KEY_SYNTHETIC}'")

    s = raw[_KEY_SYNTHETIC]
    bg, mounds, rgb, mag = s["background"], s["mounds"], s["rgb"], s["mag"]
    return SyntheticConfig(
        out_dir=Path(s["out_dir"]),
        n_sites=int(s["n_sites"]),
        size_px=int(s["size_px"]),
        seed=int(s["seed"]),
        origin_x=float(s["origin_x"]),
        origin_y=float(s["origin_y"]),
        site_spacing_m=float(s["site_spacing_m"]),
        edge_margin_m=float(s["edge_margin_m"]),
        base_elevation_m=float(bg["base_elevation_m"]),
        bg_amplitude_m=float(bg["amplitude_m"]),
        bg_sigma_px=float(bg["sigma_px"]),
        bg_max_tilt=float(bg["max_tilt"]),
        mound_count=_irange(mounds["count"]),
        mound_radius_m=_frange(mounds["radius_m"]),
        mound_height_m=_frange(mounds["height_m"]),
        mound_min_gap_m=float(mounds["min_gap_m"]),
        point_max_radius_m=float(mounds["point_max_radius_m"]),
        rampart=_linear(s["ramparts"]),
        ditch=_linear(s["ditches"]),
        rgb_noise_std=float(rgb["noise_std"]),
        rgb_field_sigma_px=float(rgb["field_sigma_px"]),
        mag_dipole_count=_irange(mag["dipole_count"]),
        mag_dipole_amplitude_nt=_frange(mag["dipole_amplitude_nt"]),
        mag_dipole_sigma_m=float(mag["dipole_sigma_m"]),
        mag_dipole_offset_m=float(mag["dipole_offset_m"]),
        mag_stripe_amplitude_nt=float(mag["stripe_amplitude_nt"]),
        mag_stripe_period_m=float(mag["stripe_period_m"]),
        mag_noise_std_nt=float(mag["noise_std_nt"]),
        mag_coverage=float(mag["coverage"]),
    )


# ------------------------------------------------------------
# Объекты
# ------------------------------------------------------------
@dataclass(frozen=True)
class Mound:
    """Курган: гауссов холм, radius = 2 sigma (граница разметки)."""

    x: float
    y: float
    radius: float
    height: float


@dataclass(frozen=True)
class LinearObject:
    """Вал (amplitude > 0) или ров (amplitude < 0): прямоугольник с косинусным профилем."""

    cls: str
    x0: float
    y0: float
    angle: float
    length: float
    width: float
    amplitude: float

    def corners(self) -> list[tuple[float, float]]:
        """Углы прямоугольника против часовой стрелки."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        hw = self.width / 2.0
        return [
            (self.x0 + t * c - n * s, self.y0 + t * s + n * c)
            for t, n in ((0.0, -hw), (self.length, -hw), (self.length, hw), (0.0, hw))
        ]


def _place_mounds(rng: np.random.Generator, grid: Grid, cfg: SyntheticConfig) -> list[Mound]:
    """Курганы без пересечений (rejection sampling)."""
    x_lo, y_lo, x_hi, y_hi = grid.bounds
    margin = cfg.edge_margin_m
    target = cfg.mound_count.sample(rng)
    mounds: list[Mound] = []
    for _ in range(target * _PLACEMENT_TRIES_PER_OBJECT):
        if len(mounds) == target:
            break
        radius = cfg.mound_radius_m.sample(rng)
        x = float(rng.uniform(x_lo + margin + radius, x_hi - margin - radius))
        y = float(rng.uniform(y_lo + margin + radius, y_hi - margin - radius))
        height = cfg.mound_height_m.sample(rng)
        if all(
            math.hypot(x - m.x, y - m.y) >= radius + m.radius + cfg.mound_min_gap_m for m in mounds
        ):
            mounds.append(Mound(x=x, y=y, radius=radius, height=height))
    if len(mounds) < target:
        logger.warning("Размещено курганов %d из %d", len(mounds), target)
    return mounds


def _place_linear(
    rng: np.random.Generator,
    grid: Grid,
    spec: LinearSpec,
    cls: str,
    sign: float,
    margin: float,
) -> list[LinearObject]:
    """Линейные объекты целиком внутри участка с отступом margin."""
    x_lo, y_lo, x_hi, y_hi = grid.bounds
    x_lo, y_lo, x_hi, y_hi = x_lo + margin, y_lo + margin, x_hi - margin, y_hi - margin
    max_length = 0.8 * min(x_hi - x_lo, y_hi - y_lo)
    target = spec.count.sample(rng)
    objects: list[LinearObject] = []
    for _ in range(target * _PLACEMENT_TRIES_PER_OBJECT):
        if len(objects) == target:
            break
        obj = LinearObject(
            cls=cls,
            x0=float(rng.uniform(x_lo, x_hi)),
            y0=float(rng.uniform(y_lo, y_hi)),
            angle=float(rng.uniform(0.0, 2.0 * math.pi)),
            length=min(spec.length_m.sample(rng), max_length),
            width=spec.width_m.sample(rng),
            amplitude=sign * spec.amplitude_m.sample(rng),
        )
        if all(x_lo <= px <= x_hi and y_lo <= py <= y_hi for px, py in obj.corners()):
            objects.append(obj)
    if len(objects) < target:
        logger.warning("Размещено %s: %d из %d", cls, len(objects), target)
    return objects


# ------------------------------------------------------------
# Рендер слоёв
# ------------------------------------------------------------
def _world_mesh(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Мировые координаты центров пикселей в конвенции Grid (row 0 = юг)."""
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    x, y = grid.transform_to_world(cols, rows)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _mound_profile(x: np.ndarray, y: np.ndarray, mound: Mound) -> np.ndarray:
    """Нормированный профиль кургана, максимум 1."""
    sigma = mound.radius / 2.0
    return np.exp(-((x - mound.x) ** 2 + (y - mound.y) ** 2) / (2.0 * sigma**2))


def _linear_profile(x: np.ndarray, y: np.ndarray, obj: LinearObject) -> np.ndarray:
    """Нормированный косинусный профиль поперёк оси, 0 вне прямоугольника."""
    c, s = math.cos(obj.angle), math.sin(obj.angle)
    dx, dy = x - obj.x0, y - obj.y0
    t = dx * c + dy * s
    n = -dx * s + dy * c
    hw = obj.width / 2.0
    inside = (t >= 0.0) & (t <= obj.length) & (np.abs(n) <= hw)
    return np.where(inside, 0.5 * (1.0 + np.cos(np.pi * n / hw)), 0.0)


def _render_dtm(
    rng: np.random.Generator,
    grid: Grid,
    x: np.ndarray,
    y: np.ndarray,
    cfg: SyntheticConfig,
    mounds: list[Mound],
    linears: list[LinearObject],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DTM + нормированные маски «поднятое» и «опущенное» (для RGB)."""
    u, v = x - grid.x_min, y - grid.y_min
    noise = gaussian_filter(rng.standard_normal(grid.shape), cfg.bg_sigma_px)
    noise *= cfg.bg_amplitude_m / (noise.std() + _EPS)
    tilt_x, tilt_y = rng.uniform(-cfg.bg_max_tilt, cfg.bg_max_tilt, size=2)
    dtm = cfg.base_elevation_m + noise + tilt_x * u + tilt_y * v

    raised = np.zeros(grid.shape)
    sunk = np.zeros(grid.shape)
    for mound in mounds:
        profile = _mound_profile(x, y, mound)
        dtm += mound.height * profile
        raised += profile
    for obj in linears:
        profile = _linear_profile(x, y, obj)
        dtm += obj.amplitude * profile
        if obj.amplitude > 0:
            raised += profile
        else:
            sunk += profile
    return dtm, np.clip(raised, 0.0, 1.0), np.clip(sunk, 0.0, 1.0)


def _render_rgb(
    rng: np.random.Generator,
    grid: Grid,
    cfg: SyntheticConfig,
    raised: np.ndarray,
    sunk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB: поле + почвенные/растительные признаки над объектами + шум. Коэффициенты — косметика."""
    field = gaussian_filter(rng.standard_normal(grid.shape), cfg.rgb_field_sigma_px)
    field /= field.std() + _EPS
    std = cfg.rgb_noise_std
    red = 0.35 + 0.03 * field + 0.10 * raised - 0.04 * sunk + rng.normal(0.0, std, grid.shape)
    green = 0.45 - 0.02 * field - 0.03 * raised + 0.10 * sunk + rng.normal(0.0, std, grid.shape)
    blue = 0.25 + 0.02 * field + rng.normal(0.0, std, grid.shape)
    return (np.clip(red, 0.0, 1.0), np.clip(green, 0.0, 1.0), np.clip(blue, 0.0, 1.0))


def _render_mag(
    rng: np.random.Generator,
    grid: Grid,
    x: np.ndarray,
    y: np.ndarray,
    cfg: SyntheticConfig,
    mounds: list[Mound],
    nodata: float,
) -> np.ndarray:
    """Магнитка: полосы (пахота) + диполи (половина — на курганах) + шум, частичное покрытие."""
    u = x - grid.x_min
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    mag = cfg.mag_stripe_amplitude_nt * np.sin(2.0 * math.pi * u / cfg.mag_stripe_period_m + phase)
    mag = mag + rng.normal(0.0, cfg.mag_noise_std_nt, grid.shape)

    x_lo, y_lo, x_hi, y_hi = grid.bounds
    two_s2 = 2.0 * cfg.mag_dipole_sigma_m**2
    off = cfg.mag_dipole_offset_m
    n_dipoles = cfg.mag_dipole_count.sample(rng)
    for k in range(n_dipoles):
        if k < n_dipoles // 2 and k < len(mounds):
            cx, cy = mounds[k].x, mounds[k].y
        else:
            cx, cy = float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi))
        amp = cfg.mag_dipole_amplitude_nt.sample(rng)
        # Индуцированный диполь (северное полушарие): максимум к югу, минимум к северу.
        mag += amp * np.exp(-((x - cx) ** 2 + (y - (cy - off)) ** 2) / two_s2)
        mag -= 0.5 * amp * np.exp(-((x - cx) ** 2 + (y - (cy + off)) ** 2) / two_s2)

    valid_rows = int(round(cfg.mag_coverage * grid.height))
    mag[valid_rows:, :] = nodata  # северная часть без съёмки (row растёт на север)
    return mag


# ------------------------------------------------------------
# Разметка
# ------------------------------------------------------------
def _circle(x: float, y: float, r: float) -> list[tuple[float, float]]:
    """Окружность многоугольником против часовой стрелки."""
    angles = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_VERTICES, endpoint=False)
    return [(x + r * math.cos(a), y + r * math.sin(a)) for a in angles]


def _ring(points: list[tuple[float, float]], transformer: Transformer) -> list[list[float]]:
    """Точки internal -> замкнутое кольцо в output CRS."""
    xs, ys = transformer.transform([p[0] for p in points], [p[1] for p in points])
    ring = [[float(a), float(b)] for a, b in zip(xs, ys, strict=True)]
    ring.append(ring[0])
    return ring


def _build_labels(
    site_name: str,
    mounds: list[Mound],
    linears: list[LinearObject],
    cfg: SyntheticConfig,
    transformer: Transformer,
    output_crs: str,
) -> dict[str, Any]:
    """FeatureCollection: мелкие курганы — Point, крупные и линейные — Polygon."""
    features: list[dict[str, Any]] = []
    for k, m in enumerate(mounds):
        props = {
            "id": f"{site_name}_{CLASS_MOUND}_{k}",
            "class": CLASS_MOUND,
            "radius_m": m.radius,
            "height_m": m.height,
        }
        if m.radius <= cfg.point_max_radius_m:
            px, py = transformer.transform(m.x, m.y)
            geom = {"type": "Point", "coordinates": [float(px), float(py)]}
        else:
            ring = _ring(_circle(m.x, m.y, m.radius), transformer)
            geom = {"type": "Polygon", "coordinates": [ring]}
        features.append({"type": "Feature", "properties": props, "geometry": geom})

    for k, obj in enumerate(linears):
        props = {
            "id": f"{site_name}_{obj.cls}_{k}",
            "class": obj.cls,
            "length_m": obj.length,
            "width_m": obj.width,
            "amplitude_m": obj.amplitude,
        }
        geom = {"type": "Polygon", "coordinates": [_ring(obj.corners(), transformer)]}
        features.append({"type": "Feature", "properties": props, "geometry": geom})

    epsg = output_crs.split(":")[-1]
    return {
        "type": "FeatureCollection",
        "name": site_name,
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}},
        "features": features,
    }


# ------------------------------------------------------------
# Участок
# ------------------------------------------------------------
def _load_build_layers() -> ModuleType:
    """Импорт scripts/build_layers.py ради _save_sample (один формат sample на всех)."""
    name = "build_layers"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _BUILD_LAYERS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить {_BUILD_LAYERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _layer(name: str, data: np.ndarray, grid: Grid, nodata: float, unit: str) -> Layer:
    """Layer с метаданными синтетики."""
    return Layer(
        name=name,
        data=np.asarray(data, dtype=np.float32),
        grid=grid,
        nodata=nodata,
        source=_SOURCE,
        license=_LICENSE,
        unit=unit,
    )


def generate_site(
    site_idx: int,
    cfg: SyntheticConfig,
    app: AppConfig,
    build_layers_module: ModuleType,
) -> dict[str, int]:
    """Генерирует один участок и пишет его на диск.

    Args:
        site_idx: номер участка с 0 (папка site_<idx+1>).
        cfg: параметры генератора.
        app: конфиг роли 1 (CRS, сетка, рельеф).
        build_layers_module: модуль build_layers (для _save_sample).

    Returns:
        Счётчики объектов по классам и типам геометрии.
    """
    rng = np.random.default_rng([cfg.seed, site_idx])
    nodata = app.grid.nodata
    grid = Grid(
        crs=app.crs.internal,
        pixel_size=app.grid.pixel_size,
        x_min=cfg.origin_x + site_idx * cfg.site_spacing_m,
        y_min=cfg.origin_y,
        width=cfg.size_px,
        height=cfg.size_px,
    )
    x, y = _world_mesh(grid)

    mounds = _place_mounds(rng, grid, cfg)
    ramparts = _place_linear(rng, grid, cfg.rampart, CLASS_RAMPART, 1.0, cfg.edge_margin_m)
    ditches = _place_linear(rng, grid, cfg.ditch, CLASS_DITCH, -1.0, cfg.edge_margin_m)
    linears = ramparts + ditches

    dtm, raised, sunk = _render_dtm(rng, grid, x, y, cfg, mounds, linears)
    red, green, blue = _render_rgb(rng, grid, cfg, raised, sunk)
    mag = _render_mag(rng, grid, x, y, cfg, mounds, nodata)

    dtm32 = dtm.astype(np.float32)
    relief = build_relief_layers(
        dtm=dtm32,
        pixel_size=grid.pixel_size,
        hillshade_azimuths=app.relief.hillshade_azimuths,
        hillshade_altitude=app.relief.hillshade_altitude,
        hillshade_z_factor=app.relief.hillshade_z_factor,
        slrm_sigmas=app.relief.slrm_sigmas,
        openness_radius=app.relief.openness_radius,
        openness_n_directions=app.relief.openness_n_directions,
        curvature_sigmas=app.relief.curvature_sigmas,
        tpi_radii=app.relief.tpi_radii,
        tri_radii=app.relief.tri_radii,
    )

    layers: dict[str, Layer] = {"dtm": _layer("dtm", dtm32, grid, nodata, "m")}
    for name, arr in relief.items():
        layers[name] = _layer(name, arr, grid, nodata, "unitless")
    layers["rgb_r"] = _layer("rgb_r", red, grid, nodata, "reflectance")
    layers["rgb_g"] = _layer("rgb_g", green, grid, nodata, "reflectance")
    layers["rgb_b"] = _layer("rgb_b", blue, grid, nodata, "reflectance")
    layers["mag"] = _layer("mag", mag, grid, nodata, "nT")

    site_name = f"site_{site_idx + 1}"
    site_dir = cfg.out_dir / site_name
    for layer in layers.values():
        write_raster(layer=layer, path=site_dir / "layers" / f"{layer.name}.tif")
    build_layers_module._save_sample(
        layers=layers,
        grid=grid,
        output_dir=site_dir / "samples",
        nodata=nodata,
    )

    transformer = Transformer.from_crs(app.crs.internal, app.crs.output, always_xy=True)
    labels = _build_labels(site_name, mounds, linears, cfg, transformer, app.crs.output)
    (site_dir / "labels.geojson").write_text(
        json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    n_points = sum(1 for f in labels["features"] if f["geometry"]["type"] == "Point")
    return {
        CLASS_MOUND: len(mounds),
        CLASS_RAMPART: len(ramparts),
        CLASS_DITCH: len(ditches),
        "points": n_points,
        "polygons": len(labels["features"]) - n_points,
        "layers": len(layers),
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы CLI; заданные переопределяют YAML."""
    parser = argparse.ArgumentParser(description="Синтетические участки в формате build_layers")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--n-sites", type=int, default=None)
    parser.add_argument("--size", type=int, default=None, help="Размер участка, пиксели")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа."""
    args = _parse_args(argv)
    app = load_config(args.config)
    setup_logging(app.logging)
    cfg = load_synthetic_config(args.config)

    overrides = {
        key: value
        for key, value in (
            ("n_sites", args.n_sites),
            ("size_px", args.size),
            ("seed", args.seed),
            ("out_dir", args.out_dir),
        )
        if value is not None
    }
    cfg = replace(cfg, **overrides)
    if cfg.size_px * app.grid.pixel_size <= 4 * cfg.edge_margin_m:
        raise ValueError(f"Участок {cfg.size_px} px слишком мал для отступа {cfg.edge_margin_m} м")

    build_layers_module = _load_build_layers()
    t0 = time.perf_counter()
    for site_idx in range(cfg.n_sites):
        counts = generate_site(site_idx, cfg, app, build_layers_module)
        logger.info("site_%d: %s", site_idx + 1, counts)
    logger.info(
        "Готово: %d участков %dx%d за %.1f с -> %s",
        cfg.n_sites,
        cfg.size_px,
        cfg.size_px,
        time.perf_counter() - t0,
        cfg.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
