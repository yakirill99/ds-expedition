"""Тесты build_site: сырой лидар -> layers/ + samples/, resolve_sites."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from expds.data.build_site import (
    RawSiteUnsupportedError,
    build_site,
    find_lidar_files,
    resolve_sites,
    site_kind,
)
from expds.features.config import load_config
from expds.utils.profiler import Profiler

_CRS = "EPSG:32637"

_TEST_YAML = """
dataset:
  name: test
  raw_dir: data/raw
  cache_dir: data/cache
  lidar_file: site.las
lidar:
  source_z_unit: meters
  target_z_unit: meters
  chunk_size: 100000
  max_points: null
crs:
  source: EPSG:32637
  internal: EPSG:32637
  output: EPSG:3857
grid:
  pixel_size: 0.5
  nodata: -9999.0
ground:
  cell_size: 0.5
  preset: soft
  window_sizes: [3, 5]
  height_threshold: 0.5
  slope_threshold: 0.2
  max_iterations: 5
  min_change_ratio: 0.001
dtm:
  statistic: mean
  fill_method: idw
  idw_k: 8
  idw_power: 2.0
  max_hole_pixels: 100
  artifact_window: 5
  artifact_threshold: 2.0
relief:
  hillshade_azimuths: [45, 135]
  hillshade_altitude: 45.0
  hillshade_z_factor: 1.0
  slrm_sigmas: [8.0]
  openness_radius: 5
  openness_n_directions: 8
  curvature_sigmas: [2.0]
  tpi_radii: [5]
  tri_radii: [3]
cloud_features:
  enabled: [point_density, mean_intensity, std_intensity, mean_z]
  nodata: -9999.0
align:
  default_method: bilinear
  max_shift: 5
  confidence_threshold: 0.3
logging:
  level: INFO
  format: "%(message)s"
  datefmt: ""
  log_file: logs/test.log
"""


def _write_las(path: Path, with_mound: bool = True, n_veg: int = 150, seed: int = 0) -> int:
    """Мини-LAS: сетка земли 32x32 c шагом 0.5 м + купол + растительность."""
    import laspy

    rng = np.random.default_rng(seed)
    n = 32
    step = 0.5
    xs = 500000.0 + np.arange(n) * step
    ys = 6200000.0 + np.arange(n) * step
    xx, yy = np.meshgrid(xs, ys)
    gx, gy = xx.ravel(), yy.ravel()
    gz = 100.0 + 0.03 * rng.standard_normal(gx.size)
    if with_mound:
        cx, cy = xs[n // 2], ys[n // 2]
        gz += 1.5 * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 2.0**2))

    idx = rng.choice(gx.size, size=min(n_veg, gx.size), replace=False)
    vx, vy = gx[idx], gy[idx]
    # растительность минимум на height_threshold выше локальной земли
    vz = 100.0 + rng.uniform(1.0, 2.5, size=idx.size)
    if with_mound:
        cx, cy = xs[n // 2], ys[n // 2]
        vz += 1.5 * np.exp(-((vx - cx) ** 2 + (vy - cy) ** 2) / (2 * 2.0**2))

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([500000.0, 6200000.0, 0.0])
    las = laspy.LasData(header)
    las.x = np.concatenate([gx, vx])
    las.y = np.concatenate([gy, vy])
    las.z = np.concatenate([gz, vz])
    las.intensity = np.concatenate(
        [np.full(gx.size, 40, dtype=np.uint16), np.full(vx.size, 20, dtype=np.uint16)]
    )
    las.return_number = np.ones(gx.size + vx.size, dtype=np.uint8)
    las.classification = np.concatenate(
        [np.full(gx.size, 2, dtype=np.uint8), np.full(vx.size, 5, dtype=np.uint8)]
    )
    las.write(path)
    return gx.size + vx.size


@pytest.fixture()
def app_cfg(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_TEST_YAML, encoding="utf-8")
    return load_config(cfg_path)


# --- site_kind / find_lidar_files ---------------------------------------------


def test_site_kind(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert site_kind(empty) == "empty"
    assert site_kind(tmp_path / "nope") == "empty"

    las_site = tmp_path / "las_site"
    las_site.mkdir()
    _write_las(las_site / "site.las", n_veg=10)
    assert site_kind(las_site) == "lidar"
    assert find_lidar_files(las_site) == [las_site / "site.las"]

    ras_site = tmp_path / "ras_site"
    ras_site.mkdir()
    (ras_site / "SpOr_1.tif").write_bytes(b"")
    assert site_kind(ras_site) == "raster"

    built = tmp_path / "built"
    (built / "layers").mkdir(parents=True)
    (built / "layers" / "dtm.tif").write_bytes(b"")
    assert site_kind(built) == "built"


# --- build_site ----------------------------------------------------------------


def test_build_site_lidar_end_to_end(tmp_path: Path, app_cfg) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    n_points = _write_las(raw / "site.las")

    prof = Profiler()
    res = build_site(raw, raw, app_cfg, prof)

    assert res.built is True
    assert res.n_points_total == n_points
    assert res.n_points_ground > 0
    assert "dtm" in res.layer_names
    for name in ("slope", "hillshade_az_45", "point_density", "mean_intensity"):
        assert name in res.layer_names, res.layer_names

    layers = raw / "layers"
    for name in res.layer_names:
        assert (layers / f"{name}.tif").is_file(), name

    # порядок каналов: dtm первый (CONTRACTS.md §3)
    meta = json.loads((raw / "samples" / "sample_meta.json").read_text(encoding="utf-8"))
    assert meta["channel_names"][0] == "dtm"
    assert len(meta["channel_mask"]) == len(meta["channel_names"]) == meta["n_channels"]
    assert meta["n_available_channels"] == meta["n_channels"]
    assert meta["grid"]["crs"] == _CRS

    # купол виден в DTM
    import rasterio

    with rasterio.open(layers / "dtm.tif") as src:
        dtm = src.read(1)
    valid = dtm[dtm > -9000.0]
    assert float(valid.max() - valid.min()) >= 1.0

    # idempotentность и force
    res2 = build_site(raw, raw, app_cfg, Profiler())
    assert res2.built is False and res2.layer_names == res.layer_names
    res3 = build_site(raw, raw, app_cfg, Profiler(), force=True)
    assert res3.built is True

    # профиль стадий напечатан и непустой
    assert prof.report() is not None


def test_build_site_raster_only(tmp_path: Path, app_cfg) -> None:
    raw = tmp_path / "048_SpOr"
    raw.mkdir()
    for i in (1, 2, 3):
        (raw / f"SpOr_{i}.tif").write_bytes(b"")
    with pytest.raises(RawSiteUnsupportedError, match="нечего собрать"):
        build_site(raw, raw, app_cfg)


def test_build_site_empty_dir(tmp_path: Path, app_cfg) -> None:
    raw = tmp_path / "junk"
    raw.mkdir()
    (raw / "readme.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(RawSiteUnsupportedError, match="нет ни собранных слоёв"):
        build_site(raw, raw, app_cfg)


def test_build_site_las_without_points(tmp_path: Path, app_cfg) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    laspy.LasData(header).write(raw / "empty.las")
    with pytest.raises(RawSiteUnsupportedError, match="ни одной точки"):
        build_site(raw, raw, app_cfg)


# --- resolve_sites ---------------------------------------------------------------


def _make_built_site(root: Path, name: str) -> Path:
    site = root / name
    (site / "layers").mkdir(parents=True)
    (site / "layers" / "dtm.tif").write_bytes(b"")
    return site


def test_resolve_sites_single_raw_site(tmp_path: Path, app_cfg) -> None:
    site = tmp_path / "013_lidar"
    site.mkdir()
    _write_las(site / "site.las", n_veg=10)

    prof = Profiler()
    sites = resolve_sites(site, app_cfg, prof)
    assert sites == [site]
    assert (site / "layers" / "dtm.tif").is_file()
    assert site_kind(site) == "built"


def test_resolve_sites_nested_las_is_folder_mode(tmp_path: Path, app_cfg) -> None:
    # LAS в подпапке: 013_lidar трактуется как папка, участок — 013_lidar/las
    # (строгое правило: сырой участок = LAS прямо в корне указанной папки)
    site = tmp_path / "013_lidar"
    (site / "las").mkdir(parents=True)
    _write_las(site / "las" / "site.las", n_veg=10)

    sites = resolve_sites(site, app_cfg, Profiler())
    assert sites == [site / "las"]
    assert (site / "las" / "layers" / "dtm.tif").is_file()


def test_resolve_sites_folder_mixed(tmp_path: Path, app_cfg) -> None:
    root = tmp_path / "sites"
    root.mkdir()
    _make_built_site(root, "site_a")
    raw_site = root / "site_b"
    raw_site.mkdir()
    _write_las(raw_site / "site.las", n_veg=10)
    raster_site = root / "site_c"
    raster_site.mkdir()
    (raster_site / "x.tif").write_bytes(b"")
    (root / "junk").mkdir()  # пустые папки — junk, пропускается

    # cfg=None: готовый проходит, сырой лидарный требует конфиг
    with pytest.raises(RawSiteUnsupportedError, match="роли 1"):
        resolve_sites(root, None, Profiler())

    sites = resolve_sites(root, app_cfg, Profiler())
    assert [p.name for p in sites] == ["site_a", "site_b"]
    assert (raw_site / "layers" / "dtm.tif").is_file()


def test_resolve_sites_folder_all_raster(tmp_path: Path, app_cfg) -> None:
    """Растровые без 03_/04_/05_ собрать нечем — пустой список, не исключение.

    Раньше здесь летел RawSiteUnsupportedError, predict возвращал код 2,
    и платформа сжигала одну из трёх попыток запуска в сутки.
    """
    root = tmp_path / "sites"
    root.mkdir()
    for name in ("048_SpOr", "101_Li"):
        site = root / name
        site.mkdir()
        (site / "ch.tif").write_bytes(b"")

    assert resolve_sites(root, app_cfg, Profiler()) == []


def test_resolve_sites_built_passthrough(tmp_path: Path, app_cfg) -> None:
    # готовый участок как --data: без AppConfig, без сборки
    site = _make_built_site(tmp_path, "site_1")
    assert resolve_sites(site, None, Profiler()) == [site]


def _write_src_tif(path: Path, west: float, south: float, size: int = 200) -> None:
    """Валидный GeoTIFF 3x(size)x(size) в EPSG:32636 — как источник организаторов."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(west, south + size, 1.0, 1.0),
    ) as dst:
        dst.write(np.full((3, size, size), 90.0, dtype="float32"))


def test_resolve_sites_single_raster_builds_rgb(tmp_path: Path, app_cfg) -> None:
    """Регион платформы: снимок в подпапке 04_, разметки нет — rgb_* собираются.

    Без этого 31 растровый регион из 48 давал пустой ответ: site_kind видел
    только корень, а источник лежит в подпапке-модальности.
    """
    site = tmp_path / "101_SpOr"
    _write_src_tif(site / "04_x_SpOr_спутник" / "s.tif", 500_000.0, 6_000_000.0)

    assert resolve_sites(site, app_cfg, Profiler()) == [site]
    for name in ("rgb_r", "rgb_g", "rgb_b"):
        assert (site / "layers" / f"{name}.tif").is_file()
    assert site_kind(site) == "built"


def test_resolve_sites_geo_only_region_is_skipped(tmp_path: Path, app_cfg) -> None:
    """Регион только с геофизикой (07_..09_) — пропуск, не падение."""
    site = tmp_path / "300_MagGr"
    (site / "07_x_MagGr").mkdir(parents=True)
    (site / "07_x_MagGr" / "MagGr.xyz").write_text("0 0 0\n", encoding="utf-8")

    assert resolve_sites(site, app_cfg, Profiler()) == []
