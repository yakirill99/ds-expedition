"""Тесты expds.tiles.dataset и expds.pipeline_config (на синтетике + ручные участки)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from expds.features.config import load_config
from expds.features.grid import Grid
from expds.pipeline_config import PipelineConfig, load_pipeline_config
from expds.tiles.dataset import KozDataset, SiteData, load_site

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "dataset_synthetic.yaml"
_TOOL = _ROOT / "tools" / "make_synthetic.py"
_SIZE = 256
_TILE, _OVERLAP = 128, 16
_ABSENT = {"point_density", "mean_intensity"}  # cloud features: в синтетике их нет


def _load_tool() -> ModuleType:
    name = "make_synthetic"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pcfg() -> PipelineConfig:
    return load_pipeline_config(_CONFIG)


@pytest.fixture(scope="module")
def sites(tmp_path_factory: pytest.TempPathFactory, pcfg: PipelineConfig) -> list[SiteData]:
    out = tmp_path_factory.mktemp("synth")
    argv = ["--config", str(_CONFIG), "--n-sites", "2", "--size", str(_SIZE), "--out-dir", str(out)]
    assert _load_tool().main(argv) == 0
    app = load_config(_CONFIG)
    return [
        load_site(
            out / f"site_{i}",
            channels=pcfg.data.channels,
            classes=pcfg.targets.classes,
            heatmap_sigma_px=pcfg.targets.heatmap_sigma_px,
            labels_crs=app.crs.output,
            labels_file=pcfg.data.labels_file,
            normalize=pcfg.data.normalize,
        )
        for i in (1, 2)
    ]


def _manual_site(width: int, height: int, labels: bool = True) -> SiteData:
    grid = Grid(crs="EPSG:32637", pixel_size=1.0, x_min=0.0, y_min=0.0, width=width, height=height)
    y_seg = np.zeros((1, height, width), dtype=np.float32)
    y_seg[0, :10, :10] = 1.0
    return SiteData(
        name="manual",
        grid=grid,
        x=np.ones((2, height, width), dtype=np.float32),
        channel_names=("a", "b"),
        channel_mask=np.array([True, False]),
        valid=np.ones((height, width), dtype=bool),
        y_seg=y_seg if labels else None,
        y_hm=np.zeros_like(y_seg) if labels else None,
    )


def test_pipeline_config(pcfg: PipelineConfig) -> None:
    assert pcfg.data.channels[0] == "dtm"
    assert "aspect" not in pcfg.data.channels
    assert _ABSENT <= set(pcfg.data.channels)
    assert pcfg.targets.classes == ("mound", "rampart", "ditch")


def test_load_site_channels_mask_targets(sites: list[SiteData], pcfg: PipelineConfig) -> None:
    site = sites[0]
    n_ch, n_cls = len(pcfg.data.channels), len(pcfg.targets.classes)
    assert site.x.shape == (n_ch, _SIZE, _SIZE) and site.x.dtype == np.float32
    absent = {n for n, m in zip(site.channel_names, site.channel_mask, strict=True) if not m}
    assert absent == _ABSENT
    for name in _ABSENT:
        assert not site.x[site.channel_names.index(name)].any()
    assert site.valid.all()
    assert site.y_seg is not None and site.y_seg.shape == (n_cls, _SIZE, _SIZE)
    assert site.y_hm is not None and site.y_hm.shape == (n_cls, _SIZE, _SIZE)


def test_site_robust_normalization(sites: list[SiteData]) -> None:
    dtm = sites[0].x[sites[0].channel_names.index("dtm")]
    assert abs(float(np.median(dtm))) < 1e-3
    assert float(np.percentile(dtm, 98) - np.percentile(dtm, 2)) == pytest.approx(1.0, abs=1e-3)


def test_items_match_site_arrays(sites: list[SiteData], pcfg: PipelineConfig) -> None:
    ds = KozDataset(sites, _TILE, _OVERLAP)
    assert len(ds) == 2 * 9  # 256 при тайле 128 / шаге 112: старты 0, 112, 128 по каждой оси
    for idx in (0, 4, len(ds) - 1):
        item = ds[idx]
        site_idx, window = ds.item_window(idx)
        rows, cols = window.slices
        site = sites[site_idx]
        assert item["x"].shape == (len(pcfg.data.channels), _TILE, _TILE)
        assert np.array_equal(item["x"].numpy(), site.x[:, rows, cols])
        assert np.array_equal(item["y_seg"].numpy(), site.y_seg[:, rows, cols])
        assert tuple(item["window"].tolist()) == window.as_tuple()


def test_targets_align_with_relief_in_tiles(sites: list[SiteData]) -> None:
    """В тайлах SLRM на пиках heatmap курганов выше фона — каналы и таргеты не разъехались."""
    ds = KozDataset(sites, _TILE, _OVERLAP)
    k_slrm = sites[0].channel_names.index("slrm_sigma_8.0")
    inside, outside = [], []
    for idx in range(len(ds)):
        item = ds[idx]
        slrm, peak = item["x"][k_slrm].numpy(), item["y_hm"][0].numpy() > 0.5
        inside.append(slrm[peak])
        outside.append(slrm[(item["y_seg"].sum(0).numpy() == 0) & ~peak])
    assert float(np.concatenate(inside).mean()) > float(np.concatenate(outside).mean()) + 0.1


def test_sample_weights_fraction() -> None:
    ds = KozDataset([_manual_site(128, 128)], 64, 0)
    assert len(ds) == 4 and int(ds.positive.sum()) == 1
    weights = ds.sample_weights(0.7)
    assert float(weights[ds.positive].sum()) == pytest.approx(0.7)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_padding_small_site() -> None:
    ds = KozDataset([_manual_site(100, 90)], 128, 16)
    item = ds[0]
    assert item["x"].shape == (2, 128, 128)
    assert bool(item["valid"][:90, :100].all()) and not bool(item["valid"][90:, :].any())
    assert not bool(item["valid"][:, 100:].any())


def test_no_labels_items() -> None:
    ds = KozDataset([_manual_site(64, 64, labels=False)], 64, 0)
    assert "y_seg" not in ds[0] and len(ds.positive) == 0
    assert ds.sample_weights(0.7).tolist() == [1.0]


def test_mixed_channels_raise() -> None:
    a = _manual_site(64, 64)
    b = SiteData(**{**a.__dict__, "channel_names": ("a", "c")})
    with pytest.raises(ValueError, match="каналы"):
        KozDataset([a, b], 64, 0)
