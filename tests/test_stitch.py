import numpy as np
import pytest
import torch

from expds.fusion import StitchConfig, Stitcher, make_window
from expds.tiles.index import Window


def _starts(n, tile, overlap):
    step = tile - overlap
    s = list(range(0, max(n - tile, 0) + 1, step))
    if s[-1] + tile < n:
        s.append(n - tile)
    return s


def _windows(h, w, tile, overlap):
    return [
        Window(row0=r, col0=c, height=min(tile, h), width=min(tile, w))
        for r in _starts(h, tile, overlap)
        for c in _starts(w, tile, overlap)
    ]


def test_make_window():
    w = make_window(64, "hann", 1e-3)
    assert w.shape == (64, 64) and w.dtype == np.float32
    assert w.min() >= 1e-6 and w.max() <= 1.0
    assert np.allclose(w, w[::-1, :]) and np.allclose(w, w.T)
    assert np.all(make_window(8, "flat") == 1)
    with pytest.raises(ValueError):
        make_window(8, "gauss")


def test_config_rejects_bad():
    with pytest.raises(ValueError):
        StitchConfig.from_dict({"window": "gauss"})
    with pytest.raises(ValueError):
        StitchConfig.from_dict({"windw": "hann"})


@pytest.mark.parametrize("kind", ["hann", "flat"])
def test_constant_tiles(kind):
    h, w, tile, ov = 200, 300, 64, 16
    s = Stitcher(2, h, w, tile, StitchConfig(window=kind))
    for win in _windows(h, w, tile, ov):
        s.add(np.full((2, tile, tile), 0.7, np.float32), win)
    out = s.result()
    assert s.coverage.all()
    assert np.allclose(out, 0.7, atol=1e-6)


def test_exact_crops_reconstruct_field():
    h, w, tile, ov = 150, 170, 64, 16
    r, c = np.mgrid[:h, :w]
    f = (np.sin(r / 20) + np.cos(c / 30)).astype(np.float32)[None]
    s = Stitcher(1, h, w, tile)
    for win in _windows(h, w, tile, ov):
        s.add(f[:, win.row0 : win.row0 + tile, win.col0 : win.col0 + tile], win)
    assert np.allclose(s.result(), f, atol=1e-5)


def _max_step(img):
    return max(np.abs(np.diff(img, axis=0)).max(), np.abs(np.diff(img, axis=1)).max())


def test_hann_smoother_than_flat():
    h, w, tile, ov = 200, 200, 64, 16
    wins = _windows(h, w, tile, ov)
    vals = np.random.default_rng(0).random(len(wins)).astype(np.float32)
    steps = {}
    for kind in ("hann", "flat"):
        s = Stitcher(1, h, w, tile, StitchConfig(window=kind))
        for v, win in zip(vals, wins, strict=True):
            s.add(np.full((1, tile, tile), v, np.float32), win)
        steps[kind] = _max_step(s.result()[0])
    assert steps["hann"] < 0.5 * steps["flat"]


def test_add_batch_tensor_matches_single():
    h, w, tile, ov = 100, 120, 64, 16
    wins = _windows(h, w, tile, ov)
    preds = torch.randn(len(wins), 3, tile, tile)
    wt = torch.tensor([[x.row0, x.col0, x.height, x.width] for x in wins])
    a, b = Stitcher(3, h, w, tile), Stitcher(3, h, w, tile)
    a.add_batch(preds, wt)
    for p, win in zip(preds, wins, strict=True):
        b.add(p, win)
    assert np.allclose(a.result(), b.result())


def test_padded_tile_on_small_grid():
    h, w, tile = 50, 40, 64
    pred = np.random.default_rng(1).random((1, tile, tile)).astype(np.float32)
    s = Stitcher(1, h, w, tile)
    s.add(pred, (0, 0, h, w))
    assert np.allclose(s.result(), pred[:, :h, :w])


def test_uncovered_fill_and_errors():
    s = Stitcher(1, 100, 100, 32)
    s.add(np.ones((1, 32, 32), np.float32), (0, 0, 32, 32))
    out = s.result(fill=-1.0)
    assert out[0, 50, 50] == -1.0 and not s.coverage[50, 50]
    assert np.isclose(out[0, 5, 5], 1.0)
    with pytest.raises(ValueError):
        s.add(np.ones((2, 32, 32), np.float32), (0, 0, 32, 32))
    with pytest.raises(ValueError):
        s.add(np.ones((1, 32, 32), np.float32), (80, 80, 32, 32))
    with pytest.raises(ValueError):
        s.add(np.ones((1, 32, 32), np.float32), (0, 0, 32))
