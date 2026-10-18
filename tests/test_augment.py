"""Тесты аугментаций: синхронность геометрии, dropout, warmup.

Ключевая проверка — синхронность: если x повернулся, а y_seg нет,
модель учится на разъехавшихся масках, и это не видно по лоссу.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from expds.tiles.augment import (
    ROT90_CHOICES,
    AugmentConfig,
    AugmentedDataset,
    channel_dropout,
    geometric,
    warmup_hm_weight,
)


def _batch(b: int = 2, c: int = 3, k: int = 2, t: int = 8):
    """Батч с узнаваемым содержимым: значение = номер канала + строка."""
    x = torch.zeros(b, c, t, t)
    for ci in range(c):
        for r in range(t):
            x[:, ci, r, :] = ci * 100 + r
    y_seg = (x[:, :k] > 0).float()
    y_hm = x[:, :k] / 1000.0
    valid = torch.ones(b, t, t, dtype=torch.bool)
    return x, y_seg, y_hm, valid


# --- геометрия синхронна ------------------------------------------------------


def test_noop_returns_same():
    x, y_seg, y_hm, valid = _batch()
    x2, s2, h2, v2 = geometric(x, y_seg, y_hm, valid)
    assert torch.equal(x, x2)
    assert torch.equal(y_seg, s2)
    assert torch.equal(y_hm, h2)
    assert torch.equal(valid, v2)


def test_hflip_applied_to_all():
    x, y_seg, y_hm, valid = _batch()
    x2, s2, h2, v2 = geometric(x, y_seg, y_hm, valid, hflip=True)
    assert torch.equal(x2, torch.flip(x, dims=[-1]))
    assert torch.equal(s2, torch.flip(y_seg, dims=[-1]))
    assert torch.equal(h2, torch.flip(y_hm, dims=[-1]))
    assert torch.equal(v2, torch.flip(valid, dims=[-1]))


def test_vflip_applied_to_all():
    x, y_seg, y_hm, valid = _batch()
    x2, s2, h2, v2 = geometric(x, y_seg, y_hm, valid, vflip=True)
    assert torch.equal(x2, torch.flip(x, dims=[-2]))
    assert torch.equal(s2, torch.flip(y_seg, dims=[-2]))
    assert torch.equal(h2, torch.flip(y_hm, dims=[-2]))
    assert torch.equal(v2, torch.flip(valid, dims=[-2]))


@pytest.mark.parametrize("k", ROT90_CHOICES)
def test_rot90_applied_to_all(k):
    x, y_seg, y_hm, valid = _batch()
    x2, s2, h2, v2 = geometric(x, y_seg, y_hm, valid, rot90=k)
    if k == 0:
        assert torch.equal(x, x2)
    else:
        assert torch.equal(x2, torch.rot90(x, k=k, dims=[-2, -1]))
        assert torch.equal(s2, torch.rot90(y_seg, k=k, dims=[-2, -1]))
        assert torch.equal(h2, torch.rot90(y_hm, k=k, dims=[-2, -1]))
        assert torch.equal(v2, torch.rot90(valid, k=k, dims=[-2, -1]))


def test_combined_symmetry_consistent():
    """x, y_seg, y_hm, valid преобразованы одной и той же операцией."""
    x, y_seg, y_hm, valid = _batch()
    x2, s2, h2, v2 = geometric(x, y_seg, y_hm, valid, hflip=True, vflip=True, rot90=1)
    # Равенство масок проверяем через инвариант: y_seg == (x > 0) сохраняется.
    expected_seg = (x2[:, : y_seg.shape[1]] > 0).float()
    assert torch.equal(s2, expected_seg)
    assert torch.equal(v2, torch.ones_like(v2))
    assert torch.equal(h2, x2[:, : y_hm.shape[1]] / 1000.0)


def test_rot90_preserves_shape_for_square():
    x, y_seg, y_hm, valid = _batch(t=8)
    x2, s2, h2, v2 = geometric(x, y_seg, y_hm, valid, rot90=1)
    assert x2.shape == x.shape
    assert s2.shape == y_seg.shape
    assert h2.shape == y_hm.shape
    assert v2.shape == valid.shape


def test_none_targets_allowed():
    """Инференс без разметки: y_seg/y_hm = None, геометрия всё равно работает."""
    x, _, _, valid = _batch()
    x2, s2, h2, v2 = geometric(x, None, None, valid, hflip=True, rot90=2)
    assert s2 is None and h2 is None
    assert torch.equal(v2, torch.rot90(torch.flip(valid, dims=[-1]), k=2, dims=[-2, -1]))


# --- channel dropout ----------------------------------------------------------


def test_channel_dropout_zeroes_and_unsets_mask():
    x, _, _, _ = _batch(b=1, c=3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    x2, m2 = channel_dropout(x, mask, p=1.0, generator=g)
    assert int((~m2).sum()) == 1, "должен выключиться ровно один канал"
    dropped = int(torch.nonzero(~m2[0]).item())
    assert torch.equal(x2[0, dropped], torch.zeros_like(x2[0, dropped]))
    # остальные каналы целы
    for ci in range(3):
        if ci != dropped:
            assert torch.equal(x2[0, ci], x[0, ci])


def test_channel_dropout_p_zero_is_noop():
    x, _, _, _ = _batch(b=1, c=3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    x2, m2 = channel_dropout(x, mask, p=0.0)
    assert torch.equal(x, x2)
    assert torch.equal(mask, m2)


def test_channel_dropout_only_available():
    """Канал с channel_mask=False не выбирается — он и так нулевой."""
    x, _, _, _ = _batch(b=1, c=3)
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        # маска заново на каждой итерации: гашение накапливается
        mask = torch.tensor([[True, False, True]])
        _x2, m2 = channel_dropout(x, mask, p=1.0, generator=g)
        # вновь погашенные = было True, стало False (канал 1 выключен изначально)
        newly = torch.nonzero(mask[0] & ~m2[0], as_tuple=False).flatten()
        assert newly.numel() == 1, f"должен погаснуть ровно один канал, погасло {newly.tolist()}"
        assert int(newly[0]) in (0, 2), f"выбран недоступный канал {int(newly[0])}"


def test_channel_dropout_rejects_shape_mismatch():
    """channel_mask (1, C) при x (2, C, T, T) — ошибка, а не тихий IndexError."""
    x, _, _, _ = _batch(b=2, c=3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="channel_mask"):
        channel_dropout(x, mask, p=1.0)


def test_channel_dropout_preserves_shape():
    x, _, _, _ = _batch(b=4, c=5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    x2, m2 = channel_dropout(x, mask, p=0.5, generator=torch.Generator().manual_seed(3))
    assert x2.shape == x.shape
    assert m2.shape == mask.shape


# --- AugmentConfig ------------------------------------------------------------


def test_config_defaults_disabled_by_zero():
    cfg = AugmentConfig(hflip_p=0.0, vflip_p=0.0, rot90_p=0.0, channel_dropout_p=0.0)
    assert cfg.enabled is False


def test_config_enabled_when_any_positive():
    assert AugmentConfig(hflip_p=1.0).enabled is True
    assert AugmentConfig(channel_dropout_p=0.1).enabled is True


def test_config_from_dict_roundtrip():
    cfg = AugmentConfig.from_dict({"hflip_p": 0.3, "rot90_p": 0.2})
    assert cfg.hflip_p == pytest.approx(0.3)
    assert cfg.rot90_p == pytest.approx(0.2)
    assert cfg.vflip_p == pytest.approx(0.5), "остальные — дефолты"


def test_config_from_dict_rejects_unknown():
    with pytest.raises(ValueError, match="unknown keys"):
        AugmentConfig.from_dict({"hflip_p": 0.5, "scale": 0.2})


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_config_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match=r"должен быть в \[0, 1\]"):
        AugmentConfig.from_dict({"hflip_p": bad})


# --- AugmentedDataset ---------------------------------------------------------


class _FakeDataset(torch.utils.data.Dataset):
    """Минимальный двойник KozDataset: словарь нужных ключей."""

    def __init__(self, n: int = 4, c: int = 2, t: int = 8) -> None:
        self.n, self.c, self.t = n, c, t

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.arange(self.c * self.t * self.t, dtype=torch.float32).reshape(
            self.c, self.t, self.t
        )
        item = {
            "x": x,
            "channel_mask": torch.ones(self.c, dtype=torch.bool),
            "valid": torch.ones(self.t, self.t, dtype=torch.bool),
            "site": 0,
            "window": torch.tensor([0, 0, self.t, self.t], dtype=torch.int64),
        }
        if idx % 2 == 0:
            item["y_seg"] = (x[:1] > 0).float()
            item["y_hm"] = x[:1] / 100.0
        return item

    def item_window(self, idx: int):
        return 0, None

    def sample_weights(self, pos_fraction: float):
        return np.ones(self.n)


def test_wrapper_proxies_len_and_attrs():
    base = _FakeDataset(n=7)
    ds = AugmentedDataset(base, AugmentConfig(hflip_p=1.0))
    assert len(ds) == 7
    assert np.allclose(ds.sample_weights(0.7), np.ones(7))


def test_wrapper_zero_prob_returns_identical():
    base = _FakeDataset()
    ds = AugmentedDataset(
        base, AugmentConfig(hflip_p=0.0, vflip_p=0.0, rot90_p=0.0, channel_dropout_p=0.0)
    )
    for i in range(len(base)):
        a, b = base[i], ds[i]
        assert torch.equal(a["x"], b["x"])
        assert torch.equal(a["valid"], b["valid"])


def test_wrapper_geometric_sync_on_item():
    base = _FakeDataset()
    ds = AugmentedDataset(base, AugmentConfig(hflip_p=1.0, vflip_p=1.0, rot90_p=0.0))
    out = ds[0]
    # двойное отражение по H и V = инвариант для этой фикстуры (x растёт по обоим)
    ref = base[0]["x"]
    assert out["x"].shape == ref.shape
    # y_seg обязан быть согласован с x: (x > 0) после тех же отражений
    assert torch.equal(out["y_seg"], (out["x"][:1] > 0).float())


def test_wrapper_deterministic_per_epoch_and_index():
    base = _FakeDataset()
    cfg = AugmentConfig(hflip_p=0.5, vflip_p=0.5, rot90_p=0.5)
    a = AugmentedDataset(base, cfg, seed=42)
    b = AugmentedDataset(base, cfg, seed=42)
    for i in range(len(base)):
        assert torch.equal(a[i]["x"], b[i]["x"])


def test_wrapper_epoch_changes_result():
    base = _FakeDataset()
    cfg = AugmentConfig(hflip_p=0.0, vflip_p=0.0, rot90_p=1.0)
    ds = AugmentedDataset(base, cfg, seed=0)
    first = ds[0]["x"].clone()
    for ep in range(1, 10):
        ds.set_epoch(ep)
        if not torch.equal(ds[0]["x"], first):
            return
    pytest.fail("аугментации не меняются между эпохами")


def test_wrapper_keeps_window_key():
    base = _FakeDataset()
    ds = AugmentedDataset(base, AugmentConfig(hflip_p=1.0))
    assert "window" in ds[0]
    assert ds[0]["window"].dtype == torch.int64


def test_wrapper_no_targets_when_absent():
    """Нечётный индекс в фикстуре — без разметки."""
    base = _FakeDataset()
    ds = AugmentedDataset(base, AugmentConfig(hflip_p=1.0, rot90_p=1.0))
    out = ds[1]
    assert "y_seg" not in out
    assert "y_hm" not in out


# --- warmup -------------------------------------------------------------------


def test_warmup_hm_zero_first_epoch():
    assert warmup_hm_weight(1, 1.0) == 0.0


def test_warmup_hm_full_from_third_epoch():
    assert warmup_hm_weight(3, 1.0) == pytest.approx(1.0)
    assert warmup_hm_weight(10, 1.0) == pytest.approx(1.0)


def test_warmup_hm_ramp_second_epoch():
    assert warmup_hm_weight(2, 1.0) == pytest.approx(0.5)
    assert warmup_hm_weight(2, 2.0) == pytest.approx(1.0)
