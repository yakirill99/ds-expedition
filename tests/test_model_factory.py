import pytest
import torch

from expds.models import (
    KozLoss,
    LossConfig,
    ModelConfig,
    build_model,
    input_divisor,
    load_checkpoint,
    save_checkpoint,
    split_heads,
)

C, K = 19, 3


@pytest.mark.parametrize(("arch", "enc"), [("stub", "resnet34"), ("unet", "resnet18")])
def test_forward_shape(arch, enc):
    torch.manual_seed(0)
    m = build_model(ModelConfig(arch=arch, encoder=enc), C, K)
    y = m(torch.randn(2, C, 64, 64))
    assert y.shape == (2, 2 * K, 64, 64)
    h = split_heads(y, K)
    assert h["seg"].shape == h["hm"].shape == (2, K, 64, 64)


def test_prior_bias():
    m = build_model(ModelConfig(arch="stub", prior_prob=0.01), C, K)
    p = m.predict(torch.zeros(1, C, 8, 8))
    assert torch.allclose(p, torch.full_like(p, 0.01), atol=1e-4)


def test_input_divisor():
    assert input_divisor(ModelConfig(arch="unet")) == 32
    assert input_divisor(ModelConfig(arch="stub")) == 1


def test_config_rejects_bad():
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"arch": "stub", "encodr": "x"})
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"arch": "vit"})


def test_split_heads_bad_channels():
    with pytest.raises(ValueError):
        split_heads(torch.zeros(1, 5, 4, 4), K)


def _targets(b=2, h=32, w=32):
    g = torch.Generator().manual_seed(0)
    y_seg = (torch.rand(b, K, h, w, generator=g) > 0.8).float()
    y_hm = torch.rand(b, K, h, w, generator=g) ** 4
    return y_seg, y_hm


def _perfect(y_seg, y_hm):
    return torch.cat([20 * (2 * y_seg - 1), torch.logit(y_hm.clamp(1e-6, 1 - 1e-6))], 1)


def test_loss_perfect_vs_zero():
    y_seg, y_hm = _targets()
    loss = KozLoss(K)
    good = loss(_perfect(y_seg, y_hm), y_seg, y_hm)["loss"].item()
    bad = loss(torch.zeros(2, 2 * K, 32, 32), y_seg, y_hm)["loss"].item()
    assert good < 1e-3
    assert bad > 0.1


def test_loss_ignores_invalid():
    y_seg, y_hm = _targets()
    valid = torch.zeros(2, 32, 32, dtype=torch.bool)
    valid[:, :, :16] = True
    a = _perfect(y_seg, y_hm)
    b = a.clone()
    b[..., 16:] = torch.randn_like(b[..., 16:]) * 5
    loss = KozLoss(K)
    assert torch.allclose(loss(a, y_seg, y_hm, valid)["loss"], loss(b, y_seg, y_hm, valid)["loss"])


def test_loss_no_objects_finite_grad():
    logits = torch.randn(2, 2 * K, 32, 32, requires_grad=True)
    z = torch.zeros(2, K, 32, 32)
    out = KozLoss(K)(logits, z, z)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(logits.grad).all()
    assert {"loss", "seg_dice", "seg_focal", "hm"} <= set(out)


def test_loss_half_logits_float32():
    y_seg, y_hm = _targets()
    out = KozLoss(K)(_perfect(y_seg, y_hm).half(), y_seg, y_hm)
    assert out["loss"].dtype == torch.float32


def test_stub_learns_simple_rule():
    torch.manual_seed(0)
    x = torch.randn(4, C, 32, 32)
    y_seg = (x[:, :K] > 0.5).float()
    y_hm = torch.sigmoid(4 * x[:, K : 2 * K])
    m = build_model(ModelConfig(arch="stub"), C, K)
    loss = KozLoss(K)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    first = None
    for _ in range(150):
        opt.zero_grad()
        v = loss(m(x), y_seg, y_hm)["loss"]
        v.backward()
        opt.step()
        first = v.item() if first is None else first
    assert v.item() < 0.5 * first


def test_checkpoint_roundtrip(tmp_path):
    names = [f"c{i}" for i in range(C)]
    classes = ["mound", "rampart", "ditch"]
    cfg = ModelConfig(arch="stub")
    m = build_model(cfg, C, K)
    p = tmp_path / "best.pt"
    save_checkpoint(p, m, cfg, names, classes, extra={"epoch": 3})
    m2, meta = load_checkpoint(p, channel_names=names, classes=classes)
    x = torch.randn(1, C, 16, 16)
    assert torch.allclose(m.predict(x), m2.predict(x))
    assert meta["extra"]["epoch"] == 3
    with pytest.raises(ValueError, match="channel_names"):
        load_checkpoint(p, channel_names=names[::-1])


def test_hm_loss_floor_without_points():
    """Батч без точек: hm нормируется на hm_min_norm, а не на 1."""
    logits = torch.zeros(2, 2 * K, 64, 64)
    z = torch.zeros(2, K, 64, 64)
    hm = KozLoss(K, LossConfig(hm_min_norm=50.0))(logits, z, z)["hm"].item()
    per_px = 0.25 * float(torch.log(torch.tensor(2.0)))  # |0.5 - 0|^2 * BCE(0.5, 0)
    assert hm == pytest.approx(2 * K * 64 * 64 * per_px / 50.0, rel=1e-4)
