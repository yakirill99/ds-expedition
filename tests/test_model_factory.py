import pytest
import torch

from expds.models import (
    KozLoss,
    LossConfig,
    ModelConfig,
    build_model,
    input_divisor,
    load_checkpoint,
    out_channels,
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
    assert hm == pytest.approx(2 * 64 * 64 * per_px / 50.0, rel=1e-4)


# ============================================================
# Расширение DoD 01-model: границы конфига, контракты голов,
# инициализация prior для unet, чекпоинт, hm-warmup.
# ============================================================


def test_out_channels_is_two_heads_per_class():
    """Выход = 2K: по каналу на seg и hm для каждого класса."""
    assert out_channels(K) == 2 * K
    assert out_channels(1) == 2


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_config_rejects_prior_out_of_range(bad):
    """prior_prob вне (0, 1) ломает инициализацию bias (log от <=0)."""
    with pytest.raises(ValueError, match="prior_prob"):
        ModelConfig.from_dict({"arch": "stub", "prior_prob": bad})


def test_config_defaults():
    cfg = ModelConfig.from_dict(None)
    assert cfg.arch == "unet"
    assert cfg.encoder == "resnet34"
    assert cfg.prior_prob == 0.01


def test_prior_bias_unet():
    """prior применяется и к unet, не только к stub (ищется последний Conv2d)."""
    m = build_model(ModelConfig(arch="unet", encoder="resnet18", prior_prob=0.02), C, K)
    p = m.predict(torch.zeros(1, C, 64, 64))
    assert torch.allclose(p, torch.full_like(p, 0.02), atol=1e-3)


@pytest.mark.parametrize("prior", [0.001, 0.05, 0.5])
def test_prior_bias_values(prior):
    m = build_model(ModelConfig(arch="stub", prior_prob=prior), C, K)
    p = m.predict(torch.zeros(1, C, 8, 8))
    assert torch.allclose(p, torch.full_like(p, prior), atol=1e-4)


def test_stub_rejects_even_kernel():
    """Чётное ядро не даёт 'same' padding -> форма выхода уедет."""
    with pytest.raises(ValueError, match="odd"):
        build_model(ModelConfig(arch="stub", stub_kernel=4), C, K)


def test_split_heads_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="expected"):
        split_heads(torch.zeros(2 * K, 4, 4), K)


def test_split_heads_order_is_seg_then_hm():
    """Порядок голов — контракт: первые K каналов seg, следующие K — hm."""
    logits = torch.zeros(1, 2 * K, 4, 4)
    logits[:, :K] = 1.0
    logits[:, K:] = 2.0
    h = split_heads(logits, K)
    assert torch.all(h["seg"] == 1.0)
    assert torch.all(h["hm"] == 2.0)


def test_unet_input_divisible_by_32():
    """H и W должны делиться на input_divisor, иначе smp падает на skip-связях."""
    cfg = ModelConfig(arch="unet", encoder="resnet18")
    d = input_divisor(cfg)
    m = build_model(cfg, C, K)
    y = m(torch.randn(1, C, 2 * d, 3 * d))
    assert y.shape == (1, 2 * K, 2 * d, 3 * d)


def test_predict_returns_probabilities():
    """predict = sigmoid(forward): значения в (0, 1)."""
    m = build_model(ModelConfig(arch="stub"), C, K)
    p = m.predict(torch.randn(2, C, 16, 16))
    assert p.shape == (2, 2 * K, 16, 16)
    assert float(p.min()) > 0.0 and float(p.max()) < 1.0


def test_checkpoint_classes_mismatch(tmp_path):
    """Несовпадение classes ловится так же, как channel_names."""
    names = [f"c{i}" for i in range(C)]
    classes = ["mound", "rampart", "ditch"]
    cfg = ModelConfig(arch="stub")
    p = tmp_path / "m.pt"
    save_checkpoint(p, build_model(cfg, C, K), cfg, names, classes)
    with pytest.raises(ValueError, match="classes"):
        load_checkpoint(p, channel_names=names, classes=["mound", "ditch", "rampart"])


def test_checkpoint_keeps_model_cfg(tmp_path):
    """arch/encoder восстанавливаются из чекпоинта без внешнего конфига."""
    names = [f"c{i}" for i in range(C)]
    classes = ["a", "b", "c"]
    cfg = ModelConfig(arch="unet", encoder="resnet18", prior_prob=0.03)
    p = tmp_path / "m.pt"
    save_checkpoint(p, build_model(cfg, C, K), cfg, names, classes)
    _m, meta = load_checkpoint(p)
    assert ModelConfig.from_dict(meta["model_cfg"]) == cfg
    assert meta["channel_names"] == names and meta["classes"] == classes


def test_checkpoint_loads_in_eval_mode(tmp_path):
    """После загрузки модель в eval: инференс не должен зависеть от dropout/BN."""
    cfg = ModelConfig(arch="stub")
    p = tmp_path / "m.pt"
    save_checkpoint(p, build_model(cfg, C, K), cfg, ["c"] * C, ["a", "b", "c"])
    m, _ = load_checkpoint(p)
    assert not m.training


def test_loss_hm_weight_zero_equals_seg_only():
    """Контракт warmup: при hm_weight=0 полный лосс = только seg-компонента."""
    y_seg, y_hm = _targets()
    logits = torch.randn(2, 2 * K, 32, 32)
    full = KozLoss(K, LossConfig(hm_weight=0.0))(logits, y_seg, y_hm)
    cfg = LossConfig()
    expected = cfg.seg_weight * (
        cfg.dice_weight * full["seg_dice"] + cfg.focal_weight * full["seg_focal"]
    )
    assert full["loss"] == pytest.approx(float(expected), rel=1e-5)


def test_loss_components_are_detached():
    """seg_dice/seg_focal/hm — для логов, градиент только через loss."""
    logits = torch.randn(2, 2 * K, 16, 16, requires_grad=True)
    z = torch.zeros(2, K, 16, 16)
    out = KozLoss(K)(logits, z, z)
    assert out["loss"].requires_grad
    for key in ("seg_dice", "seg_focal", "hm"):
        assert not out[key].requires_grad, key


def test_loss_hm_min_norm_scales_inversely():
    """hm нормируется на max(масса, hm_min_norm): больше порог — меньше лосс."""
    logits = torch.zeros(2, 2 * K, 32, 32)
    z = torch.zeros(2, K, 32, 32)
    a = KozLoss(K, LossConfig(hm_min_norm=50.0))(logits, z, z)["hm"].item()
    b = KozLoss(K, LossConfig(hm_min_norm=2000.0))(logits, z, z)["hm"].item()
    assert a == pytest.approx(b * 40.0, rel=1e-4)


def test_build_model_single_class():
    """K=1 — выход 2 канала, граничный случай для split_heads."""
    m = build_model(ModelConfig(arch="stub"), C, 1)
    y = m(torch.randn(1, C, 8, 8))
    assert y.shape == (1, 2, 8, 8)
    h = split_heads(y, 1)
    assert h["seg"].shape == h["hm"].shape == (1, 1, 8, 8)
