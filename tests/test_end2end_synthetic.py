"""E2E на синтетике: генерация -> обучение (stub, CPU) -> inference/entrypoint.py -> метрика.

Проверяем плумбинг, а не качество: stub (линейная свёртка 3x3) упирается в F1 ~0.14.
Главная проверка — TP/FP/FN entrypoint совпадают с val лучшей эпохи обучения: пути
train-val и инференса (нормализация, каналы, тайлинг, CRS) не расходятся.
Качество U-Net — ручной прогон на GPU (scripts/train.py), не CI.
"""

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from expds.eval import evaluate_files
from expds.pipeline_config import load_pipeline_config
from expds.train import train_from_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "dataset_synthetic.yaml"
E2E_EPOCHS = 8
E2E_TILES_PER_EPOCH = 256
MIN_F1 = 0.1  # stub учится: эпоха 1 ~0.03, плато ~0.14
MAX_SECONDS = 180


def _make_synthetic():
    spec = importlib.util.spec_from_file_location(
        "make_synthetic", ROOT / "tools" / "make_synthetic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass ищет модуль в sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_end2end_synthetic(tmp_path):
    t0 = time.perf_counter()
    sites = tmp_path / "sites"
    rc = _make_synthetic().main(
        [
            "--config",
            str(CONFIG),
            "--n-sites",
            "2",
            "--size",
            "512",
            "--seed",
            "0",
            "--out-dir",
            str(sites),
        ]
    )
    assert rc in (0, None)
    assert (sites / "site_1" / "layers").is_dir() and (sites / "site_2" / "labels.geojson").exists()

    cfg = load_pipeline_config(CONFIG)
    cfg = replace(
        cfg,
        data=replace(cfg.data, sites_dir=sites),
        model=replace(cfg.model, arch="stub"),
        train=replace(
            cfg.train,
            epochs=E2E_EPOCHS,
            tiles_per_epoch=E2E_TILES_PER_EPOCH,
            num_workers=0,
            device="cpu",
            train_sites=("site_1",),
            val_sites=("site_2",),
        ),
    )
    run_dir = train_from_config(cfg, None, tmp_path / "run")
    assert (run_dir / "best.pt").exists()

    out = tmp_path / "pred.geojson"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "inference" / "entrypoint.py"),
            "--data",
            str(sites / "site_2"),
            "--config",
            str(CONFIG),
            "--weights",
            str(run_dir / "best.pt"),
            "--out",
            str(out),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=MAX_SECONDS,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection" and fc["features"]

    res = evaluate_files(out, sites / "site_2" / "labels.geojson", cfg.eval)["__all__"]
    elapsed = time.perf_counter() - t0
    print(f"\ne2e: {res.to_dict()} | {elapsed:.1f} s")
    log = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    best = max(log, key=lambda r: r["val_f1"])  # первая эпоха с максимумом = best.pt
    assert (res.tp, res.fp, res.fn) == (best["val_tp"], best["val_fp"], best["val_fn"])
    assert res.f1 > MIN_F1, res.to_dict()
    assert elapsed < MAX_SECONDS
