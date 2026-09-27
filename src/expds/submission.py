"""Сборка архива сабмита и smoke-проверка.

Архив: inference/{entrypoint.py, expds/ (копия пакета), config.yaml, weights.pt, MANIFEST.json,
requirements.txt}. Проверки: каналы/классы чекпойнта == YAML, размер <= лимита (2 ГБ),
smoke: распаковка во временную папку, запуск entrypoint отдельным процессом с заблокированной
сетью; expds должен импортироваться из архива, выход — валидный FeatureCollection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import torch

from expds.models.checkpoint import load_checkpoint
from expds.pipeline_config import load_pipeline_config

logger = logging.getLogger(__name__)

MAX_BYTES = 2 * 1024**3
PKG_DIR = Path(__file__).resolve().parent  # src/expds (namespace-пакет, без __init__)
REPO_ROOT = PKG_DIR.parents[1]

# Запуск entrypoint с заблокированной сетью: любой connect -> RuntimeError.
_OFFLINE_GUARD = """\
import pathlib, runpy, socket, sys
def _blocked(*a, **k):
    raise RuntimeError("сетевой вызов в офлайн-инференсе")
socket.socket.connect = _blocked
socket.create_connection = _blocked
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
"""


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def build_submission(
    config: Path,
    weights: Path,
    out_zip: Path,
    smoke_data: Path | None = None,
    device: str = "cpu",
    max_bytes: int = MAX_BYTES,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Собирает zip; при smoke_data прогоняет smoke. Сводка: zip, bytes, manifest, smoke."""
    config, weights, out_zip = Path(config), Path(weights), Path(out_zip)
    cfg = load_pipeline_config(config)
    _, payload = load_checkpoint(
        weights, channel_names=cfg.data.channels, classes=cfg.targets.classes
    )
    entry = Path(repo_root) / "inference" / "entrypoint.py"
    if not entry.exists():
        raise FileNotFoundError(entry)

    manifest: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": (_run(["git", "rev-parse", "--short", "HEAD"], repo_root) or "").strip() or None,
        "torch": torch.__version__,
        "config_source": str(config),
        "weights_source": str(weights),
        "weights_md5": _md5(weights),
        "model_cfg": payload["model_cfg"],
        "channels": list(cfg.data.channels),
        "classes": list(cfg.targets.classes),
        "labels_crs": cfg.data.labels_crs,
        "checkpoint_extra": payload.get("extra", {}),
    }
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "inference"
        stage.mkdir()
        shutil.copy2(entry, stage / "entrypoint.py")
        shutil.copytree(
            PKG_DIR, stage / "expds", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        init = stage / "expds" / "__init__.py"
        if not init.exists():
            # обычный пакет перекрывает namespace-части из других путей sys.path:
            # недостающий в архиве модуль не подтянется молча из src/
            init.write_text('"""expds (копия для сабмита)."""\n', encoding="utf-8")
        shutil.copy2(config, stage / "config.yaml")
        shutil.copy2(weights, stage / "weights.pt")
        req = _run(
            ["uv", "export", "--no-hashes", "--no-dev", "--no-emit-project", "--no-header"],
            repo_root,
        )
        if req:
            (stage / "requirements.txt").write_text(req, encoding="utf-8")
        else:
            logger.warning("uv export не сработал: requirements.txt не добавлен")
        (stage / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(tmp).as_posix())

    size = out_zip.stat().st_size
    if size > max_bytes:
        raise ValueError(f"архив {size / 1024**3:.2f} ГБ > лимита {max_bytes / 1024**3:.2f} ГБ")
    result: dict[str, Any] = {"zip": str(out_zip), "bytes": size, "manifest": manifest}
    if smoke_data is not None:
        result["smoke"] = smoke_test(out_zip, Path(smoke_data), device)
    return result


def smoke_test(
    zip_path: Path, data: Path, device: str = "cpu", timeout: int = 1800
) -> dict[str, Any]:
    """Распаковка архива во временную папку и запуск entrypoint без сети."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        entry = root / "inference" / "entrypoint.py"
        out = root / "pred.geojson"
        cmd = [
            sys.executable, "-c", _OFFLINE_GUARD, str(entry),
            "--data", str(Path(data).resolve()), "--out", str(out), "--device", device,
        ]  # fmt: skip
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, timeout=timeout, check=False
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"smoke: entrypoint упал (код {proc.returncode})\n{proc.stderr[-3000:]}"
            )
        vendored = str(root / "inference" / "expds")
        if f"expds: {vendored}" not in proc.stdout:
            raise RuntimeError(f"smoke: expds импортирован не из архива\n{proc.stdout[-1500:]}")
        fc = json.loads(out.read_text(encoding="utf-8"))
        if fc.get("type") != "FeatureCollection":
            raise RuntimeError("smoke: выход не FeatureCollection")
        return {"seconds": round(elapsed, 2), "n_features": len(fc.get("features", []))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сборка архива сабмита + smoke")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("dist/submission.zip"))
    ap.add_argument("--smoke-data", type=Path, help="участок или папка участков для smoke")
    ap.add_argument("--device", default="cpu", help="устройство smoke-запуска")
    ap.add_argument("--max-gb", type=float, default=MAX_BYTES / 1024**3)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = build_submission(
        args.config,
        args.weights,
        args.out,
        smoke_data=args.smoke_data,
        device=args.device,
        max_bytes=int(args.max_gb * 1024**3),
    )
    smoke = res.get("smoke")
    print(
        f"OK: {res['zip']} {res['bytes'] / 1024**2:.1f} МБ | git {res['manifest']['git']} | "
        f"weights md5 {res['manifest']['weights_md5'][:8]}"
        + (f" | smoke {smoke['seconds']} с, {smoke['n_features']} объектов" if smoke else ""),
        flush=True,
    )
    return 0
