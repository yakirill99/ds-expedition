"""Точка входа платформы: участки со слоями -> GeoJSON (EPSG:3857). Без сети.

python inference/entrypoint.py --data <участок|папка участков> --out pred.geojson \
    [--config inference/config.yaml] [--weights inference/weights.pt] [--device auto]

expds ищется: установленный пакет -> inference/expds (архив сабмита) -> ../src (репозиторий).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _ensure_expds() -> None:
    try:
        import expds  # noqa: F401
    except ImportError:
        for cand in (_HERE, _HERE.parent / "src"):
            if (cand / "expds").is_dir():
                sys.path.insert(0, str(cand))
                return
        raise


def main(argv: list[str] | None = None) -> int:
    _ensure_expds()
    from expds.predict import main as predict_main

    return predict_main(
        argv,
        default_config=_HERE / "config.yaml",
        default_weights=_HERE / "weights.pt",
    )


if __name__ == "__main__":
    sys.exit(main())
