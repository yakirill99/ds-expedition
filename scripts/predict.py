"""Инференс из репозитория: CLI как у inference/entrypoint.py, --config/--weights обязательны.

uv run python scripts/predict.py --data data/synthetic/site_3 \
    --config configs/dataset_synthetic.yaml --weights runs/<run>/best.pt --out pred.geojson
"""

import sys

from expds.predict import main

if __name__ == "__main__":
    sys.exit(main())
