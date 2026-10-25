"""Отчёт метрики по классам (pred против gt).

uv run python scripts/evaluate.py --pred pred.geojson --gt labels.geojson \
    --config configs/dataset_real.yaml --json runs/report.json

Флаги --validate (валидатор формата платформы) и --errors (FP/FN в GeoJSON
для QGIS) описаны в expds.evaluate.
"""

import sys

from expds.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
