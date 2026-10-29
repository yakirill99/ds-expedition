"""Обучение в смеси синтетики и реальных участков (фаза B2).

uv run python scripts/train_mix.py --config configs/dataset_real.yaml \\
    --train-site data/synthetic/site_1 --train-site data/raw/006_Осечки_1.1км \\
    --val-site data/synthetic/site_3
"""

import sys

from expds.train_mix import main

if __name__ == "__main__":
    sys.exit(main())
