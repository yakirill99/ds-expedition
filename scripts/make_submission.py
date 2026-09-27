"""Архив сабмита + smoke из распакованного архива без сети.

uv run python scripts/make_submission.py --config configs/dataset_synthetic.yaml \
    --weights runs/<run>/best.pt --out dist/submission.zip --smoke-data data/synthetic/site_3
"""

import sys

from expds.submission import main

if __name__ == "__main__":
    sys.exit(main())
