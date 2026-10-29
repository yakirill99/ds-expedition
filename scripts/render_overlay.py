"""Картинка с найденными объектами поверх подложки.

uv run python scripts/render_overlay.py \\
    --base data/raw/006_Осечки_1.1км/layers/hillshade_az_315.tif \\
    --geojson data/raw/006_Осечки_1.1км/labels.geojson \\
    --out /tmp/006_gt.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from expds.render import SUBMISSION_CRS, render_overlay


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="GeoTIFF подложки")
    ap.add_argument("--geojson", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src-crs", default=SUBMISSION_CRS)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--max-side", type=int, default=4096)
    ap.add_argument("--no-legend", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    render_overlay(
        args.base,
        args.geojson,
        args.out,
        src_crs=args.src_crs,
        alpha=args.alpha,
        legend=not args.no_legend,
        max_side=args.max_side,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
