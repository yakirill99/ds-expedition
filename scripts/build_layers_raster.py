"""CLI: сборка растровых каналов rgb_* для участков без лидара (D4.3).

uv run python scripts/build_layers_raster.py                    # все
uv run python scripts/build_layers_raster.py data/raw/082_Артпозиции_Иссык_Куль
uv run python scripts/build_layers_raster.py --force            # пересобрать
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from expds.data.build_raster import RGB_PREFIXES, build_site_rasters
from expds.pipeline_config import load_pipeline_config

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Растровые каналы (rgb_*) из исходников организаторов"
    )
    parser.add_argument(
        "site", type=Path, nargs="?", help="Один участок (иначе весь data.sites_dir)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_real.yaml"))
    parser.add_argument("--max-pixels", type=int, default=12_000_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_pipeline_config(args.config)

    root = cfg.data.sites_dir
    sites = [args.site] if args.site else sorted(p for p in root.iterdir() if p.is_dir())
    built = skipped = failed = 0
    t_start = time.perf_counter()

    for site in sites:
        if not any(p.is_dir() and p.name[:3] in RGB_PREFIXES for p in site.iterdir()):
            print(f"{site.name:34} пропуск: нет 03_/04_/05_", flush=True)
            continue

        print(f"\n=== {site.name}", flush=True)
        t0 = time.perf_counter()
        try:
            res = build_site_rasters(
                site,
                cfg.data.labels_crs,
                max_pixels=args.max_pixels,
                force=args.force,
                log=lambda s: print(s, flush=True),
            )
        except Exception as exc:  # noqa: BLE001 — CLI-граница
            print(f"{site.name:34} ОШИБКА {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            continue
        took = time.perf_counter() - t0

        if res["built"]:
            g = res["grid"]
            print(
                f"{site.name:34} собрано {g.width}x{g.height} {g.crs} "
                f"покрытие {res['coverage']:.1%} [{took:.1f} c]",
                flush=True,
            )
            built += 1
        else:
            print(f"{site.name:34} пропуск: {res['reason']} [{took:.1f} c]", flush=True)
            skipped += 1

    print(
        f"\nсобрано {built} | пропущено {skipped} | ошибок {failed} "
        f"| всего {time.perf_counter() - t_start:.1f} c"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
