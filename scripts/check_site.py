#!/usr/bin/env python3
"""check_site: прогон участка через тренировочный путь (load_pipeline_config -> load_site).

Проверяет ровно то, что увидит train: сетку (CRS из UTM.json -> build_site),
каналы (сколько из контракта доступно), валидные пиксели и таргеты y_seg/y_hm
из labels.geojson (labels_crs из конфига). Ненулевой y_seg = разметка легла
в сетку в правильном кадре.

Использование:
    uv run python scripts/check_site.py --site "006_Осечки_1.1км"
    uv run python scripts/check_site.py --site "013_Нора_Виннон" --config configs/dataset_real.yaml

Коды выхода: 0 — OK (разметка в сетке), 1 — проблемы (смотри «!!» строки и ИТОГ).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from expds.pipeline_config import load_pipeline_config
from expds.tiles.dataset import load_site

logger = logging.getLogger(__name__)


def _peek_labels(path: Path) -> tuple[int, list[str], list[str]]:
    """Сырой подсчёт фич разметки (до растеризации): (n, классы, типы геометрий)."""
    gj = json.loads(path.read_text(encoding="utf-8"))
    feats = gj.get("features", [])
    classes = [str((f.get("properties") or {}).get("class", "?")) for f in feats]
    geom_types = sorted({str((f.get("geometry") or {}).get("type", "?")) for f in feats})
    return len(feats), classes, geom_types


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка участка тренировочным путём")
    parser.add_argument("--site", required=True, help="Имя папки участка в data.sites_dir")
    parser.add_argument(
        "--config", default="configs/dataset_real.yaml", help="Pipeline-YAML (секции data/targets)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # rasterio-читалка пишет INFO на каждый слой — глушим до WARNING.
    logging.getLogger("expds.data.raster").setLevel(logging.WARNING)

    cfg = load_pipeline_config(Path(args.config))
    site_dir = cfg.data.sites_dir / args.site
    print(f"Конфиг: {args.config} | sites_dir: {cfg.data.sites_dir}")
    print(f"Участок: {site_dir}")
    print(f"labels_crs: {cfg.data.labels_crs} | normalize: {cfg.data.normalize}")

    if not site_dir.is_dir():
        print("!! Папка участка не найдена")
        return 1

    problems: list[str] = []

    labels_path = site_dir / cfg.data.labels_file
    if labels_path.exists():
        try:
            n, classes, geoms = _peek_labels(labels_path)
            print(
                f"Разметка: {labels_path.name} | фич: {n} | классы: {classes} | геометрии: {geoms}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"!! Не удалось прочитать разметку: {exc}")
            problems.append("labels.geojson не читается")
    else:
        print(f"!! Разметки нет: {labels_path}")
        problems.append("нет labels.geojson")

    site = load_site(
        site_dir=site_dir,
        channels=cfg.data.channels,
        classes=cfg.targets.classes,
        heatmap_sigma_px=cfg.targets.heatmap_sigma_px,
        labels_crs=cfg.data.labels_crs,
        labels_file=cfg.data.labels_file,
        normalize=cfg.data.normalize,
    )

    g = site.grid
    x_max = g.x_min + g.width * g.pixel_size
    y_max = g.y_min + g.height * g.pixel_size
    print(f"Сетка: CRS {g.crs} | {g.width}x{g.height} | px {g.pixel_size} м")
    print(f"Границы: X {g.x_min:.1f}..{x_max:.1f} | Y {g.y_min:.1f}..{y_max:.1f}")

    n_total = len(site.channel_names)
    n_avail = int(site.channel_mask.sum())
    missing = [n for n, ok in zip(site.channel_names, site.channel_mask, strict=True) if not ok]
    print(f"Каналы: доступно {n_avail}/{n_total}")
    if missing:
        print(f"  занулены ({len(missing)}): {', '.join(missing)}")

    n_valid = int(site.valid.sum())
    pct = 100.0 * n_valid / site.valid.size
    print(f"Валидные пиксели: {n_valid}/{site.valid.size} ({pct:.1f}%)")

    if site.y_seg is None:
        print("!! Таргетов нет — load_site не собрал y_seg/y_hm")
        return 1

    per_class = (site.y_seg > 0).sum(axis=(1, 2))
    for cls, npx in zip(cfg.targets.classes, per_class, strict=True):
        print(f"  y_seg '{cls}': {int(npx)} px")
    hm_max = float(site.y_hm.max())
    print(f"  y_hm max: {hm_max:.4f} (0 при Polygon-разметке — известная особенность)")

    if int(per_class.sum()) == 0:
        print("!! y_seg пустой во всех классах — разметка не попала в сетку (CRS?)")
        problems.append("y_seg пустой")
    if n_avail == 0:
        problems.append("нет ни одного канала")
    if n_valid == 0:
        problems.append("нет валидных пикселей")

    if problems:
        print("ИТОГ: ПРОБЛЕМЫ — " + "; ".join(problems))
        return 1
    print("ИТОГ: OK — разметка в сетке, каналы на месте")
    return 0


if __name__ == "__main__":
    sys.exit(main())
