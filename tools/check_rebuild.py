"""Сравнение участка до и после пересборки рельефа.

Проверяет три вещи:
  1. покрытие DTM выросло, измеренные пиксели не тронуты;
  2. у каждого GT-объекта выросла доля пикселей с реальными данными
     (а не интерполяцией) — это и есть честная оценка пригодности участка;
  3. все производные имеют ту же маску, что dtm, а облачные признаки
     побитово совпадают (их пересборка не должна была менять).

Разметка читается из ИСХОДНОЙ папки участка: rebuild копирует только
layers/ и samples/, labels.geojson остаётся на месте.

Запуск:
    python3 tools/check_rebuild.py --before data/raw/010_Эсбю_укрепление \
        --after data/rebuilt/010_Эсбю_укрепление
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from expds.data.raster import read_raster
from expds.io.geojson import GEOM_POLYGON, read_labels
from expds.pipeline_config import load_pipeline_config
from expds.targets.rasterize import polygons_to_mask

_LAYERS = "layers"
_RELIEF_PREFIXES = (
    "slope",
    "aspect_sin",
    "aspect_cos",
    "hillshade_az_",
    "slrm_sigma_",
    "positive_openness",
    "negative_openness",
    "sky_view_factor",
    "curvature_sigma_",
    "tpi_radius_",
    "tri_radius_",
)


def _is_relief(name: str) -> bool:
    """Имя слоя относится к производным рельефа."""
    return name.startswith(_RELIEF_PREFIXES)


def _load_layers(site: Path) -> dict[str, object]:
    """Все слои участка, прочитанные в его сетке."""
    out: dict[str, object] = {}
    for tif in sorted((site / _LAYERS).glob("*.tif")):
        try:
            out[tif.stem] = read_raster(tif)
        except Exception as exc:  # noqa: BLE001 — битый слой не должен ронять отчёт
            print(f"  ВНИМАНИЕ: {tif.name} не читается: {type(exc).__name__}")
    return out


def _object_coverage(
    labels_path: Path,
    grid,
    classes: list[str],
    measured: np.ndarray,
    labels_crs: str,
) -> list[tuple[str, int, float]]:
    """(класс, площадь px, доля измеренных) по каждому полигональному объекту."""
    feats = [
        f
        for f in read_labels(labels_path, grid, src_crs=labels_crs)
        if f.geom_type == GEOM_POLYGON and f.pixel and f.cls in classes
    ]
    rows: list[tuple[str, int, float]] = []
    for f in feats:
        mask = polygons_to_mask([f], grid, classes)[classes.index(f.cls)] > 0
        area = int(mask.sum())
        if area == 0:
            continue
        rows.append((f.cls, area, float((mask & measured).sum()) / area))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--before", type=Path, required=True, help="исходная папка участка")
    p.add_argument("--after", type=Path, required=True, help="папка после пересборки")
    p.add_argument("--config", type=Path, default=Path("configs/dataset_real.yaml"))
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="labels.geojson (по умолчанию из --before)",
    )
    args = p.parse_args()

    cfg = load_pipeline_config(args.config)
    classes = list(cfg.targets.classes)
    labels_path = args.labels or (args.before / cfg.data.labels_file)

    before = _load_layers(args.before)
    after = _load_layers(args.after)
    if "dtm" not in before or "dtm" not in after:
        print("ОШИБКА: в одной из папок нет dtm.tif")
        return 2

    dtm_b, dtm_a = before["dtm"], after["dtm"]
    m_b, m_a = dtm_b.valid_mask, dtm_a.valid_mask
    if m_b.shape != m_a.shape:
        print(f"ОШИБКА: разные сетки {m_b.shape} vs {m_a.shape}")
        return 2

    recovered = m_a & ~m_b
    lost = m_b & ~m_a
    still = ~m_a

    print(f"\n=== {args.before.name} ===")
    print(
        f"DTM: measured {m_b.mean():.1%} -> {m_a.mean():.1%}  "
        f"(+{recovered.mean():.1%} заполнено, nodata {still.mean():.1%})"
    )
    if lost.any():
        print(f"  ВНИМАНИЕ: потеряно {int(lost.sum())} измеренных пикселей — этого быть не должно")
    untouched = np.array_equal(dtm_a.data[m_b], dtm_b.data[m_b])
    print(f"  измеренные пиксели не тронуты: {untouched}")

    # --- согласованность масок ------------------------------------------------
    relief = [n for n in after if _is_relief(n)]
    bad_mask = [n for n in relief if not np.array_equal(after[n].valid_mask, m_a)]
    print(f"\nпроизводные: {len(relief)} слоёв, маска != маске dtm: {len(bad_mask)}")
    for n in bad_mask[:10]:
        print(f"  {n}: valid {after[n].valid_mask.mean():.1%} vs dtm {m_a.mean():.1%}")

    # Пиксели, где dtm заполнен, но производная осталась нулём — это тот самый
    # рассинхрон, из-за которого лосс считает медиану настоящими данными.
    zeros = {n: int((m_a & (after[n].data == 0.0)).sum()) for n in relief}
    suspicious = {n: c for n, c in zeros.items() if c > 0}
    if suspicious:
        print("  valid=True и значение ровно 0.0 (проверить, не артефакт ли):")
        for n, c in sorted(suspicious.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {n:24} {c:6d} px = {c / max(int(m_a.sum()), 1):.1%} valid")

    # --- облачные признаки не должны меняться --------------------------------
    unchanged, changed = [], []
    for name in after:
        if _is_relief(name) or name == "dtm" or name not in before:
            continue
        same = np.array_equal(
            after[name].data.astype(np.float64),
            before[name].data.astype(np.float64),
            equal_nan=True,
        )
        (unchanged if same else changed).append(name)
    print(f"\nне-рельефные слои: {len(unchanged)} без изменений, {len(changed)} изменены")
    if changed:
        print(f"  изменены: {changed[:8]}{' ...' if len(changed) > 8 else ''}")

    # --- покрытие объектов ----------------------------------------------------
    if not labels_path.exists():
        print(f"\nразметки нет ({labels_path}) — сравнение по объектам пропущено")
        return 0

    rows_b = _object_coverage(labels_path, dtm_b.grid, classes, m_b, cfg.data.labels_crs)
    rows_a = _object_coverage(labels_path, dtm_a.grid, classes, m_a, cfg.data.labels_crs)
    if len(rows_b) != len(rows_a):
        print(f"\nВНИМАНИЕ: объектов {len(rows_b)} до и {len(rows_a)} после — сравниваю по парам")
    n = min(len(rows_b), len(rows_a))
    if n == 0:
        print("\nполигональных объектов нет")
        return 0

    cov_b = np.array([r[2] for r in rows_b[:n]])
    cov_a = np.array([r[2] for r in rows_a[:n]])
    usable_b = int((cov_b >= 0.9).sum())
    usable_a = int((cov_a >= 0.9).sum())
    changed_objs = int((np.abs(cov_a - cov_b) > 0.01).sum())

    print(f"\nобъекты ({n}):")
    print(f"  среднее покрытие      {cov_b.mean():.1%} -> {cov_a.mean():.1%}")
    print(f"  >=90% измерено        {usable_b} -> {usable_a} из {n}")
    print(f"  >70% измерено         {int((cov_b >= 0.7).sum())} -> {int((cov_a >= 0.7).sum())}")
    print(f"  покрытие изменилось у {changed_objs} объектов")

    print(f"\n{'#':>3} {'класс':22} {'px':>7} {'было':>7} {'стало':>7} {'дельта':>8}")
    print("-" * 60)
    for i in range(n):
        cls, area, cb = rows_b[i]
        ca = rows_a[i][2]
        flag = "" if abs(ca - cb) < 0.01 else ("  <-- вылечен" if ca > cb else "  <-- ухудшился")
        print(f"{i:3d} {cls:22} {area:7d} {cb:6.1%} {ca:6.1%} {ca - cb:+8.1%}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
