"""Отчёт метрики по классам: pred.geojson против gt.geojson.

Считает TP/FP/FN, precision, recall, F1 — отдельно по классам и суммарно.
Параметры метрики берутся из секции `eval` конфига датасета, их можно
переопределить флагами. Вывод — таблица в stdout и опционально JSON.

    uv run python scripts/evaluate.py \
        --pred runs/<run>/pred.geojson \
        --gt data/raw/site_3/labels.geojson \
        --config configs/dataset_real.yaml \
        --json report.json

Без --config используются дефолты EvalConfig. CRS предсказаний и разметки
должны совпадать (по регламенту EPSG:3857) — метрика не перепроецирует.

Дополнительно:
    --validate   — сначала прогнать валидатор формата платформы по --pred;
                   при проблемах напечатать их и выйти с кодом 2;
    --errors F   — сохранить FP/FN с типами (duplicate, localization, ...)
                   в GeoJSON F для QGIS.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from expds.eval.errors import analyze_errors, errors_to_geojson, format_error_report
from expds.eval.metric import ALL, EvalConfig, MatchResult, evaluate, read_objects
from expds.eval.validate import validate_submission

logger = logging.getLogger(__name__)

#: Порядок колонок отчёта.
COLUMNS = ("tp", "fp", "fn", "precision", "recall", "f1")

#: Код выхода при провале --validate.
EXIT_INVALID = 2


def eval_config_from_yaml(path: Path | None) -> EvalConfig:
    """Читает секцию `eval` конфига датасета.

    Args:
        path: путь к YAML или None (тогда дефолты).

    Returns:
        EvalConfig.
    """
    if path is None:
        return EvalConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EvalConfig.from_dict(raw.get("eval"))


def format_report(results: dict[str, MatchResult]) -> str:
    """Таблица метрик: сначала классы по алфавиту, в конце итог.

    Args:
        results: выход evaluate / evaluate_files.

    Returns:
        Многострочная таблица для stdout.
    """
    classes = sorted(k for k in results if k != ALL)
    rows = [(c, results[c]) for c in classes]
    if ALL in results:
        rows.append(("ВСЕГО", results[ALL]))

    width = max([len("класс"), *(len(name) for name, _ in rows)]) if rows else 5
    head = f"{'класс':<{width}}  {'tp':>4} {'fp':>4} {'fn':>4}  {'prec':>6} {'rec':>6} {'f1':>6}"
    lines = [head, "-" * len(head)]
    for name, r in rows:
        if name == "ВСЕГО":
            lines.append("-" * len(head))
        lines.append(
            f"{name:<{width}}  {r.tp:>4} {r.fp:>4} {r.fn:>4}  "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f}"
        )
    return "\n".join(lines)


def report_to_dict(results: dict[str, MatchResult], cfg: EvalConfig) -> dict[str, Any]:
    """Отчёт в JSON-совместимый словарь (метрики + параметры метрики)."""
    return {
        "eval_config": {
            "iou_thr": cfg.iou_thr,
            "r_tol": cfg.r_tol,
            "r_tol_units": cfg.r_tol_units,
            "iou_samples": cfg.iou_samples,
            "class_agnostic": cfg.class_agnostic,
        },
        "total": results[ALL].to_dict() if ALL in results else {},
        "per_class": {k: v.to_dict() for k, v in sorted(results.items()) if k != ALL},
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: считает метрику и печатает отчёт. Возвращает код выхода."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True, help="GeoJSON предсказаний")
    ap.add_argument("--gt", type=Path, required=True, help="GeoJSON разметки")
    ap.add_argument("--config", type=Path, help="YAML датасета (секция eval)")
    ap.add_argument("--crs", default="EPSG:3857", help="CRS обоих файлов")
    ap.add_argument("--class-key", default="class", help="ключ класса в properties")
    ap.add_argument("--iou-thr", type=float, help="переопределить eval.iou_thr")
    ap.add_argument("--r-tol", type=float, help="переопределить eval.r_tol")
    ap.add_argument("--r-tol-units", choices=["crs", "meters"], help="единицы r_tol")
    ap.add_argument("--class-agnostic", action="store_true", help="игнорировать классы")
    ap.add_argument("--json", type=Path, help="куда сохранить отчёт в JSON")
    ap.add_argument(
        "--validate", action="store_true", help=f"валидатор формата; провал -> выход {EXIT_INVALID}"
    )
    ap.add_argument("--errors", type=Path, help="FP/FN с типами в GeoJSON (для QGIS)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for path in (args.pred, args.gt):
        if not path.is_file():
            ap.error(f"файл не найден: {path}")

    if args.validate:
        problems = validate_submission(args.pred, args.crs, args.class_key)
        if problems:
            print("Формат платформы не пройден:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return EXIT_INVALID
        logger.info("формат платформы: OK (%s)", args.pred)

    cfg = eval_config_from_yaml(args.config)
    overrides = {
        k: v
        for k, v in {
            "iou_thr": args.iou_thr,
            "r_tol": args.r_tol,
            "r_tol_units": args.r_tol_units,
        }.items()
        if v is not None
    }
    if args.class_agnostic:
        overrides["class_agnostic"] = True
    if overrides:
        # replace минует from_dict, поэтому валидируем явно
        cfg = EvalConfig.from_dict({**asdict(cfg), **overrides})

    preds = read_objects(args.pred, args.crs, args.class_key)
    gts = read_objects(args.gt, args.crs, args.class_key)
    results = evaluate(preds, gts, cfg)
    print(format_report(results))

    if args.errors:
        report = analyze_errors(preds, gts, results[ALL], cfg)
        n = errors_to_geojson(args.errors, preds, gts, report, args.crs, args.class_key)
        logger.info("FP/FN (%d объектов) -> %s", n, args.errors)
        print(format_error_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report_to_dict(results, cfg), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("отчёт сохранён: %s", args.json)
    return 0
