"""Тесты CLI отчёта метрики и регламентных случаев матчинга.

Регламент конкурса: метрика считается в EPSG:3857, один GeoJSON на всё,
Polygon матчится по IoU, Point — по расстоянию до центроида в пределах r_tol.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from expds.eval.metric import ALL, EvalConfig, MatchResult, evaluate_files
from expds.evaluate import eval_config_from_yaml, format_report, main, report_to_dict

X0, Y0 = 4_200_000.0, 7_500_000.0  # порядок величин EPSG:3857


def _ring(x, y, s):
    """Замкнутое кольцо квадрата со стороной s (как в файле GeoJSON)."""
    return [
        [
            [X0 + x, Y0 + y],
            [X0 + x + s, Y0 + y],
            [X0 + x + s, Y0 + y + s],
            [X0 + x, Y0 + y + s],
            [X0 + x, Y0 + y],
        ]
    ]


def _feat(cls, geometry, geom_type="Polygon", score=1.0):
    return {
        "type": "Feature",
        "properties": {"class": cls, "score": score},
        "geometry": {"type": geom_type, "coordinates": geometry},
    }


def _write(path, feats):
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": feats,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


# --- конфиг метрики из YAML ---------------------------------------------------


def test_eval_config_from_yaml_reads_section(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text(
        "data:\n  sites_dir: x\neval:\n  iou_thr: 0.4\n  r_tol: 15.0\n  r_tol_units: meters\n",
        encoding="utf-8",
    )
    cfg = eval_config_from_yaml(p)
    assert cfg.iou_thr == pytest.approx(0.4)
    assert cfg.r_tol == pytest.approx(15.0)
    assert cfg.r_tol_units == "meters"


def test_eval_config_from_yaml_defaults_without_section(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text("data:\n  sites_dir: x\n", encoding="utf-8")
    assert eval_config_from_yaml(p) == EvalConfig()


def test_eval_config_from_yaml_none_is_defaults():
    assert eval_config_from_yaml(None) == EvalConfig()


def test_eval_config_from_yaml_rejects_unknown_key(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text("eval:\n  iou_thr: 0.5\n  typo_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        eval_config_from_yaml(p)


# --- формат отчёта ------------------------------------------------------------


def test_format_report_has_classes_and_total():
    res = {
        ALL: MatchResult(tp=3, fp=1, fn=2),
        "mound": MatchResult(tp=2, fp=1, fn=0),
        "ditch": MatchResult(tp=1, fp=0, fn=2),
    }
    out = format_report(res)
    assert "mound" in out and "ditch" in out and "ВСЕГО" in out
    # классы по алфавиту: ditch раньше mound
    assert out.index("ditch") < out.index("mound") < out.index("ВСЕГО")


def test_format_report_empty_is_safe():
    assert format_report({}) == format_report({})  # не падает


def test_report_to_dict_structure():
    res = {ALL: MatchResult(tp=1, fp=0, fn=0), "mound": MatchResult(tp=1, fp=0, fn=0)}
    d = report_to_dict(res, EvalConfig(iou_thr=0.4))
    assert d["eval_config"]["iou_thr"] == pytest.approx(0.4)
    assert d["total"]["f1"] == 1.0
    assert d["per_class"]["mound"]["tp"] == 1
    assert ALL not in d["per_class"]
    json.dumps(d)  # должен быть сериализуем


# --- CLI ----------------------------------------------------------------------


def test_cli_prints_and_saves_json(tmp_path, capsys):
    gt = _write(
        tmp_path / "gt.geojson",
        [_feat("mound", _ring(0, 0, 20)), _feat("ditch", _ring(200, 0, 30))],
    )
    pred = _write(
        tmp_path / "pred.geojson",
        [
            _feat("mound", _ring(1, 0, 20), score=0.9),
            _feat("ditch", _ring(200, 0, 30), score=0.8),
        ],
    )
    out_json = tmp_path / "rep.json"
    rc = main(["--pred", str(pred), "--gt", str(gt), "--json", str(out_json)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "ВСЕГО" in printed
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["total"]["tp"] == 2
    assert saved["total"]["f1"] == 1.0


def test_cli_missing_file_exits(tmp_path):
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", _ring(0, 0, 20))])
    with pytest.raises(SystemExit):
        main(["--pred", str(tmp_path / "нет.geojson"), "--gt", str(gt)])


def test_cli_overrides_iou_thr(tmp_path, capsys):
    """Сдвиг на 8 из 20: IoU ~0.41 — проходит при 0.4, не проходит при 0.5."""
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", _ring(0, 0, 20))])
    pred = _write(tmp_path / "pred.geojson", [_feat("mound", _ring(8, 0, 20))])

    main(["--pred", str(pred), "--gt", str(gt), "--iou-thr", "0.9"])
    assert " 0 " in capsys.readouterr().out  # tp = 0

    j = tmp_path / "r.json"
    main(["--pred", str(pred), "--gt", str(gt), "--iou-thr", "0.4", "--json", str(j)])
    assert json.loads(j.read_text(encoding="utf-8"))["total"]["tp"] == 1


def test_cli_class_agnostic(tmp_path):
    """class_agnostic сводит классы: предсказание другого класса становится TP."""
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", _ring(0, 0, 20))])
    pred = _write(tmp_path / "pred.geojson", [_feat("ditch", _ring(0, 0, 20))])

    strict = tmp_path / "s.json"
    main(["--pred", str(pred), "--gt", str(gt), "--json", str(strict)])
    assert json.loads(strict.read_text(encoding="utf-8"))["total"]["tp"] == 0

    loose = tmp_path / "l.json"
    main(["--pred", str(pred), "--gt", str(gt), "--class-agnostic", "--json", str(loose)])
    assert json.loads(loose.read_text(encoding="utf-8"))["total"]["tp"] == 1


def test_cli_uses_config_section(tmp_path):
    """Параметры метрики берутся из секции eval конфига датасета."""
    cfg = tmp_path / "ds.yaml"
    cfg.write_text("eval:\n  iou_thr: 0.4\n", encoding="utf-8")
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", _ring(0, 0, 20))])
    pred = _write(tmp_path / "pred.geojson", [_feat("mound", _ring(8, 0, 20))])
    j = tmp_path / "r.json"
    main(["--pred", str(pred), "--gt", str(gt), "--config", str(cfg), "--json", str(j)])
    saved = json.loads(j.read_text(encoding="utf-8"))
    assert saved["eval_config"]["iou_thr"] == pytest.approx(0.4)
    assert saved["total"]["tp"] == 1


# --- регламент: Point в GT против Polygon в предсказании ----------------------


def test_regulation_point_gt_matched_by_centroid(tmp_path):
    """GT-точка матчится с полигоном предсказания: центроид в пределах r_tol."""
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", [X0 + 50, Y0 + 50], "Point")])
    pred = _write(tmp_path / "pred.geojson", [_feat("mound", _ring(48, 48, 4))])
    j = tmp_path / "r.json"
    main(["--pred", str(pred), "--gt", str(gt), "--r-tol", "5", "--json", str(j)])
    assert json.loads(j.read_text(encoding="utf-8"))["total"]["tp"] == 1


def test_regulation_mixed_geometry_single_file(tmp_path):
    """Один GeoJSON содержит и Point, и Polygon — так подаётся сабмит."""
    gt = _write(
        tmp_path / "gt.geojson",
        [
            _feat("mound", [X0 + 50, Y0 + 50], "Point"),
            _feat("rampart", _ring(0, 0, 20)),
        ],
    )
    pred = _write(
        tmp_path / "pred.geojson",
        [
            _feat("mound", _ring(49, 49, 2), score=0.7),
            _feat("rampart", _ring(0, 0, 20), score=0.9),
        ],
    )
    res = evaluate_files(pred, gt, EvalConfig(r_tol=5.0))
    assert res[ALL].tp == 2
    assert res["mound"].tp == 1 and res["rampart"].tp == 1


def test_regulation_r_tol_units_meters_scales_with_latitude(tmp_path):
    """В EPSG:3857 метры на местности != единицы CRS; множитель cosh(y/R)."""
    lat = np.deg2rad(60.0)
    y60 = 6378137.0 * np.log(np.tan(np.pi / 4 + lat / 2))  # масштаб 3857 на 60° = 2
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", [X0, y60], "Point")])
    pred = _write(tmp_path / "pred.geojson", [_feat("mound", [X0 + 8, y60], "Point")])

    crs_units = tmp_path / "crs.json"
    main(["--pred", str(pred), "--gt", str(gt), "--r-tol", "5",
          "--r-tol-units", "crs", "--json", str(crs_units)])  # fmt: skip
    assert json.loads(crs_units.read_text(encoding="utf-8"))["total"]["tp"] == 0

    meters = tmp_path / "m.json"
    main(["--pred", str(pred), "--gt", str(gt), "--r-tol", "5",
          "--r-tol-units", "meters", "--json", str(meters)])  # fmt: skip
    assert json.loads(meters.read_text(encoding="utf-8"))["total"]["tp"] == 1


# --- валидатор формата ---------------------------------------------------------


def test_cli_validate_accepts_valid(tmp_path):
    """Валидный файл проходит --validate и метрика считается как обычно."""
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", _ring(0, 0, 20))])
    pred = _write(tmp_path / "pred.geojson", [_feat("mound", _ring(0, 0, 20))])
    assert main(["--pred", str(pred), "--gt", str(gt), "--validate"]) == 0


def test_cli_validate_rejects_bad(tmp_path, capsys):
    """Невалидный файл: --validate печатает проблемы в stderr и выходит с кодом 2."""
    gt = _write(tmp_path / "gt.geojson", [_feat("mound", _ring(0, 0, 20))])
    pred = tmp_path / "bad.geojson"
    pred.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0, 0], [1, 1]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--pred", str(pred), "--gt", str(gt), "--validate"]) == 2
    err = capsys.readouterr().err
    assert "не пройден" in err and "LineString" in err
