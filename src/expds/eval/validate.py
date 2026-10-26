"""Валидатор формата сабмита платформы: структура GeoJSON, геометрии, свойства.

Собирает ВСЕ проблемы сразу (не fail-fast), каждая — самодостаточное сообщение
с номером feature. Проверки:
- корень: type == FeatureCollection, features — список;
- feature: словарь; geometry не None; тип Point | MultiPoint | Polygon | MultiPolygon
  (Multi* разрешены — разворачиваются на стороне метрики);
- точка: 2-3 числа; кольцо: вершины по 2-3 числа, >= 4 вершин, замкнуто;
- properties: есть ключ класса; score, если есть, — число в [0, 1];
- член crs, если есть, — EPSG, совпадающий с ожидаемым (отсутствие не ошибка:
  RFC 7946 не требует его; регламентный EPSG:3857 проставляет write_detections).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pyproj import CRS

_GEOMS = ("Point", "MultiPoint", "Polygon", "MultiPolygon")
_MAX_REPORT = 50
_NUMBER = (int, float)


def _epsg_of(value: str) -> int | None:
    """EPSG-код из строки CRS или None."""
    try:
        return CRS.from_user_input(value).to_epsg()
    except Exception:  # noqa: BLE001 — любую строку CRS трактуем как «не распознано»
        return None


def _declared_epsg(fc: dict[str, Any]) -> int | None:
    """EPSG из legacy-члена crs, если он есть и распознан."""
    member = fc.get("crs")
    if not isinstance(member, dict):
        return None
    name = str((member.get("properties") or {}).get("name") or "")
    return _epsg_of(name) if name else None


def _is_number(v: Any) -> bool:
    return isinstance(v, _NUMBER) and not isinstance(v, bool)


def validate_submission(
    path: str | Path, crs: str = "EPSG:3857", class_key: str = "class"
) -> list[str]:
    """Проверяет файл формата платформы. Пустой список = валидно."""
    try:
        fc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [f"файл не читается: {e}"]
    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        return ["корень: ожидался FeatureCollection"]
    features = fc.get("features")
    if not isinstance(features, list):
        return ["корень: features должен быть списком"]

    problems: list[str] = []
    expected, declared = _epsg_of(crs), _declared_epsg(fc)
    if expected is not None and declared is not None and declared != expected:
        problems.append(f"crs: объявлен EPSG:{declared}, ожидался EPSG:{expected}")

    for i, feat in enumerate(features):
        _validate_feature(i, feat, class_key, problems)
        if len(problems) >= _MAX_REPORT:
            problems.append(f"... дальше обрезано (лимит {_MAX_REPORT} сообщений)")
            break
    return problems


def _validate_feature(i: int, feat: Any, class_key: str, problems: list[str]) -> None:
    """Добавляет проблемы одной feature в список."""
    where = f"feature {i}"
    if not isinstance(feat, dict):
        problems.append(f"{where}: не объект")
        return
    props = feat.get("properties")
    geom = feat.get("geometry")
    if geom is not None and not isinstance(geom, dict):
        problems.append(f"{where}: geometry не объект")
        return
    if isinstance(props, dict):
        if class_key not in props:
            problems.append(f"{where}: нет свойства '{class_key}'")
        score = props.get("score")
        if score is not None and (not _is_number(score) or not 0.0 <= float(score) <= 1.0):
            problems.append(f"{where}: score {score!r} — не число в [0, 1]")
    if geom is None:
        return  # null-геометрия: read_labels предупреждает и пропускает

    gtype = geom.get("type")
    if gtype not in _GEOMS:
        problems.append(f"{where}: тип геометрии {gtype!r} не из {_GEOMS}")
        return
    coords = geom.get("coordinates")
    if gtype == "Point":
        _validate_point(coords, f"{where} ({gtype})", problems)
    elif gtype == "MultiPoint":
        if not isinstance(coords, list):
            problems.append(f"{where}: coordinates MultiPoint — список точек")
            return
        for c in coords:
            _validate_point(c, f"{where} ({gtype})", problems)
    elif gtype == "Polygon":
        _validate_polygon(coords, f"{where} ({gtype})", problems)
    else:  # MultiPolygon
        if not isinstance(coords, list):
            problems.append(f"{where}: coordinates MultiPolygon — список полигонов")
            return
        for poly in coords:
            _validate_polygon(poly, f"{where} ({gtype})", problems)


def _validate_point(coords: Any, where: str, problems: list[str]) -> None:
    """Точка: 2-3 числа, без NaN."""
    if (
        not isinstance(coords, (list, tuple))
        or len(coords) not in (2, 3)
        or not all(_is_number(c) for c in coords)
    ):
        problems.append(f"{where}: координаты точки — 2-3 числа")
        return
    if any(isinstance(c, float) and math.isnan(c) for c in coords):
        problems.append(f"{where}: NaN в координатах")


def _validate_polygon(coords: Any, where: str, problems: list[str]) -> None:
    """Полигоны: непустой список колец, кольцо >= 4 вершин, замкнуто, вершины 2-3 числа."""
    if not isinstance(coords, list) or not coords:
        problems.append(f"{where}: coordinates полигона — непустой список колец")
        return
    for r, ring in enumerate(coords):
        rwhere = f"{where}, кольцо {r}"
        if not isinstance(ring, list) or len(ring) < 4:
            problems.append(f"{rwhere}: список из >= 4 вершин")
            continue
        if (
            not isinstance(ring[0], (list, tuple))
            or not isinstance(ring[-1], (list, tuple))
            or ring[0][0] != ring[-1][0]
            or ring[0][1] != ring[-1][1]
        ):
            problems.append(f"{rwhere}: кольцо не замкнуто (первая != последняя)")
        for v in ring:
            if (
                not isinstance(v, (list, tuple))
                or len(v) not in (2, 3)
                or not all(_is_number(c) for c in v)
            ):
                problems.append(f"{rwhere}: вершина не 2-3 числа")
                break
