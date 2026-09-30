"""GeoJSON: чтение разметки (GT) и запись детекций.

Внутри пайплайна — crs.internal (Grid.crs) и пиксели Grid (row 0 = юг).
На входе и выходе — crs.output (EPSG:3857 по регламенту). Перепроекция — только здесь.

Поддержка: Point, MultiPoint, Polygon (с дырами), MultiPolygon. Multi* разворачиваются
в отдельные объекты. Кольца хранятся без замыкающей точки.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS, Transformer

from expds.features.grid import Grid

logger = logging.getLogger(__name__)

GEOM_POINT = "Point"
GEOM_POLYGON = "Polygon"

_DEFAULT_CLASS_KEY = "class"
_SCORE_KEY = "score"
_AREA_EPS = 1e-12


# ------------------------------------------------------------
# Геометрия
# ------------------------------------------------------------
def ring_centroid(ring: np.ndarray) -> np.ndarray:
    """Центр масс многоугольника (формула шнурования).

    Координаты сдвигаются к первой вершине — иначе в EPSG:3857 (~1e7 м)
    теряется точность. Вырожденное кольцо -> среднее вершин.

    Args:
        ring: (N, 2) вершины без замыкающей точки.

    Returns:
        (2,) координаты центра масс.
    """
    ring = np.asarray(ring, dtype=np.float64)
    if len(ring) < 3:
        return ring.mean(axis=0)
    origin = ring[0]
    x, y = ring[:, 0] - origin[0], ring[:, 1] - origin[1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    if abs(area) < _AREA_EPS:
        return ring.mean(axis=0)
    cx = ((x + xn) * cross).sum() / (6.0 * area)
    cy = ((y + yn) * cross).sum() / (6.0 * area)
    return np.array([cx + origin[0], cy + origin[1]])


def _strip_closing(ring: np.ndarray) -> np.ndarray:
    """Убирает замыкающую точку, если она есть."""
    if len(ring) > 1 and np.array_equal(ring[0], ring[-1]):
        return ring[:-1]
    return ring


def _as_ring(coords: Any) -> np.ndarray:
    """Координаты GeoJSON -> (N, 2) float64 (Z отбрасывается)."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :2]


def _explode(geom: dict[str, Any]) -> list[tuple[str, list[np.ndarray]]]:
    """Геометрия GeoJSON -> список (тип, кольца). Multi* разворачиваются."""
    gtype, coords = geom.get("type"), geom.get("coordinates")
    if gtype == GEOM_POINT:
        return [(GEOM_POINT, [_as_ring(coords)])]
    if gtype == "MultiPoint":
        return [(GEOM_POINT, [_as_ring(c)]) for c in coords]
    if gtype == GEOM_POLYGON:
        return [(GEOM_POLYGON, [_strip_closing(_as_ring(r)) for r in coords])]
    if gtype == "MultiPolygon":
        return [(GEOM_POLYGON, [_strip_closing(_as_ring(r)) for r in poly]) for poly in coords]
    raise ValueError(f"Неподдерживаемый тип геометрии: {gtype}")


def _transform(transformer: Transformer, ring: np.ndarray) -> np.ndarray:
    """Перепроецирует кольцо (N, 2)."""
    xs, ys = transformer.transform(ring[:, 0], ring[:, 1])
    return np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])


def _to_pixel(grid: Grid, ring: np.ndarray) -> np.ndarray:
    """Мир (crs.internal) -> (col, row) Grid, float."""
    col, row = grid.transform_to_pixel(ring[:, 0], ring[:, 1])
    return np.column_stack([np.asarray(col, dtype=np.float64), np.asarray(row, dtype=np.float64)])


def _declared_crs(fc: dict[str, Any]) -> str | None:
    """CRS из legacy-члена `crs` GeoJSON (если есть)."""
    member = fc.get("crs")
    if not isinstance(member, dict):
        return None
    name = (member.get("properties") or {}).get("name")
    return str(name) if name else None


def _crs_member(crs: str) -> dict[str, Any]:
    """Legacy-член `crs` для GeoJSON (QGIS/GDAL его читают)."""
    epsg = CRS.from_user_input(crs).to_epsg()
    if epsg is None:
        raise ValueError(f"CRS без EPSG-кода: {crs}")
    return {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}


# ------------------------------------------------------------
# Контейнеры
# ------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class LabelFeature:
    """Объект разметки в crs.internal и в пикселях Grid.

    geom_type: Point | Polygon.
    world: кольца (N, 2) в crs.internal; Point — одно кольцо (1, 2);
        Polygon — [0] внешнее, остальные — дыры.
    pixel: те же кольца в (col, row) Grid, float.
    """

    cls: str
    geom_type: str
    world: tuple[np.ndarray, ...]
    pixel: tuple[np.ndarray, ...]
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def centroid_world(self) -> np.ndarray:
        """Точка или центр масс внешнего кольца, crs.internal."""
        if self.geom_type == GEOM_POINT:
            return self.world[0][0]
        return ring_centroid(self.world[0])


@dataclass(frozen=True, eq=False)
class Detection:
    """Предсказанный объект в crs.internal.

    geometry: Point — (1, 2); Polygon — внешнее кольцо (N, 2) без замыкания.
    """

    cls: str
    score: float
    geom_type: str
    geometry: np.ndarray

    def __post_init__(self) -> None:
        """Валидация типа и формы."""
        if self.geom_type not in (GEOM_POINT, GEOM_POLYGON):
            raise ValueError(f"Detection: неизвестный тип {self.geom_type}")
        if self.geometry.ndim != 2 or self.geometry.shape[1] != 2:
            raise ValueError(
                f"Detection: geometry должна быть (N, 2), получено {self.geometry.shape}"
            )
        if self.geom_type == GEOM_POLYGON and len(self.geometry) < 3:
            raise ValueError("Detection: у полигона меньше 3 вершин")

    @property
    def centroid(self) -> np.ndarray:
        """Точка или центр масс полигона, crs.internal."""
        if self.geom_type == GEOM_POINT:
            return self.geometry[0]
        return ring_centroid(self.geometry)


# ------------------------------------------------------------
# Чтение / запись
# ------------------------------------------------------------
def read_labels(
    path: Path,
    grid: Grid,
    src_crs: str,
    class_key: str = _DEFAULT_CLASS_KEY,
) -> list[LabelFeature]:
    """Читает разметку GeoJSON и переводит в crs.internal и пиксели Grid.

    Args:
        path: путь к GeoJSON (FeatureCollection).
        grid: целевая сетка (её CRS — crs.internal).
        src_crs: CRS координат файла (crs.output, EPSG:3857).
        class_key: имя свойства с классом.

    Returns:
        Список LabelFeature (Multi* развёрнуты).

    Raises:
        ValueError: не FeatureCollection, объявленный CRS файла != src_crs,
            нет класса, неподдерживаемая геометрия.
    """
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    if fc.get("type") != "FeatureCollection":
        raise ValueError(f"{path}: ожидался FeatureCollection, получено {fc.get('type')}")

    declared = _declared_crs(fc)
    if declared is not None and CRS.from_user_input(declared) != CRS.from_user_input(src_crs):
        raise ValueError(f"{path}: CRS файла {declared} != ожидаемого {src_crs}")

    transformer = Transformer.from_crs(src_crs, grid.crs, always_xy=True)
    out: list[LabelFeature] = []
    for i, feat in enumerate(fc.get("features", [])):
        props = feat.get("properties") or {}
        geom = feat.get("geometry")
        if geom is None:
            logger.warning("%s: feature %d без геометрии, пропуск", Path(path).name, i)
            continue
        if class_key not in props:
            raise ValueError(f"{path}: feature {i} без свойства '{class_key}'")
        cls = str(props[class_key])
        for gtype, rings in _explode(geom):
            world = tuple(_transform(transformer, r) for r in rings)
            pixel = tuple(_to_pixel(grid, r) for r in world)
            out.append(
                LabelFeature(cls=cls, geom_type=gtype, world=world, pixel=pixel, properties=props)
            )

    logger.info(
        "Разметка %s: %d объектов (Point %d, Polygon %d)",
        Path(path).name,
        len(out),
        sum(f.geom_type == GEOM_POINT for f in out),
        sum(f.geom_type == GEOM_POLYGON for f in out),
    )
    return out


def write_detections(
    path: Path,
    detections: Iterable[Detection],
    src_crs: str,
    dst_crs: str,
    class_key: str = _DEFAULT_CLASS_KEY,
) -> int:
    """Пишет детекции в GeoJSON в dst_crs.

    Args:
        path: выходной файл (папка создаётся).
        detections: детекции в src_crs (crs.internal).
        src_crs: CRS координат детекций.
        dst_crs: CRS выхода (crs.output, EPSG:3857).
        class_key: имя свойства с классом.

    Returns:
        Число записанных объектов.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    features: list[dict[str, Any]] = []
    for det in detections:
        coords = _transform(transformer, det.geometry)
        if det.geom_type == GEOM_POINT:
            geometry = {"type": GEOM_POINT, "coordinates": coords[0].tolist()}
        else:
            ring = coords.tolist()
            ring.append(ring[0])
            geometry = {"type": GEOM_POLYGON, "coordinates": [ring]}
        features.append(
            {
                "type": "Feature",
                "properties": {class_key: det.cls, _SCORE_KEY: float(det.score)},
                "geometry": geometry,
            }
        )

    fc = {"type": "FeatureCollection", "crs": _crs_member(dst_crs), "features": features}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    logger.info("Записано детекций: %d -> %s", len(features), path)
    return len(features)
