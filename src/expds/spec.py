"""Контракты данных КОЗ №3.

Единый источник правды для Grid / Layer / Site / Detection / DataSpec.
Без Hydra, без pydantic — только dataclasses.

ПРАВИЛО КООРДИНАТ: работаем через ЦЕНТР пикселя.
    x = c + (col + 0.5) * a
    y = f - (row + 0.5) * e        (стандартный north-up transform)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Допустимые группы слоёв. Выход за пределы — ошибка загрузки спеки.
VALID_GROUPS = {"relief", "optic", "geo"}
VALID_KINDS = {"raster", "pointcloud"}


@dataclass(frozen=True)
class Grid:
    """Описывает общую растровую сетку (одна на сайт).

    transform — аффинный tuple (a, b, c, d, e, f), как в rasterio.
    Для north-up: b = d = 0, a > 0, e < 0.
    """

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    @property
    def pixel_size(self) -> float:
        """Размер пикселя в единицах CRS (по оси X)."""
        a = self.transform[0]
        e = self.transform[4]
        return float(max(abs(a), abs(e)))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Границы (left, bottom, right, top) в координатах CRS."""
        a, _b, c, _d, e, f = self.transform
        left = c
        top = f
        right = c + a * self.width
        bottom = f + e * self.height
        # e < 0 (north-up) => bottom < top
        return (
            float(min(left, right)),
            float(min(bottom, top)),
            float(max(left, right)),
            float(max(bottom, top)),
        )

    def pixel_to_world(self, rows: Any, cols: Any):
        """Индексы (row, col) -> координаты ЦЕНТРА пикселя (x, y)."""
        import numpy as np

        a, _b, c, _d, e, f = self.transform
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        xs = c + (cols + 0.5) * a
        ys = f + (rows + 0.5) * e
        return xs, ys

    def world_to_pixel(self, xs: Any, ys: Any):
        """Координаты (x, y) -> индексы пикселей (rows, cols) как float."""
        import numpy as np

        a, _b, c, _d, e, f = self.transform
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        cols = (xs - c) / a - 0.5
        rows = (ys - f) / e - 0.5
        return rows, cols

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "transform": list(self.transform),
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Grid:
        return cls(
            crs=str(d["crs"]),
            transform=tuple(float(v) for v in d["transform"]),  # type: ignore[arg-type]
            width=int(d["width"]),
            height=int(d["height"]),
        )


@dataclass(frozen=True)
class Layer:
    """Описание одного слоя внутри сайта."""

    name: str  # "dtm", "hillshade", "rgb", "mag", "gpr_slice"
    group: str  # "relief" | "optic" | "geo"
    glob: str  # шаблон пути внутри сайта, напр. "*.tif" или "dtm.tif"
    kind: str  # "raster" | "pointcloud"
    nodata: float | None = None
    optional: bool = False  # True = слой может отсутствовать на части территории


@dataclass
class Detection:
    """Одно предсказание в EPSG:3857."""

    geometry: dict[str, Any]  # GeoJSON-геометрия (Polygon / Point)
    score: float
    class_id: int
    class_name: str
    source: str  # "relief" | "optic" | "geo" | "fused"

    def to_geojson_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": self.geometry,
            "properties": {
                "score": float(self.score),
                "class_id": int(self.class_id),
                "class_name": self.class_name,
                "source": self.source,
            },
        }


@dataclass(frozen=True)
class Site:
    """Один участок съёмки."""

    site_id: str  # "site_1"
    root: Path  # абсолютный путь
    grid: Grid | None  # None => вывести из первого растра на диске
    layers: dict[str, Path] = field(default_factory=dict)  # name -> путь к данным


@dataclass(frozen=True)
class DataSpec:
    """Полная спецификация датасета."""

    sites: list[Site]
    classes: dict[int, str]  # {0: "background", 1: "kurgan", ...}
    layer_defs: list[Layer]
    tile: int
    overlap: int
    r_tolerance: float
    iou_threshold: float
    sigma_px: float = 8.0
    r_nms: float = 10.0


def _fail(msg: str) -> None:
    raise ValueError(f"[spec] {msg}")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        _fail(msg)


def _parse_layer(raw: dict[str, Any]) -> Layer:
    _require("name" in raw, "Layer без поля name")
    _require("group" in raw, f"Layer {raw.get('name')} без поля group")
    group = str(raw["group"])
    _require(group in VALID_GROUPS, f"Layer {raw['name']}: group={group!r} не в {VALID_GROUPS}")

    glob = raw.get("glob", "")
    _require(isinstance(glob, str) and glob.strip() != "", f"Layer {raw['name']}: пустой glob")

    kind = str(raw.get("kind", "raster"))
    _require(kind in VALID_KINDS, f"Layer {raw['name']}: kind={kind!r} не в {VALID_KINDS}")

    nodata = raw.get("nodata", None)
    if nodata is not None:
        nodata = float(nodata)

    return Layer(
        name=str(raw["name"]),
        group=group,
        glob=glob,
        kind=kind,
        nodata=nodata,
        optional=bool(raw.get("optional", False)),
    )


def load_spec(path: str | Path) -> DataSpec:
    """Загружает и ВАЛИДИРУЕТ DataSpec из YAML.

    Падаем громко и рано: несовпадение CRS, отсутствие обязательного слоя,
    кривые tile/overlap/r_tolerance/iou_threshold — это ошибки загрузки,
    а не что-то, что можно молча проглотить.
    """
    path = Path(path)
    _require(path.is_file(), f"файл спеки не найден: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _require(isinstance(raw, dict), "верхний уровень YAML должен быть словарём")

    project_root = path.resolve().parent.parent  # configs/ -> корень репо

    # --- layer_defs ---
    raw_layers = raw.get("layers", [])
    _require(
        isinstance(raw_layers, list) and len(raw_layers) > 0, "layers: должен быть непустым списком"
    )
    layer_defs = [_parse_layer(d) for d in raw_layers]
    layer_names = [ldef.name for ldef in layer_defs]
    _require(len(set(layer_names)) == len(layer_names), f"дублирующиеся имена слоёв: {layer_names}")

    # --- classes ---
    raw_classes = raw.get("classes", {})
    _require(
        isinstance(raw_classes, dict) and len(raw_classes) > 0,
        "classes: должен быть непустым словарём {id: name}",
    )
    classes = {int(k): str(v) for k, v in raw_classes.items()}

    # --- скалярные параметры ---
    tile = int(raw.get("tile", 512))
    overlap = int(raw.get("overlap", 128))
    r_tolerance = float(raw.get("r_tolerance", 10.0))
    iou_threshold = float(raw.get("iou_threshold", 0.5))

    _require(
        tile > overlap >= 0,
        f"требуется tile > overlap >= 0, получено tile={tile}, overlap={overlap}",
    )
    _require(tile % 32 == 0, f"tile={tile} должен делиться на 32")
    _require(r_tolerance > 0, f"r_tolerance={r_tolerance} должен быть > 0")
    _require(0 < iou_threshold <= 1, f"iou_threshold={iou_threshold} должен быть в (0, 1]")

    # --- sites ---
    raw_sites = raw.get("sites", [])
    _require(
        isinstance(raw_sites, list) and len(raw_sites) > 0, "sites: должен быть непустым списком"
    )

    sites: list[Site] = []
    crs_seen: set[str] = set()

    for rs in raw_sites:
        _require(isinstance(rs, dict), "каждый site — словарь")
        site_id = str(rs.get("site_id", ""))
        _require(site_id != "", "у site пустой site_id")

        root_raw = rs.get("root", "")
        _require(str(root_raw).strip() != "", f"site {site_id}: пустой root")
        site_root = (
            (project_root / root_raw).resolve()
            if not Path(root_raw).is_absolute()
            else Path(root_raw).resolve()
        )

        # grid: либо в спеке, либо None (выведем из растра)
        grid = None
        if rs.get("grid"):
            grid = Grid.from_dict(rs["grid"])
            crs_seen.add(grid.crs)

        # layers: name -> путь (глоб)
        site_layers = rs.get("layers", {})
        _require(
            isinstance(site_layers, dict),
            f"site {site_id}: layers должен быть словарём {{name: путь}}",
        )

        found: dict[str, Path] = {}
        for ldef in layer_defs:
            pat = site_layers.get(ldef.name, ldef.glob)
            base = site_root if site_root.exists() else project_root / str(root_raw)
            matches = sorted(base.glob(pat)) if base.exists() else []
            if matches:
                found[ldef.name] = matches[0].resolve()
            else:
                _require(
                    ldef.optional,
                    f"site {site_id}: обязательный слой {ldef.name!r} "
                    f"(glob={pat!r}) не найден под {base}",
                )
        sites.append(Site(site_id=site_id, root=site_root, grid=grid, layers=found))

    _require(
        len(crs_seen) <= 1, f"все sites должны иметь одинаковый crs, найдено: {sorted(crs_seen)}"
    )

    return DataSpec(
        sites=sites,
        classes=classes,
        layer_defs=layer_defs,
        tile=tile,
        overlap=overlap,
        r_tolerance=r_tolerance,
        iou_threshold=iou_threshold,
        sigma_px=float(raw.get("sigma_px", 8.0)),
        r_nms=float(raw.get("r_nms", 10.0)),
    )
