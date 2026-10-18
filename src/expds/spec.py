"""Контракты данных КОЗ №3: реестр слоёв и участков.

`load_spec` читает ТОТ ЖЕ YAML, что и ``expds.pipeline_config``: масштаб тайлов
берётся из ``tiles``, классы — из ``targets``, метрика — из ``eval``, радиус NMS —
из ``post``. Переименований нет, поэтому значения физически не могут разойтись
между двумя парсерами. Дополнительно читаются top-level блоки ``layers`` и
``sites`` — их ``pipeline_config`` не смотрит.

`Grid` и `Detection` здесь НЕ определяются, а берутся из канонических модулей:

* ``Grid`` — :mod:`expds.features.grid` (row 0 = юг, одна конвенция на пайплайн);
* ``Detection`` — :mod:`expds.io.geojson` (numpy-геометрия во внутреннем CRS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from expds.features.grid import Grid
from expds.io.geojson import Detection

__all__ = ["DataSpec", "Detection", "Grid", "Layer", "Site", "load_spec"]

#: Допустимые группы слоёв. Выход за пределы — ошибка загрузки спеки.
VALID_GROUPS = frozenset({"relief", "optic", "geo"})
#: Допустимые типы данных слоя.
VALID_KINDS = frozenset({"raster", "pointcloud"})

#: tile_px должен делиться на это число (согласовано с network stride).
TILE_DIVISOR = 32


@dataclass(frozen=True)
class Layer:
    """Описание одного слоя внутри участка."""

    name: str
    group: str
    glob: str
    kind: str
    nodata: float | None = None
    optional: bool = False


@dataclass
class Site:
    """Один участок съёмки."""

    site_id: str
    root: Path
    grid: Grid | None
    layers: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class DataSpec:
    """Полная спецификация датасета."""

    sites: list[Site]
    classes: tuple[str, ...]
    layer_defs: list[Layer]
    tile: int
    overlap: int
    r_tolerance: float
    iou_threshold: float
    sigma_px: float
    r_nms: float


def _fail(msg: str) -> None:
    """Падает с сообщением, помеченным префиксом [spec]."""
    raise ValueError(f"[spec] {msg}")


def _require(cond: bool, msg: str) -> None:
    """Падает, если условие ложно."""
    if not cond:
        _fail(msg)


def _parse_layer(raw: Any) -> Layer:
    """Разбирает и валидирует один элемент блока ``layers``."""
    _require(isinstance(raw, dict), "каждый layer — словарь")
    _require("name" in raw, "Layer без поля name")
    _require("group" in raw, f"Layer {raw.get('name')} без поля group")

    name = str(raw["name"])
    _require(name.strip() != "", "у layer пустое name")

    group = str(raw["group"])
    _require(group in VALID_GROUPS, f"Layer {name}: group={group!r} не в {sorted(VALID_GROUPS)}")

    glob = raw.get("glob", "")
    _require(
        isinstance(glob, str) and glob.strip() != "",
        f"Layer {name}: пустой glob",
    )

    kind = str(raw.get("kind", "raster"))
    _require(kind in VALID_KINDS, f"Layer {name}: kind={kind!r} не в {sorted(VALID_KINDS)}")

    nodata_raw = raw.get("nodata")
    nodata = None if nodata_raw is None else float(nodata_raw)

    return Layer(
        name=name,
        group=group,
        glob=glob,
        kind=kind,
        nodata=nodata,
        optional=bool(raw.get("optional", False)),
    )


def _grid_from_dict(d: Any, *, site_id: str) -> Grid:
    """Строит :class:`expds.features.grid.Grid` из блока ``grid`` в YAML."""
    _require(isinstance(d, dict), f"site {site_id}: grid должен быть словарём или null")
    missing = {"crs", "pixel_size", "x_min", "y_min", "width", "height"} - set(d)
    _require(
        not missing,
        f"site {site_id}: в grid не хватает полей {sorted(missing)}",
    )
    return Grid(
        crs=str(d["crs"]),
        pixel_size=float(d["pixel_size"]),
        x_min=float(d["x_min"]),
        y_min=float(d["y_min"]),
        width=int(d["width"]),
        height=int(d["height"]),
    )


def load_spec(path: str | Path) -> DataSpec:
    """Загружает и ВАЛИДИРУЕТ :class:`DataSpec` из YAML.

    Читает тот же файл, что ``expds.pipeline_config``: обязательны секции
    ``data``/``targets``/``tiles``. Падаем громко и рано: несовпадение CRS
    между участками, отсутствие обязательного слоя, кривые tile/overlap —
    это ошибки загрузки, а не то, что можно молча проглотить.

    Args:
        path: путь к YAML датасета.

    Returns:
        DataSpec.
    """
    path = Path(path)
    _require(path.is_file(), f"файл спеки не найден: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _require(isinstance(raw, dict), "верхний уровень YAML должен быть словарём")

    project_root = path.resolve().parent.parent  # configs/ -> корень репо

    # --- обязательные секции костяка ---
    data_raw = raw.get("data")
    _require(isinstance(data_raw, dict), "нет секции 'data'")
    targets_raw = raw.get("targets")
    _require(isinstance(targets_raw, dict), "нет секции 'targets'")
    tiles_raw = raw.get("tiles")
    _require(isinstance(tiles_raw, dict), "нет секции 'tiles'")

    raw_classes = targets_raw.get("classes")
    _require(
        isinstance(raw_classes, list) and len(raw_classes) > 0,
        "targets.classes должен быть непустым списком",
    )
    _require(
        len(set(map(str, raw_classes))) == len(raw_classes),
        f"дублирующиеся классы: {raw_classes}",
    )
    classes = tuple(str(c) for c in raw_classes)

    sites_dir_raw = data_raw.get("sites_dir")
    _require(
        isinstance(sites_dir_raw, str) and sites_dir_raw.strip() != "",
        "data.sites_dir пустой",
    )
    sites_dir = Path(sites_dir_raw)

    # --- слои ---
    raw_layers = raw.get("layers", [])
    _require(
        isinstance(raw_layers, list) and len(raw_layers) > 0,
        "блок 'layers' должен быть непустым списком",
    )
    layer_defs = [_parse_layer(d) for d in raw_layers]
    layer_names = [ldef.name for ldef in layer_defs]
    _require(
        len(set(layer_names)) == len(layer_names),
        f"дублирующиеся имена слоёв: {layer_names}",
    )

    # --- масштаб тайлов: те же ключи, что в pipeline_config ---
    _require("tile_px" in tiles_raw, "tiles.tile_px обязателен")
    _require("overlap_px" in tiles_raw, "tiles.overlap_px обязателен")
    tile = int(tiles_raw["tile_px"])
    overlap = int(tiles_raw["overlap_px"])
    _require(
        tile > overlap >= 0,
        f"требуется tiles.tile_px > tiles.overlap_px >= 0, получено {tile} / {overlap}",
    )
    _require(
        tile % TILE_DIVISOR == 0,
        f"tiles.tile_px={tile} должен делиться на {TILE_DIVISOR}",
    )

    # --- метрика ---
    eval_raw = raw.get("eval") or {}
    _require(isinstance(eval_raw, dict), "секция 'eval' должна быть словарём")
    r_tolerance = float(eval_raw.get("r_tol", 5.0))
    iou_threshold = float(eval_raw.get("iou_thr", 0.5))
    _require(r_tolerance > 0, f"eval.r_tol={r_tolerance} должен быть > 0")
    _require(0 < iou_threshold <= 1, f"eval.iou_thr={iou_threshold} должен быть в (0, 1]")

    post_raw = raw.get("post") or {}
    _require(isinstance(post_raw, dict), "секция 'post' должна быть словарём")
    r_nms = float(post_raw.get("nms_radius_px", 6.0))

    sigma_px = float(targets_raw.get("heatmap_sigma_px", 8.0))

    # --- участки ---
    raw_sites = raw.get("sites", [])
    _require(
        isinstance(raw_sites, list) and len(raw_sites) > 0,
        "блок 'sites' должен быть непустым списком",
    )

    sites: list[Site] = []
    crs_seen: set[str] = set()
    for rs in raw_sites:
        _require(isinstance(rs, dict), "каждый site — словарь")
        site_id = str(rs.get("site_id", ""))
        _require(site_id != "", "у site пустой site_id")

        # root необязателен: по умолчанию <data.sites_dir>/<site_id>.
        root_raw = rs.get("root")
        if root_raw is None or str(root_raw).strip() == "":
            site_root = sites_dir / site_id
        else:
            site_root = Path(str(root_raw))
        if not site_root.is_absolute():
            site_root = project_root / site_root
        site_root = site_root.resolve()

        grid = None
        if rs.get("grid"):
            grid = _grid_from_dict(rs["grid"], site_id=site_id)
            crs_seen.add(grid.crs)

        site_layers = rs.get("layers", {})
        _require(
            isinstance(site_layers, dict),
            f"site {site_id}: layers должен быть словарём {{name: путь}}",
        )

        found: dict[str, Path] = {}
        for ldef in layer_defs:
            pat = site_layers.get(ldef.name, ldef.glob)
            base = site_root if site_root.exists() else project_root / sites_dir
            matches = sorted(base.glob(str(pat))) if base.exists() else []
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
        len(crs_seen) <= 1,
        f"все sites должны иметь одинаковый crs, найдено: {sorted(crs_seen)}",
    )

    return DataSpec(
        sites=sites,
        classes=classes,
        layer_defs=layer_defs,
        tile=tile,
        overlap=overlap,
        r_tolerance=r_tolerance,
        iou_threshold=iou_threshold,
        sigma_px=sigma_px,
        r_nms=r_nms,
    )
