"""Слияние файлов разметки участка в один labels.geojson.

Организаторы кладут разметку по файлу на класс:
    06_<Участок>_разметка/
        <Участок>_Li_курганы_целые.geojson
        <Участок>_MagGr_FindsPoints.geojson
Класс закодирован в имени файла, в properties его нет.

Файлы приходят в двух разных CRS: чаще EPSG:3857, но у части участков
(030_Эсбю, 061_Граборг) — urn:ogc:def:crs:OGC:1.3:CRS84, то есть WGS84
в градусах. Конвейер метрический, поэтому здесь всё приводится к 3857.

    uv run python scripts/merge_labels.py --site data/raw/013_Нора_Виннон
    uv run python scripts/merge_labels.py --site data/raw/*/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pyproj import Transformer

logger = logging.getLogger(__name__)

#: Стем имени файла -> канонический класс регламента.
#: Геофизические слои организаторы называют латиницей (MagGr_FindsPoints),
#: лидарные и аэро — по-русски (Li_курганы_целые).
CLASS_BY_STEM: dict[str, str] = {
    # кириллица (La / Li / Ae / Or / SpOr)
    "курганы_целые": "kurgany_tselye",
    "курганы_поврежденные": "kurgany_povrezhdennye",
    "курганы_повреждённые": "kurgany_povrezhdennye",
    "городища": "gorodishcha",
    "фортификации": "fortifikatsii",
    "архитектуры": "arkhitektury",
    "находки": "finds_points",
    "прочие": "object_poly",
    "объекты": "object_poly",
    # латиница (MagGr / EtGr / TerFl)
    "FindsPoints": "finds_points",
    "ObjectPoly": "object_poly",
    "KurganyTselye": "kurgany_tselye",
    "KurganyPovrezhdennye": "kurgany_povrezhdennye",
    "Gorodishcha": "gorodishcha",
    "Fortifikatsii": "fortifikatsii",
    "Arkhitektury": "arkhitektury",
}

#: Токены модальности в имени файла.
MODALITY_TOKENS = ("MagGr", "EtGr", "TerFl", "SpOr", "La", "Li", "Ae", "Or")

#: Маркер папки разметки.
MARKUP_GLOB = "0*_*разметка*"

#: CRS, в которой конвейер ждёт разметку (data.labels_crs в конфиге).
TARGET_CRS = "EPSG:3857"

#: Объявленный в geojson CRS -> EPSG-код.
CRS_ALIASES: dict[str, str] = {
    "urn:ogc:def:crs:OGC:1.3:CRS84": "EPSG:4326",
    "urn:ogc:def:crs:OGC:2:84": "EPSG:4326",
    "CRS84": "EPSG:4326",
    "WGS84": "EPSG:4326",
    "urn:ogc:def:crs:EPSG::3857": "EPSG:3857",
    "urn:ogc:def:crs:EPSG::4326": "EPSG:4326",
}


def normalize_crs(declared: str) -> str:
    """Объявленный CRS -> EPSG-код.

    Args:
        declared: строка из crs.properties.name.

    Returns:
        'EPSG:NNNN'. Пустая строка трактуется как EPSG:3857 (так делает
        большинство файлов организаторов).

    Raises:
        ValueError: если CRS не распознан — молча угадывать нельзя,
            иначе разметка уедет на тысячи километров.
    """
    if not declared:
        return TARGET_CRS
    if declared in CRS_ALIASES:
        return CRS_ALIASES[declared]
    if "EPSG::" in declared:
        return "EPSG:" + declared.split("EPSG::")[-1]
    if declared.startswith("EPSG:"):
        return declared
    raise ValueError(f"неизвестный CRS: {declared!r}")


def reproject_geometry(geom: dict, tf: Transformer) -> dict:
    """Перепроецирует координаты геометрии любой вложенности.

    Args:
        geom: GeoJSON-геометрия (Point / LineString / Polygon /
            MultiPolygon / MultiPoint).
        tf: трансформер pyproj, созданный с always_xy=True.

    Returns:
        Новая геометрия с пересчитанными координатами.
    """

    def walk(coords: list) -> list:
        if coords and isinstance(coords[0], (int, float)):
            x, y = tf.transform(coords[0], coords[1])
            return [x, y]
        return [walk(c) for c in coords]

    if not geom or "coordinates" not in geom:
        return geom
    return {**geom, "coordinates": walk(geom["coordinates"])}


def split_name(stem: str) -> tuple[str, str]:
    """`Эрентуна_Li_курганы_целые` -> (модальность, стем класса).

    Args:
        stem: имя файла без расширения.

    Returns:
        (модальность, стем класса). Модальность '' если токена нет.
    """
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if part in MODALITY_TOKENS:
            return part, "_".join(parts[i + 1 :])
    return "", "_".join(parts[1:]) if len(parts) > 1 else stem


def class_of(path: Path) -> tuple[str, str] | None:
    """Класс и модальность по имени файла.

    Args:
        path: файл разметки.

    Returns:
        (класс, модальность) или None, если стем не распознан.
    """
    mod, stem = split_name(path.stem)
    cls = CLASS_BY_STEM.get(stem)
    if cls is None:
        for key, val in CLASS_BY_STEM.items():
            if key in stem:
                return val, mod
        return None
    return cls, mod


def merge_site(site: Path) -> tuple[dict, dict[str, int], list[str]]:
    """Собирает FeatureCollection по всем файлам разметки участка.

    Args:
        site: каталог участка.

    Returns:
        (коллекция в EPSG:3857, счётчик по классам, нераспознанные имена).

    Raises:
        FileNotFoundError: если папки разметки нет.
    """
    dirs = sorted(site.glob(MARKUP_GLOB))
    if not dirs:
        raise FileNotFoundError(f"{site}: нет папки разметки ({MARKUP_GLOB})")

    features: list[dict] = []
    counts: dict[str, int] = {}
    skipped: list[str] = []
    no_geom = 0

    for d in dirs:
        for p in sorted(d.glob("*.geojson")):
            parsed = class_of(p)
            if parsed is None:
                skipped.append(p.name)
                continue
            cls, mod = parsed
            raw = json.loads(p.read_text(encoding="utf-8"))

            declared = (raw.get("crs") or {}).get("properties", {}).get("name", "")
            file_crs = normalize_crs(declared)
            tf = (
                Transformer.from_crs(file_crs, TARGET_CRS, always_xy=True)
                if file_crs != TARGET_CRS
                else None
            )
            if tf is not None:
                logger.info("%s: перепроекция %s -> %s", p.name, file_crs, TARGET_CRS)

            for feat in raw.get("features", []):
                geom = feat.get("geometry")
                # geometry: null или пустые coordinates ([] / [[]]) —
                # строка атрибутивной таблицы без нарисованного контура.
                if geom is None or not geom.get("coordinates"):
                    no_geom += 1
                    continue
                if tf is not None:
                    geom = reproject_geometry(geom, tf)
                props = dict(feat.get("properties") or {})
                props["class"] = cls
                if mod:
                    props["source"] = mod
                props.setdefault("markup_file", p.name)
                features.append({"type": "Feature", "properties": props, "geometry": geom})
                counts[cls] = counts.get(cls, 0) + 1

    if no_geom:
        logger.warning("%s: фич без геометрии пропущено: %d", site.name, no_geom)

    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": TARGET_CRS}},
        "features": features,
    }
    return collection, counts, skipped


def main(argv: list[str] | None = None) -> int:
    """CLI слияния разметки. Возвращает число участков с проблемами."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", type=Path, nargs="+", required=True)
    ap.add_argument("--out-name", default="_labels_merged.geojson")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    problems = 0
    for site in args.site:
        print(f"\n=== {site.name} ===")
        try:
            coll, counts, skipped = merge_site(site)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            problems += 1
            continue

        print(f"  фич всего: {len(coll['features'])} (все в {TARGET_CRS})")
        for cls, n in sorted(counts.items()):
            print(f"    {cls:24} {n:4d}")
        if skipped:
            print(f"  НЕ РАСПОЗНАНО: {skipped}")
            problems += 1

        if not args.dry_run:
            out = site / args.out_name
            out.write_text(json.dumps(coll, ensure_ascii=False), encoding="utf-8")
            print(f"  -> {out}")
    return problems


if __name__ == "__main__":
    sys.exit(main())
