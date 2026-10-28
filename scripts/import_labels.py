"""Импорт разметки участка в labels.geojson (конвенция tiles-лоадера).

Вся остальная цепочка GT -> таргеты в репо уже есть, скрипт её не дублирует:

    src/expds/io/geojson.py::read_labels       — проверяет CRS, переводит
        координаты 3857 -> crs.internal (EPSG:25833) и в пиксели Grid;
    src/expds/tiles/dataset.py::labels_to_targets — растеризует в y_seg/y_hm;
    src/expds/tiles/dataset.py::load_site      — сам подхватывает
        <site_dir>/labels.geojson (labels_crs из конфига, у real = EPSG:3857).

Здесь НЕТ ни репроекции, ни растеризации: разметка 006 уже в crs.output
(EPSG:3857). Скрипт только приводит файл к конвенции:

    <site_dir>/labels.geojson — FeatureCollection, у каждого feature свойство
    класса (--class-key, default "class" = _DEFAULT_CLASS_KEY в io/geojson.py)
    со значением из targets.classes (mound | rampart | ditch).

Откуда берётся класс (по убыванию приоритета):
  1. properties: авто-детект ключа по узнаваемым значениям (ALIASES);
  2. --default-class (у 006 класса в properties НЕТ — только fid);
  3. имя файла: "..._Li_курганы_поврежденные.geojson" -> mound.

Реальный пример (участок 006, пути из реестра PLAN.md):
    uv run python scripts/import_labels.py \
        --src data/raw/006_Осечки_1.1км/06_Осечки_1.1км_разметка \
        --out data/raw/006_Осечки_1.1км \
        --las data/raw/006_Осечки_1.1км/01_Осечки_1.1км_La_точки \
        --default-class mound

CRS облака las для bbox-сверки берётся из --check-crs, а если не задан —
автоматически из UTM.json в корне участка (у las TerraScan CRS в заголовке пуст).

Член "crs" в выходной файл сознательно не пишется: read_labels при его
отсутствии пропускает сверку объявленного CRS и берёт labels_crs из конфига.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from pyproj import Transformer

# Узнаваемые значения класса -> канонические имена targets.classes.
ALIASES: dict[str, str] = {
    "mound": "mound",
    "rampart": "rampart",
    "ditch": "ditch",
    # русские значения
    "курган": "mound",
    "насыпь": "mound",
    "вал": "rampart",
    "валик": "rampart",
    "ров": "ditch",
    "канава": "ditch",
}

# Стемы для распознавания класса в ИМЕНИ файла разметки.
# Порядок важен: первый совпавший стем выигрывает.
NAME_STEMS: tuple[tuple[str, str], ...] = (
    ("курган", "mound"),
    ("mound", "mound"),
    ("насып", "mound"),
    ("rampart", "rampart"),
    ("вал", "rampart"),
    ("ditch", "ditch"),
    ("рв", "ditch"),  # рвы, рвов
    ("ров", "ditch"),
)


def norm(value: object) -> str:
    return str(value).strip().lower()


def norm_crs(value: object) -> str | None:
    """EPSG:3857 / urn:ogc:def:crs:EPSG::3857 / 3857 -> epsg:3857."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v.startswith("urn:ogc:def:crs:"):
        v = v.rsplit(":", 1)[-1]
    if v.startswith("epsg:"):
        v = v.split(":", 1)[1]
    return f"epsg:{v}" if v.isdigit() else (v or None)


def load_fc(src: Path) -> tuple[list[dict], str | None]:
    """Прочитать разметку: (features, объявленный CRS | None)."""
    data = json.loads(src.read_text(encoding="utf-8"))
    declared: str | None = None
    if isinstance(data, dict):
        crs = data.get("crs")
        if isinstance(crs, dict):
            declared = (crs.get("properties") or {}).get("name") or None
        if data.get("type") == "FeatureCollection":
            return list(data.get("features") or []), declared
        if data.get("type") == "Feature":
            return [data], declared
    if isinstance(data, list):
        return [f for f in data if isinstance(f, dict)], declared
    raise SystemExit(f"{src}: ожидаю FeatureCollection/Feature/список, не понял формат")


def detect_class_key(features: list[dict]) -> str | None:
    """Ключ properties, чьи значения чаще всего узнаются по ALIASES."""
    scores: dict[str, int] = {}
    for feat in features:
        for key, value in (feat.get("properties") or {}).items():
            if norm(value) in ALIASES:
                scores[key] = scores.get(key, 0) + 1
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def infer_class_from_name(name: str) -> str | None:
    """Класс из имени файла: 'Осечки_1.1км_Li_курганы_поврежденные.geojson' -> mound."""
    n = name.lower()
    for stem, cls in NAME_STEMS:
        if stem in n:
            return cls
    return None


def iter_points(geometry: dict) -> Iterator[tuple[float, float]]:
    """Плоский обход всех вершин геометрии (x, y)."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Point":
        yield coords[0], coords[1]
    elif gtype in ("MultiPoint", "LineString"):
        for p in coords:
            yield p[0], p[1]
    elif gtype in ("MultiLineString", "Polygon"):
        for part in coords:
            for p in part:
                yield p[0], p[1]
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for p in ring:
                    yield p[0], p[1]
    else:
        raise SystemExit(f"Неизвестный тип геометрии: {gtype!r}")


def bounds_in_crs(
    features: list[dict], src_crs: str, dst_crs: str
) -> tuple[float, float, float, float]:
    """BBox всех геометрий, переведённый из src_crs в dst_crs (для сверки с LAS)."""
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs: list[float] = []
    ys: list[float] = []
    for feat in features:
        for x, y in iter_points(feat.get("geometry") or {}):
            tx, ty = tr.transform(x, y)
            xs.append(tx)
            ys.append(ty)
    if not xs:
        raise SystemExit("Ни одной координаты в геометриях")
    return min(xs), min(ys), max(xs), max(ys)


def resolve_las(path: Path) -> Path:
    """Файл LAS/LAZ или каталог с ним (например, 01_*_La_точки)."""
    if path.is_file():
        return path
    if path.is_dir():
        cands = sorted({*path.glob("*.las"), *path.glob("*.laz")})
        if not cands:
            cands = sorted(path.rglob("*.las")) + sorted(path.rglob("*.laz"))
        if cands:
            if len(cands) > 1:
                print(f"!! В {path} несколько LAS/LAZ, беру {cands[0].name}", file=sys.stderr)
            return cands[0]
    raise SystemExit(f"--las {path}: не найден .las/.laz")


def las_bounds(path: Path) -> tuple[float, float, float, float]:
    """Границы LAS из заголовка (сами точки не читаем)."""
    import laspy

    with laspy.open(path) as reader:
        header = reader.header
        try:
            return header.x_min, header.y_min, header.x_max, header.y_max
        except AttributeError:  # старый laspy
            return header.mins[0], header.mins[1], header.maxs[0], header.maxs[1]


def resolve_check_crs(explicit: str | None, site_dir: Path) -> str | None:
    """CRS облака las: --check-crs или UTM.json в корне участка.

    las участков пишет TerraScan с пустым CRS в заголовке; реальный кадр
    декларирует UTM.json (для 006: urn:ogc:def:crs:EPSG::32637, UTM 37N).
    """
    if explicit:
        return explicit
    meta = site_dir / "UTM.json"
    if meta.is_file():
        try:
            crs = json.loads(meta.read_text(encoding="utf-8")).get("crs")
        except Exception:
            return None
        if crs:
            print(f"CRS сверки bbox: {crs} (из {meta.name})")
            return crs
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Разметка участка -> labels.geojson (конвенция tiles-лоадера)",
    )
    parser.add_argument("--src", required=True, type=Path, help="GeoJSON разметки или папка с ним")
    parser.add_argument(
        "--out", required=True, type=Path, help="Файл labels.geojson или папка сайта"
    )
    parser.add_argument(
        "--class-key",
        default="class",
        help="Имя свойства класса на выходе (default: class — совпадает с _DEFAULT_CLASS_KEY)",
    )
    parser.add_argument(
        "--classes",
        default="mound,rampart,ditch",
        help="Разрешённые классы через запятую (targets.classes конфига)",
    )
    parser.add_argument(
        "--default-class",
        default=None,
        help="Класс всем фичам, если в properties его нет (для 006: mound)",
    )
    parser.add_argument(
        "--src-crs",
        default="EPSG:3857",
        help="CRS координат разметки (default: EPSG:3857; URN-запись тоже понимается)",
    )
    parser.add_argument(
        "--las",
        type=Path,
        default=None,
        help="LAS или каталог 01_*_La_точки: bbox-сверка разметки с облаком",
    )
    parser.add_argument(
        "--check-crs",
        default=None,
        help="CRS облака las для bbox-сверки (default: из UTM.json в корне участка)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Писать файл, даже если часть значений не распознана"
    )
    args = parser.parse_args(argv)

    allowed = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.default_class and args.default_class not in allowed:
        raise SystemExit(f"--default-class {args.default_class!r} не входит в --classes {allowed}")

    src = args.src
    if src.is_dir():
        # .geojson предпочтительнее .json — чтобы не схватить UTM.json в корне участка.
        candidates = sorted({*src.glob("*.geojson")}) or sorted({*src.glob("*.json")})
        if len(candidates) != 1:
            raise SystemExit(
                f"{src}: ждал один файл разметки, нашёл {len(candidates)}: {candidates}"
            )
        src = candidates[0]

    features, declared = load_fc(src)
    if not features:
        raise SystemExit(f"{src}: ни одного feature")
    print(f"Источник: {src} — {len(features)} feature(s)")
    if declared is not None and norm_crs(declared) != norm_crs(args.src_crs):
        print(
            f"!! Файл объявляет CRS {declared!r}, а считаю его {args.src_crs!r} — "
            "если разметка реально в другой проекции, проверь --src-crs.",
            file=sys.stderr,
        )

    detected = detect_class_key(features)
    all_class = ""
    if detected is not None:
        print(f"Свойство класса: {detected!r} -> в выходном файле {args.class_key!r}")
    else:
        fallback = args.default_class or infer_class_from_name(src.name)
        if fallback is None:
            props = [f.get("properties") or {} for f in features]
            raise SystemExit(
                "В properties нет класса, и --default-class/имя файла не помогли.\n"
                f"Свойства фич: {props}\n"
                "-> задай --default-class mound|rampart|ditch или дополни ALIASES/NAME_STEMS."
            )
        origin = "--default-class" if args.default_class else f"имя файла ({src.name})"
        all_class = fallback
        print(
            f"!! Класса в properties нет — назначаю всем {fallback!r} "
            f"(источник: {origin}); в выходном файле свойство {args.class_key!r}"
        )

    counts: dict[str, int] = {}
    gtypes: dict[str, int] = {}
    problems: list[str] = []
    out_features: list[dict] = []
    for i, feat in enumerate(features):
        props = dict(feat.get("properties") or {})
        if detected is not None:
            raw = props.get(detected)
            canonical = ALIASES.get(norm(raw)) if raw is not None else None
            if canonical is None or canonical not in allowed:
                problems.append(
                    f"feature {i}: {detected}={raw!r} -> не распознано (разрешено: {allowed})"
                )
                continue
        else:
            canonical = all_class
        props[args.class_key] = canonical
        counts[canonical] = counts.get(canonical, 0) + 1
        gtype = (feat.get("geometry") or {}).get("type") or "?"
        gtypes[gtype] = gtypes.get(gtype, 0) + 1
        out_features.append({**feat, "properties": props})

    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"  геометрии: {dict(sorted(gtypes.items()))}")

    if problems:
        for line in problems:
            print(f"  !! {line}", file=sys.stderr)
        if not args.force:
            raise SystemExit(
                f"Не распознано {len(problems)} из {len(features)} — файл не записан. "
                "Дополни ALIASES или запусти с --force (осознанно)."
            )

    if not out_features:
        raise SystemExit("Ни одного валидного feature — писать нечего.")

    out_is_file = args.out.suffix.lower() in (".json", ".geojson")
    out_path = args.out if out_is_file else args.out / "labels.geojson"
    site_dir = args.out.parent if out_is_file else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Без члена "crs": read_labels тогда берёт labels_crs из конфига и
    # сверка "объявленный != ожидаемый" не может отвалиться на формате.
    out_path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": out_features}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Записано: {out_path}")

    check_crs = resolve_check_crs(args.check_crs, site_dir)
    if check_crs is not None:
        gx0, gy0, gx1, gy1 = bounds_in_crs(out_features, args.src_crs, check_crs)
        print(f"BBox разметки ({check_crs}): X {gx0:.1f}..{gx1:.1f}, Y {gy0:.1f}..{gy1:.1f}")
    if args.las is not None:
        if check_crs is None:
            print(
                "!! --las задан, но CRS сверки неизвестен: "
                "укажи --check-crs или положи UTM.json в корень участка",
                file=sys.stderr,
            )
        else:
            lx0, ly0, lx1, ly1 = las_bounds(resolve_las(args.las))
            print(
                f"BBox LAS (как {check_crs}):        X {lx0:.1f}..{lx1:.1f}, Y {ly0:.1f}..{ly1:.1f}"
            )
            overlaps = gx1 >= lx0 and gx0 <= lx1 and gy1 >= ly0 and gy0 <= ly1
            if overlaps:
                print("Разметка пересекается с облаком точек ✓")
            else:
                print(
                    "ПЕРЕСЕЧЕНИЯ НЕТ — проверь --src-crs/--check-crs и границы участка",
                    file=sys.stderr,
                )
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
