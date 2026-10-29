"""Отрисовка найденных объектов поверх растра-подложки.

Подложка — любой GeoTIFF участка (хиллшейд, ортофото, спутник).
Детекции читаются из GeoJSON в EPSG:3857 (формат сабмита) и
перепроецируются в CRS подложки, затем в пиксели по её affine-трансформу.

Цвета задаются палитрой класса. Поддерживаются обе схемы именования:
внутренняя (mound/rampart/ditch) и каноническая по регламенту
(kurgany_tselye и др.) — ключ ищется как есть, затем через _ALIASES.

    uv run python scripts/render_overlay.py \\
        --base data/raw/006_Осечки_1.1км/layers/hillshade_az_315.tif \\
        --geojson runs/<run>/pred.geojson \\
        --out runs/<run>/overlay.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

SUBMISSION_CRS = "EPSG:3857"

#: Палитра по каноническим классам регламента.
PALETTE: dict[str, str] = {
    "kurgany_tselye": "#2e7d32",
    "kurgany_povrezhdennye": "#ef6c00",
    "gorodishcha": "#1565c0",
    "fortifikatsii": "#c62828",
    "arkhitektury": "#6a1b9a",
    "finds_points": "#f9a825",
    "object_poly": "#616161",
}

#: Внутренние имена -> канонические (для отображения одним цветом).
_ALIASES: dict[str, str] = {
    "mound": "kurgany_tselye",
    "rampart": "fortifikatsii",
    "ditch": "object_poly",
}

_FALLBACK_COLOR = "#e91e63"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b)."""
    v = value.lstrip("#")
    return tuple(int(v[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def color_for(cls: str, palette: dict[str, str]) -> tuple[int, int, int]:
    """Цвет класса: прямое имя, затем алиас, затем запасной."""
    key = cls if cls in palette else _ALIASES.get(cls, "")
    return _hex_to_rgb(palette.get(key, _FALLBACK_COLOR))


def _stretch(band: np.ndarray, nodata: float | None) -> np.ndarray:
    """Растягивает одноканальный растр в uint8 по 2/98 перцентилям."""
    data = band.astype(np.float64)
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    if not valid.any():
        return np.zeros(band.shape, np.uint8)
    lo, hi = np.percentile(data[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0
    out[~valid] = 0
    return out.astype(np.uint8)


def load_base(path: Path, max_side: int = 4096) -> tuple[Image.Image, Any, str, float]:
    """Читает подложку и приводит к RGB.

    Args:
        path: GeoTIFF подложки.
        max_side: ограничение длинной стороны (память/вес PNG).

    Returns:
        (изображение RGB, affine-трансформ прочитанного окна, CRS, масштаб).
    """
    with rasterio.open(path) as src:
        scale = min(1.0, max_side / max(src.width, src.height))
        out_h = max(int(src.height * scale), 1)
        out_w = max(int(src.width * scale), 1)
        count = min(src.count, 3)
        data = src.read(
            indexes=list(range(1, count + 1)),
            out_shape=(count, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        crs = str(src.crs) if src.crs else ""
        nodata = src.nodata

    if count >= 3:
        rgb = np.stack([_stretch(data[i], nodata) for i in range(3)], axis=-1)
    else:
        g = _stretch(data[0], nodata)
        rgb = np.stack([g, g, g], axis=-1)
    return Image.fromarray(rgb, mode="RGB"), transform, crs, scale


def read_features(path: Path) -> list[dict[str, Any]]:
    """Читает FeatureCollection, возвращает список фич."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = raw.get("features", [])
    if not isinstance(feats, list):
        raise ValueError(f"{path}: нет массива features")
    return feats


def _class_of(props: dict[str, Any]) -> str:
    """Имя класса: class_name (регламент) или class (внутренний)."""
    for key in ("class_name", "class", "cls"):
        val = props.get(key)
        if isinstance(val, str) and val:
            return val
    return "object_poly"


def _rings(geom: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Геометрия -> список колец мировых координат."""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Polygon":
        return [[(float(x), float(y)) for x, y in ring] for ring in coords]
    if gtype == "MultiPolygon":
        return [[(float(x), float(y)) for x, y in ring] for poly in coords for ring in poly]
    if gtype == "Point":
        x, y = float(coords[0]), float(coords[1])
        return [[(x, y)]]
    logger.warning("пропущена геометрия типа %s", gtype)
    return []


def render_overlay(
    base_path: Path,
    geojson_path: Path,
    out_path: Path,
    *,
    src_crs: str = SUBMISSION_CRS,
    palette: dict[str, str] | None = None,
    alpha: float = 0.35,
    width: int = 3,
    point_radius: int = 6,
    legend: bool = True,
    max_side: int = 4096,
) -> Path:
    """Рисует объекты поверх подложки и сохраняет PNG.

    Args:
        base_path: GeoTIFF подложки.
        geojson_path: детекции (обычно EPSG:3857).
        out_path: куда сохранить PNG.
        src_crs: CRS геометрий в geojson.
        palette: класс -> '#rrggbb'.
        alpha: прозрачность заливки полигонов.
        width: толщина контура в пикселях.
        point_radius: радиус маркера для Point.
        legend: рисовать легенду.
        max_side: ограничение размера подложки.

    Returns:
        Путь сохранённого файла.
    """
    palette = palette or PALETTE
    img, transform, base_crs, _ = load_base(base_path, max_side=max_side)
    if not base_crs:
        raise ValueError(f"{base_path}: у подложки нет CRS, выравнивание невозможно")

    tf = Transformer.from_crs(src_crs, base_crs, always_xy=True)
    inv = ~transform

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill_a = int(max(0.0, min(alpha, 1.0)) * 255)

    seen: dict[str, int] = {}
    drawn = 0
    for feat in read_features(geojson_path):
        props = feat.get("properties") or {}
        cls = _class_of(props)
        rgb = color_for(cls, palette)
        for ring in _rings(feat.get("geometry") or {}):
            if not ring:
                continue
            xs, ys = zip(*ring, strict=True)
            wx, wy = tf.transform(xs, ys)
            px = [inv * (x, y) for x, y in zip(wx, wy, strict=True)]
            if len(px) == 1:
                cx, cy = px[0]
                draw.ellipse(
                    [
                        cx - point_radius,
                        cy - point_radius,
                        cx + point_radius,
                        cy + point_radius,
                    ],  # fmt: skip
                    fill=(*rgb, fill_a),
                    outline=(*rgb, 255),
                    width=width,
                )
            else:
                draw.polygon(px, fill=(*rgb, fill_a), outline=(*rgb, 255))
                draw.line([*px, px[0]], fill=(*rgb, 255), width=width)
            drawn += 1
        seen[cls] = seen.get(cls, 0) + 1

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    if legend and seen:
        _draw_legend(out, seen, palette)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(out_path, format="PNG", optimize=True)
    logger.info("отрисовано колец: %d, классов: %d -> %s", drawn, len(seen), out_path)
    return out_path


def _draw_legend(img: Image.Image, counts: dict[str, int], palette: dict[str, str]) -> None:
    """Легенда с цветом, именем класса и числом объектов."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    pad, box, step = 12, 18, 26
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    w = 300
    h = pad * 2 + step * len(rows)
    draw.rectangle([pad, pad, pad + w, pad + h], fill=(255, 255, 255, 220))
    y = pad * 2
    for cls, n in rows:
        rgb = color_for(cls, palette)
        draw.rectangle([pad * 2, y, pad * 2 + box, y + box], fill=(*rgb, 255))
        draw.text((pad * 2 + box + 8, y), f"{cls}  ({n})", fill=(20, 20, 20), font=font)
        y += step
