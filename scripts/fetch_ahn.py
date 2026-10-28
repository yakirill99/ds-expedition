"""Скачивание вырезки AHN (DTM/DSM 0.5 м) как GeoTIFF для участка-вырезки.

Источник — официальный ArcGIS ImageServer AHN (Esri Nederland), без авторизации:

    https://ahn.arcgisonline.nl/arcgis/rest/services/Hoogtebestand/AHN5_DTM_50cm

Примеры:

    # вырезка 512x512 м вокруг точки (EPSG:28992, RD New — метры):
    uv run python scripts/fetch_ahn.py --site ahn_vaals_1 \
        --center 180256 318256 --size-m 512

    # прямоугольником xmin,ymin,xmax,ymax (EPSG:28992):
    uv run python scripts/fetch_ahn.py --site ahn_vaals_1 \
        --bbox 180000 318000 180512 318512

    # DSM вторым слоем (опционально) и версия AHN4:
    uv run python scripts/fetch_ahn.py --site ahn_vaals_1 --bbox ... --dsm --ahn4

Результат: data/ahn/raw/<site>/dtm.tif (+ dsm.tif) в нативном CRS AHN
EPSG:28992, 0.5 м/px, float32 — дальше обычный костяк:

    uv run python scripts/build_layers.py --config configs/dataset_ahn.yaml
    uv run python scripts/inventory.py --config configs/dataset_ahn.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

#: База ArcGIS ImageServer AHN (Esri Nederland, open data).
AHN_BASE = "https://ahn.arcgisonline.nl/arcgis/rest/services/Hoogtebestand"

#: Версия AHN -> ImageServer для DTM/DSM на 0.5 м.
SERVICES: dict[str, dict[str, str]] = {
    "5": {"dtm": "AHN5_DTM_50cm", "dsm": "AHN5_DSM_50cm"},
    "4": {"dtm": "AHN4_DTM_50cm", "dsm": "AHN4_DSM_50cm"},
}

#: Нативное разрешение сервисов AHN *_50cm, м/px.
PIXEL_SIZE = 0.5

#: EPSG:28992 (Amersfoort / RD New) — рабочая CRS всех сервисов AHN.
RD_NEW = "EPSG:28992"

#: Защита от слишком больших запросов (лимиты ImageServer ~4100 px по оси).
DEFAULT_MAX_PX = 4096

_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")


def resolve_service(ahn: str, dataset: str) -> str:
    """Имя ImageServer по версии AHN и типу поверхности.

    Args:
        ahn: "5" или "4".
        dataset: "dtm" или "dsm".

    Returns:
        Имя сервиса, например "AHN5_DTM_50cm".

    Raises:
        ValueError: неизвестная версия или тип.
    """
    if ahn not in SERVICES:
        raise ValueError(f"версия AHN должна быть одной из {sorted(SERVICES)}, получено {ahn!r}")
    if dataset not in SERVICES[ahn]:
        raise ValueError(
            f"dataset должен быть одной из {sorted(SERVICES[ahn])}, получено {dataset!r}"
        )
    return SERVICES[ahn][dataset]


def parse_bbox(args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Бокс участка (xmin, ymin, xmax, ymax) в EPSG:28992 из --bbox или --center.

    Raises:
        ValueError: оба/ни один способ задан или бокс вырожден.
    """
    if (args.bbox is None) == (args.center is None):
        raise ValueError(
            "задайте ровно один способ: --bbox xmin ymin xmax ymax ИЛИ --center x y --size-m"
        )
    if args.bbox is not None:
        if len(args.bbox) != 4:
            raise ValueError("--bbox ожидает 4 числа: xmin ymin xmax ymax")
        xmin, ymin, xmax, ymax = (float(v) for v in args.bbox)
    else:
        if args.size_m is None or args.size_m <= 0:
            raise ValueError("--center требует --size-m (сторона вырезки в метрах, например 512)")
        cx, cy = (float(v) for v in args.center)
        half = float(args.size_m) / 2.0
        xmin, ymin, xmax, ymax = cx - half, cy - half, cx + half, cy + half
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"вырожденный бокс: ({xmin}, {ymin}, {xmax}, {ymax})")
    return xmin, ymin, xmax, ymax


def build_export_url(
    service: str,
    bbox: tuple[float, float, float, float],
    max_px: int = DEFAULT_MAX_PX,
) -> tuple[str, int, int]:
    """URL exportImage для бокса + размер в пикселях.

    Args:
        service: имя ImageServer (см. resolve_service).
        bbox: (xmin, ymin, xmax, ymax) в EPSG:28992.
        max_px: максимум пикселей по оси (защита от лимитов сервера).

    Returns:
        (url, width_px, height_px).

    Raises:
        ValueError: вырезка больше max_px по любой оси.
    """
    xmin, ymin, xmax, ymax = bbox
    width_m, height_m = xmax - xmin, ymax - ymin
    width_px = round(width_m / PIXEL_SIZE)
    height_px = round(height_m / PIXEL_SIZE)
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"вырезка меньше пикселя: {width_px}x{height_px}")
    if width_px > max_px or height_px > max_px:
        raise ValueError(
            f"вырезка {width_px}x{height_px} px больше лимита {max_px} px — "
            f"уменьшите бокс (при 0.5 м/px лимит {max_px} px = {max_px * PIXEL_SIZE:.0f} м по оси)"
        )
    params = urllib.parse.urlencode(
        {
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "bboxSR": "28992",
            "imageSR": "28992",
            "size": f"{width_px},{height_px}",
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_NearestNeighbor",
            "f": "image",
        }
    )
    return f"{AHN_BASE}/{service}/ImageServer/exportImage?{params}", width_px, height_px


def download_tif(url: str, dst: Path, timeout: float = 120.0, force: bool = False) -> Path:
    """Скачивает GeoTIFF в dst, проверяя TIFF-магию ответа.

    Raises:
        RuntimeError: сервер вернул не-GeoTIFF (ошибка сервиса, лимит и т.п.).
    """
    if dst.exists() and not force:
        raise RuntimeError(f"{dst} уже есть — укажите --force для перезаписи")
    request = urllib.request.Request(url, headers={"User-Agent": "ds-expedition-fetch-ahn/1"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 — фиксированный хост
        payload = resp.read()
    if not payload.startswith(_TIFF_MAGIC):
        head = payload[:300].decode("utf-8", "replace")
        raise RuntimeError(f"сервер вернул не GeoTIFF (начало ответа): {head}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    logger.info("Записан %s (%.1f КБ)", dst, len(payload) / 1024)
    return dst


def _describe(path: Path) -> str:
    """Краткое описание скачанного растра (crs/размер/диапазон Z)."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        valid = data[data != nodata]
        z = f"Z=[{float(valid.min()):.2f}, {float(valid.max()):.2f}]" if valid.size else "Z=пусто"
        return f"{path.name}: {src.crs} | {src.width}x{src.height} | {src.dtypes[0]} | {z}"


def main(argv: list[str] | None = None) -> int:
    """CLI: скачать вырезку AHN в data/ahn/raw/<site>."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--site", required=True, help="имя участка: подкаталог в --sites-dir")
    ap.add_argument("--sites-dir", type=Path, default=Path("data/ahn/raw"), help="корень участков")
    ap.add_argument(
        "--bbox", nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="бокс в EPSG:28992"
    )
    ap.add_argument("--center", nargs=2, metavar=("X", "Y"), help="центр вырезки в EPSG:28992")
    ap.add_argument("--size-m", type=float, help="сторона вырезки в метрах (для --center)")
    ap.add_argument("--dsm", action="store_true", help="скачать ещё и DSM (dsm.tif)")
    ap.add_argument("--ahn", choices=sorted(SERVICES), default="5", help="версия AHN")
    ap.add_argument("--max-px", type=int, default=DEFAULT_MAX_PX, help="лимит пикселей по оси")
    ap.add_argument("--timeout", type=float, default=120.0, help="таймаут скачивания, сек")
    ap.add_argument(
        "--force", action="store_true", help="перезаписать существующие dtm.tif/dsm.tif"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        bbox = parse_bbox(args)
    except ValueError as exc:
        ap.error(str(exc))
        return 2

    site_dir = args.sites_dir / args.site
    downloads: list[tuple[str, Path]] = [("dtm", site_dir / "dtm.tif")]
    if args.dsm:
        downloads.append(("dsm", site_dir / "dsm.tif"))

    try:
        for dataset, dst in downloads:
            service = resolve_service(args.ahn, dataset)
            url, width_px, height_px = build_export_url(service, bbox, args.max_px)
            logger.info("%s -> %s (%dx%d px)", service, dst, width_px, height_px)
            download_tif(url, dst, timeout=args.timeout, force=args.force)
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    site_meta = {
        "source": "ahn.arcgisonline.nl ImageServer",
        "service": resolve_service(args.ahn, "dtm"),
        "crs": RD_NEW,
        "pixel_size": PIXEL_SIZE,
        "bbox_rd_new": list(bbox),
        "ahn_version": args.ahn,
    }
    (site_dir / "ahn_meta.json").write_text(json.dumps(site_meta, indent=2), encoding="utf-8")

    for _, dst in downloads:
        print(_describe(dst))
    print("Дальше: uv run python scripts/build_layers.py --config configs/dataset_ahn.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
