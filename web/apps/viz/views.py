"""Представления для визуализации карт и раздачи тайлов."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, HttpResponseNotFound, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET
from pyproj import Transformer

from .models import Location
from .storage import TilePath, get_map_storage


@require_GET
def index(request):
    """Главная страница визуализации с интерактивной картой."""
    return render(request, "viz/map.html")


@require_GET
def tile_serve(request, location_slug: str, layer_slug: str, z: int, x: int, y: str):
    """Раздать один тайл из файлового хранилища.

    Параметр ``y`` передаётся как строка вида ``123.png``; расширение
    определяется динамически. Поддерживаются только локальные бэкенды.
    """
    storage = get_map_storage()
    local_path = storage.local_path(
        TilePath(
            location_slug=location_slug,
            layer_slug=layer_slug,
            z=z,
            x=x,
            y=0,
        )
    )
    if local_path is None:
        return HttpResponseNotFound("Remote storage is not served directly")

    # y приходит как "123.png"; отделяем номер и расширение.
    y_stem, extension = Path(y).stem, Path(y).suffix.lstrip(".") or "png"
    try:
        y_int = int(y_stem)
    except ValueError:
        return HttpResponseNotFound("Invalid tile coordinates")

    tile_path = TilePath(
        location_slug=location_slug,
        layer_slug=layer_slug,
        z=z,
        x=x,
        y=y_int,
        extension=extension,
    )
    file_path = storage.local_path(tile_path)
    if not file_path.is_file():
        return HttpResponseNotFound("Tile not found")

    content_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(open(file_path, "rb"), content_type=content_type or "image/png")


@require_GET
def location_list(request):
    """API: список локаций с базовой информацией."""
    locations = Location.objects.all().order_by("name")
    data = [
        {
            "slug": loc.slug,
            "name": loc.name,
            "center": {"lat": loc.center_lat, "lon": loc.center_lon},
            "zoom": loc.zoom,
        }
        for loc in locations
    ]
    return JsonResponse({"locations": data})


def _raster_tile_url_template(storage, slug: str, layer_slug: str) -> str:
    """Собрать шаблон URL тайлов для растрового слоя."""
    sample = TilePath(location_slug=slug, layer_slug=layer_slug, z=0, x=0, y=0)
    return storage.tile_url(sample).replace("/0/0/0.png", "/{z}/{x}/{y}.png")


@require_GET
def location_layers(request, location_slug: str):
    """API: список слоёв для выбранной локации с публичными URL-шаблонами."""
    location = get_object_or_404(Location, slug=location_slug)
    storage = get_map_storage()

    layers = []
    for layer in location.layers.all():
        item: dict = {
            "slug": layer.slug,
            "name": layer.name,
            "type": layer.layer_type,
            "order": layer.order,
            "is_visible": layer.is_visible,
        }
        if layer.layer_type == "geojson":
            item["geojson_url"] = layer.geojson_url
        else:
            item["tile_url_template"] = _raster_tile_url_template(
                storage, location.slug, layer.slug
            )
        layers.append(item)

    return JsonResponse(
        {
            "location": {
                "slug": location.slug,
                "name": location.name,
                "center": {"lat": location.center_lat, "lon": location.center_lon},
                "zoom": location.zoom,
            },
            "layers": layers,
        }
    )


def _transform_coordinates(obj, transformer):
    """Рекурсивно перепроецировать координаты GeoJSON-объекта."""
    geom_type = obj.get("type")
    coords = obj.get("coordinates")

    if geom_type == "Point":
        obj["coordinates"] = list(transformer.transform(coords[0], coords[1]))
    elif geom_type in ("LineString", "MultiPoint"):
        obj["coordinates"] = [list(transformer.transform(c[0], c[1])) for c in coords]
    elif geom_type in ("Polygon", "MultiLineString"):
        obj["coordinates"] = [
            [list(transformer.transform(c[0], c[1])) for c in ring] for ring in coords
        ]
    elif geom_type == "MultiPolygon":
        obj["coordinates"] = [
            [[list(transformer.transform(c[0], c[1])) for c in ring] for ring in poly]
            for poly in coords
        ]
    elif geom_type == "GeometryCollection":
        for geom in obj.get("geometries", []):
            _transform_coordinates(geom, transformer)


@require_GET
def geojson_serve(request, location_slug: str, filename: str):
    """Раздать GeoJSON-разметку, перепроецированную в WGS84."""
    get_object_or_404(Location, slug=location_slug)

    # GeoJSON лежит в корне папки локации рядом с тайлами.
    location_dir = Path(settings.MAP_STORAGE_ROOT) / location_slug
    file_path = location_dir / filename
    if not file_path.is_file():
        return HttpResponseNotFound("GeoJSON not found")

    data = json.loads(file_path.read_text(encoding="utf-8"))

    crs = _extract_geojson_crs(data) or _crs_from_manifest(location_dir)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    if data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            geom = feature.get("geometry")
            if geom:
                _transform_coordinates(geom, transformer)
    elif "coordinates" in data:
        _transform_coordinates(data, transformer)

    # Убираем исходное поле crs, чтобы Leaflet не путал его с WGS84.
    data.pop("crs", None)

    return JsonResponse(data, json_dumps_params={"ensure_ascii": False})


def _extract_geojson_crs(data: dict) -> str | None:
    """Извлечь EPSG-код из поля ``crs`` GeoJSON."""
    crs_obj = data.get("crs")
    if not crs_obj or crs_obj.get("type") != "name":
        return None
    name = crs_obj.get("properties", {}).get("name", "")
    if "EPSG" in name:
        # Пример: urn:ogc:def:crs:EPSG::3857
        parts = name.split(":")
        if parts:
            try:
                code = int(parts[-1])
                return f"EPSG:{code}"
            except ValueError:
                pass
    return None


def _crs_from_manifest(location_dir: Path) -> str:
    """Вернуть CRS локации из manifest.json или Web Mercator по умолчанию."""
    manifest_path = location_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("crs", "EPSG:3857")
    return "EPSG:3857"
