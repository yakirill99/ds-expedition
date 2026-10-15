"""Представления для визуализации карт и раздачи тайлов."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from django.http import FileResponse, HttpResponseNotFound, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET

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


@require_GET
def location_layers(request, location_slug: str):
    """API: список слоёв для выбранной локации с публичными URL-шаблонами."""
    location = get_object_or_404(Location, slug=location_slug)
    storage = get_map_storage()

    layers = []
    for layer in location.layers.all():
        sample = TilePath(
            location_slug=location.slug,
            layer_slug=layer.slug,
            z=0,
            x=0,
            y=0,
        )
        # Заменяем фиктивные 0/0/0 на шаблоны Leaflet: {z}/{x}/{y}.png
        url_template = storage.tile_url(sample).replace("/0/0/0.png", "/{z}/{x}/{y}.png")
        layers.append(
            {
                "slug": layer.slug,
                "name": layer.name,
                "type": layer.layer_type,
                "order": layer.order,
                "is_visible": layer.is_visible,
                "tile_url_template": url_template,
            }
        )

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
