"""Импорт локации с тайлами в хранилище карт."""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils.text import slugify

from web.apps.viz.models import Location, MapLayer
from web.apps.viz.storage import TilePath, get_map_storage


def _load_manifest(location_dir: Path) -> dict:
    """Прочитать manifest.json локации, если он есть."""
    manifest_path = location_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _discover_tile_layers(location_dir: Path) -> list[str]:
    """Найти папки слоёв тайлов внутри локации."""
    tiles_dir = location_dir / "tiles"
    search_root = tiles_dir if tiles_dir.is_dir() else location_dir
    return sorted(
        item.name
        for item in search_root.iterdir()
        if item.is_dir() and not item.name.startswith(("_", "."))
    )


def _discover_zoom_range(location_dir: Path, layer_slug: str) -> tuple[int | None, int | None]:
    """Определить минимальный и максимальный зум тайлов слоя.

    Сканирует папку ``{location_dir}/{layer_slug}/{z}/`` и возвращает
    минимальный/максимальный уровень масштабирования среди числовых папок.
    """
    tiles_dir = location_dir / "tiles"
    search_root = tiles_dir if tiles_dir.is_dir() else location_dir
    layer_dir = search_root / layer_slug
    if not layer_dir.is_dir():
        return None, None

    zooms = [
        int(item.name) for item in layer_dir.iterdir() if item.is_dir() and item.name.isdigit()
    ]
    if not zooms:
        return None, None
    return min(zooms), max(zooms)


def _discover_geojson_layers(location_dir: Path) -> list[tuple[str, str]]:
    """Найти GeoJSON-файлы разметки в корне папки локации.

    Возвращает список пар (layer_slug, filename).
    """
    files = sorted(
        item
        for item in location_dir.iterdir()
        if item.is_file() and item.suffix.lower() == ".geojson"
    )
    layers = []
    for f in files:
        base = f.stem
        # Для файлов вида annotation_...geojson убираем префикс
        layer_slug = slugify(base) or "annotation"
        layers.append((layer_slug, f.name))
    return layers


class Command(BaseCommand):
    """Создать или обновить локацию по локальной папке с тайлами.

    Ожидаемая структура исходных данных:

        <tiles-dir>/<slug>/<layer>/{z}/{x}/{y}.png
        <tiles-dir>/<slug>/annotation_<name>.geojson

    Если в папке локации есть ``manifest.json``, название, центр и зум
    берутся оттуда. Иначе их нужно передать явно через аргументы.

    Пример:
        python web/manage.py import_location \\
            --tiles-dir web/tiles \\
            --slug 001-gnezdovo-nivlenskoe
    """

    help = "Импортировать локацию с тайлами из локальной папки в хранилище карт"

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--tiles-dir", required=True, type=Path, help="Корень с папками локаций"
        )
        parser.add_argument(
            "--slug", default=None, help="Код локации (по умолчанию из manifest.json)"
        )
        parser.add_argument("--name", default=None, help="Название локации")
        parser.add_argument("--lat", type=float, default=None, help="Широта центра")
        parser.add_argument("--lon", type=float, default=None, help="Долгота центра")
        parser.add_argument("--zoom", type=int, default=None, help="Начальный зум")
        parser.add_argument(
            "--layer-type",
            default="other",
            choices=[choice for choice, _ in MapLayer.LAYER_TYPES],
            help="Тип слоя по умолчанию для растровых слоёв",
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Только показать, что будет импортировано"
        )

    def handle(self, *args, **options) -> None:
        tiles_dir = Path(options["tiles_dir"])
        storage = get_map_storage()

        # Определяем slug: аргумент > manifest > ошибка.
        slug = options["slug"]
        if slug is None:
            location_dirs = [item for item in tiles_dir.iterdir() if item.is_dir()]
            if len(location_dirs) == 1:
                slug = location_dirs[0].name
            else:
                raise CommandError(
                    "Не удалось определить slug локации. "
                    "Передайте --slug или укажите папку с одной локацией."
                )

        location_dir = tiles_dir / slug
        manifest = _load_manifest(location_dir)
        slug = manifest.get("slug", slugify(slug))
        name = options["name"] or manifest.get("name") or slug
        center_lat = (
            options["lat"] if options["lat"] is not None else manifest.get("center", {}).get("lat")
        )
        center_lon = (
            options["lon"] if options["lon"] is not None else manifest.get("center", {}).get("lon")
        )
        zoom = options["zoom"] if options["zoom"] is not None else manifest.get("zoom")

        if center_lat is None or center_lon is None or zoom is None:
            raise CommandError(
                "Для локации не задан центр/зум. "
                "Добавьте manifest.json или передайте --lat, --lon, --zoom."
            )

        if not location_dir.is_dir():
            raise CommandError(f"Не найдена папка локации: {location_dir}")

        tile_layer_slugs = _discover_tile_layers(location_dir)
        geojson_layers = _discover_geojson_layers(location_dir)

        if not tile_layer_slugs and not geojson_layers:
            raise CommandError(f"В папке {location_dir} не найдено слоёв")

        if options["dry_run"]:
            self.stdout.write(f"Будет импортирована локация '{name}' ({slug}):")
            for layer_slug in tile_layer_slugs:
                search_root = (
                    (location_dir / "tiles") if (location_dir / "tiles").is_dir() else location_dir
                )
                tiles = sum(1 for _ in (search_root / layer_slug).rglob("*.png"))
                min_zoom, max_zoom = _discover_zoom_range(location_dir, layer_slug)
                zoom_info = f" (зум {min_zoom}-{max_zoom})" if min_zoom is not None else ""
                self.stdout.write(f"  - слой {layer_slug}: {tiles} тайлов{zoom_info}")
            for layer_slug, filename in geojson_layers:
                self.stdout.write(f"  - векторный слой {layer_slug}: {filename}")
            return

        location, created = Location.objects.update_or_create(
            slug=slug,
            defaults={
                "name": name,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "zoom": zoom,
            },
        )
        self.stdout.write(
            self.style.SUCCESS(f"Локация {'создана' if created else 'обновлена'}: {location}")
        )

        order = 0
        for layer_slug in tile_layer_slugs:
            min_zoom, max_zoom = _discover_zoom_range(location_dir, layer_slug)
            MapLayer.objects.update_or_create(
                location=location,
                slug=layer_slug,
                defaults={
                    "name": layer_slug.replace("_", " ").capitalize(),
                    "layer_type": options["layer_type"],
                    "order": order,
                    "geojson_url": "",
                    "min_zoom": min_zoom,
                    "max_zoom": max_zoom,
                },
            )
            sample = TilePath(location_slug=slug, layer_slug=layer_slug, z=0, x=0, y=0)
            zoom_info = f" (зум {min_zoom}-{max_zoom})" if min_zoom is not None else ""
            self.stdout.write(f"  - слой {layer_slug}{zoom_info} -> {storage.tile_url(sample)}...")
            order += 1

        for idx, (_, filename) in enumerate(geojson_layers, start=1):
            layer_slug = f"annotation_{idx}"
            name = filename.replace("annotation_", "").replace(".geojson", "").replace("_", " ")
            MapLayer.objects.update_or_create(
                location=location,
                slug=layer_slug,
                defaults={
                    "name": name,
                    "layer_type": "geojson",
                    "order": order,
                    "geojson_url": f"/viz/geojson/{slug}/{filename}",
                },
            )
            self.stdout.write(
                f"  - векторный слой {layer_slug} ({name}) -> /viz/geojson/{slug}/{filename}"
            )
            order += 1
