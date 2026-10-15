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


def _discover_layers(location_dir: Path) -> list[str]:
    """Найти папки слоёв тайлов внутри локации.

    Поддерживаются две структуры:
        <location_dir>/<layer>/{z}/{x}/{y}.png
        <location_dir>/tiles/<layer>/{z}/{x}/{y}.png
    """
    tiles_dir = location_dir / "tiles"
    search_root = tiles_dir if tiles_dir.is_dir() else location_dir
    return sorted(
        item.name
        for item in search_root.iterdir()
        if item.is_dir() and not item.name.startswith(("_", "."))
    )


class Command(BaseCommand):
    """Создать или обновить локацию по локальной папке с тайлами.

    Ожидаемая структура исходных данных (любой из вариантов):

        <tiles-dir>/<slug>/<layer>/<z>/<x>/<y>.<ext>
        <tiles-dir>/<slug>/tiles/<layer>/<z>/<x>/<y>.<ext>

    Если в папке локации есть ``manifest.json``, название, центр и зум
    берутся оттуда. Иначе их нужно передать явно через аргументы.

    Пример:
        python web/manage.py import_location \\
            --tiles-dir data \\
            --slug 001-gnezdovo-nivlenskoe

        python web/manage.py import_location \\
            --name "Участок 1" --slug uchastok-1 \\
            --tiles-dir data/tiles --lat 55.75 --lon 37.61 --zoom 14
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
            help="Тип слоя по умолчанию для всех импортируемых слоёв",
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
            # Пытаемся угадать slug, если в корне одна папка локации.
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

        layer_slugs = _discover_layers(location_dir)
        if not layer_slugs:
            raise CommandError(f"В папке {location_dir} не найдено слоёв с тайлами")

        if options["dry_run"]:
            self.stdout.write(f"Будет импортирована локация '{name}' ({slug}):")
            for layer_slug in layer_slugs:
                search_root = (
                    (location_dir / "tiles") if (location_dir / "tiles").is_dir() else location_dir
                )
                tiles = sum(1 for _ in (search_root / layer_slug).rglob("*.png"))
                self.stdout.write(f"  - слой {layer_slug}: {tiles} тайлов")
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

        for order, layer_slug in enumerate(layer_slugs):
            MapLayer.objects.update_or_create(
                location=location,
                slug=layer_slug,
                defaults={
                    "name": layer_slug.replace("_", " ").capitalize(),
                    "layer_type": options["layer_type"],
                    "order": order,
                },
            )
            sample = TilePath(location_slug=slug, layer_slug=layer_slug, z=0, x=0, y=0)
            self.stdout.write(f"  - слой {layer_slug} -> {storage.tile_url(sample)}...")
