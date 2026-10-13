"""Импорт локации с тайлами в хранилище карт."""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils.text import slugify

from web.apps.viz.models import Location, MapLayer
from web.apps.viz.storage import TilePath, get_map_storage


class Command(BaseCommand):
    """Создать или обновить локацию по локальной папке с тайлами.

    Ожидаемая структура исходных данных:
        <tiles-dir>/<location_slug>/<layer_slug>/<z>/<x>/<y>.<ext>

    Пример:
        python web/manage.py import_location \\
            --name "Участок 1" --slug uchastok-1 \\
            --tiles-dir data/tiles --lat 55.75 --lon 37.61 --zoom 14
    """

    help = "Импортировать локацию с тайлами из локальной папки в хранилище карт"

    def add_arguments(self, parser) -> None:
        parser.add_argument("--name", required=True, help="Название локации")
        parser.add_argument("--slug", default=None, help="Код локации (по умолчанию из названия)")
        parser.add_argument("--tiles-dir", required=True, type=Path, help="Корень с тайлами")
        parser.add_argument("--lat", required=True, type=float, help="Широта центра")
        parser.add_argument("--lon", required=True, type=float, help="Долгота центра")
        parser.add_argument("--zoom", type=int, default=14, help="Начальный зум")
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
        tiles_dir: Path = options["tiles_dir"]
        slug = options["slug"] or slugify(options["name"])
        storage = get_map_storage()

        location_dir = tiles_dir / slug
        if not location_dir.is_dir():
            raise CommandError(f"Не найдена папка локации: {location_dir}")

        # Слои — это подпапки локации второго уровня.
        layer_slugs = sorted(item.name for item in location_dir.iterdir() if item.is_dir())
        if not layer_slugs:
            raise CommandError(f"В папке {location_dir} нет ни одного слоя")

        if options["dry_run"]:
            self.stdout.write(f"Будет импортирована локация '{options['name']}' ({slug}):")
            for layer_slug in layer_slugs:
                tiles = sum(1 for _ in (location_dir / layer_slug).rglob("*.png"))
                self.stdout.write(f"  - слой {layer_slug}: {tiles} тайлов")
            return

        location, created = Location.objects.update_or_create(
            slug=slug,
            defaults={
                "name": options["name"],
                "center_lat": options["lat"],
                "center_lon": options["lon"],
                "zoom": options["zoom"],
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
