"""Management-команда для запуска обучения."""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Запускает обучение модели через expds.train"

    def add_arguments(self, parser):
        parser.add_argument("--config", required=True, help="Путь к yaml-конфигу")

    def handle(self, *args, **options):
        config_path = options["config"]
        self.stdout.write(self.style.NOTICE(f"Запуск обучения: {config_path}"))
        # TODO: импортировать и вызвать expds.train.train(config_path)
        self.stdout.write(self.style.SUCCESS("Обучение завершено (заглушка)"))
