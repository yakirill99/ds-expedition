"""Management-команда для запуска инференса."""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Запускает инференс через expds.infer"

    def add_arguments(self, parser):
        parser.add_argument("--config", required=True, help="Путь к yaml-конфигу")
        parser.add_argument("--site", required=True, help="ID участка")

    def handle(self, *args, **options):
        config_path = options["config"]
        site_id = options["site"]
        self.stdout.write(
            self.style.NOTICE(f"Запуск инференса: config={config_path}, site={site_id}")
        )
        # TODO: импортировать и вызвать expds.infer.predict_probs(...)
        self.stdout.write(self.style.SUCCESS("Инференс завершён (заглушка)"))
