"""Начальная миграция для приложения jobs."""

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="InferenceJob",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("site_id", models.CharField(max_length=255, verbose_name="ID участка")),
                (
                    "config_path",
                    models.CharField(max_length=512, verbose_name="Путь к конфигу"),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "В очереди"),
                            ("running", "Выполняется"),
                            ("done", "Завершена"),
                            ("failed", "Ошибка"),
                        ],
                        default="pending",
                        max_length=20,
                        verbose_name="Статус",
                    ),
                ),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, verbose_name="Создана"),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, verbose_name="Обновлена"),
                ),
                (
                    "result",
                    models.JSONField(
                        blank=True,
                        default=dict,
                        null=True,
                        verbose_name="Результат",
                    ),
                ),
            ],
            options={
                "verbose_name": "Задача инференса",
                "verbose_name_plural": "Задачи инференса",
                "ordering": ["-created_at"],
            },
        ),
    ]
