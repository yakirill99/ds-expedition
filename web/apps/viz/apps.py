"""Конфигурация приложения viz."""

from django.apps import AppConfig


class VizConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.apps.viz"
    verbose_name = "Визуализация"
