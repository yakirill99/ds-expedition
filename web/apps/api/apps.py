"""Конфигурация приложения api."""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.apps.api"
    verbose_name = "API"
