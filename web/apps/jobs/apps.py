"""Конфигурация приложения jobs."""

from django.apps import AppConfig


class JobsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.apps.jobs"
    verbose_name = "ML-задачи"
