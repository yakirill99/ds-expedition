"""Модели для хранения состояния ML-задач."""

from django.db import models


class InferenceJob(models.Model):
    """Фоновая задача инференса."""

    class Status(models.TextChoices):
        PENDING = "pending", "В очереди"
        RUNNING = "running", "Выполняется"
        DONE = "done", "Завершена"
        FAILED = "failed", "Ошибка"

    site_id = models.CharField("ID участка", max_length=255)
    config_path = models.CharField("Путь к конфигу", max_length=512)
    status = models.CharField(
        "Статус",
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    created_at = models.DateTimeField("Создана", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлена", auto_now=True)
    result = models.JSONField("Результат", null=True, blank=True, default=dict)

    class Meta:
        verbose_name = "Задача инференса"
        verbose_name_plural = "Задачи инференса"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"InferenceJob({self.site_id}, {self.status})"
