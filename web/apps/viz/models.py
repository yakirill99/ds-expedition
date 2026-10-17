"""Модели приложения viz."""

from django.db import models


class Location(models.Model):
    """Географическая локация (участок карты)."""

    name = models.CharField("Название локации", max_length=200)
    slug = models.SlugField("Код локации", unique=True, db_index=True)
    description = models.TextField("Описание", blank=True)

    # Начальная точка обзора карты
    center_lat = models.FloatField("Широта центра")
    center_lon = models.FloatField("Долгота центра")
    zoom = models.PositiveSmallIntegerField("Начальный зум", default=14)

    created_at = models.DateTimeField("Создано", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        verbose_name = "Локация"
        verbose_name_plural = "Локации"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class MapLayer(models.Model):
    """Слой карты для локации (спутник, лидар, аэрофотосъёмка и т.д.)."""

    LAYER_TYPES = [
        ("satellite", "Спутниковый снимок"),
        ("lidar", "Карта высот (LiDAR)"),
        ("aerial", "Аэрофотосъёмка"),
        ("geojson", "Векторная разметка"),
        ("other", "Другое"),
    ]

    location = models.ForeignKey(
        Location,
        on_delete=models.CASCADE,
        related_name="layers",
        verbose_name="Локация",
    )
    name = models.CharField("Название слоя", max_length=100)
    slug = models.SlugField("Код слоя")
    layer_type = models.CharField("Тип слоя", max_length=20, choices=LAYER_TYPES, default="other")

    # Порядок отображения в переключателе слоёв
    order = models.PositiveSmallIntegerField("Порядок", default=0)
    is_visible = models.BooleanField("Видим по умолчанию", default=True)

    # Для векторных слоёв (GeoJSON)
    geojson_url = models.CharField("URL GeoJSON", max_length=500, blank=True)

    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "Слой карты"
        verbose_name_plural = "Слои карты"
        ordering = ["location", "order", "name"]
        unique_together = ["location", "slug"]

    def __str__(self) -> str:
        return f"{self.location.name} / {self.name}"
