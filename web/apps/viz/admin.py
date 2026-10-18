"""Администрирование моделей визуализации."""

from django.contrib import admin

from .models import Location, MapLayer


class MapLayerInline(admin.TabularInline):
    """Редактирование слоёв внутри локации."""

    model = MapLayer
    extra = 1


@admin.register(Location)
class LocationAdmin(admin.ModelAdmin):
    """Админка для локаций."""

    list_display = ["name", "slug", "center_lat", "center_lon", "zoom", "updated_at"]
    search_fields = ["name", "slug"]
    prepopulated_fields = {"slug": ("name",)}
    inlines = [MapLayerInline]


@admin.register(MapLayer)
class MapLayerAdmin(admin.ModelAdmin):
    """Админка для слоёв карты."""

    list_display = ["name", "location", "layer_type", "order", "is_visible"]
    list_filter = ["layer_type", "location"]
    search_fields = ["name", "slug"]
