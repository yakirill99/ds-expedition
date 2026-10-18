"""Админка для ML-задач."""

from django.contrib import admin

from .models import InferenceJob


@admin.register(InferenceJob)
class InferenceJobAdmin(admin.ModelAdmin):
    list_display = ("id", "site_id", "status", "created_at", "updated_at")
    list_filter = ("status",)
    search_fields = ("site_id", "config_path")
