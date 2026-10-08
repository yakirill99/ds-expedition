"""Маршруты приложения viz."""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="viz-index"),
]
