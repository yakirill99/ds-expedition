"""Маршруты приложения api."""

from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("predict/", views.predict, name="predict"),
]
