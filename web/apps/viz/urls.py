"""Маршруты приложения viz."""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="viz-index"),
    path("api/locations/", views.location_list, name="viz-location-list"),
    path(
        "api/locations/<slug:location_slug>/layers/",
        views.location_layers,
        name="viz-location-layers",
    ),
    path(
        "tiles/<slug:location_slug>/<slug:layer_slug>/<int:z>/<int:x>/<path:y>",
        views.tile_serve,
        name="viz-tile",
    ),
]
