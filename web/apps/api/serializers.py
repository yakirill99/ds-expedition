"""Сериализаторы API."""

from rest_framework import serializers


class PredictRequestSerializer(serializers.Serializer):
    """Пример сериализатора запроса на инференс."""

    config_path = serializers.CharField(required=True)
    site_id = serializers.CharField(required=True)
