"""API-представления для ML-инференса."""

from __future__ import annotations

from drf_spectacular.utils import extend_schema
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from .serializers import PredictRequestSerializer


@extend_schema(
    summary="Проверка работоспособности",
    responses={200: {"type": "object", "properties": {"status": {"type": "string"}}}},
)
@api_view(["GET"])
def health(request: Request) -> Response:
    """Проверка работоспособности сервиса."""
    return Response({"status": "ok"})


@extend_schema(
    summary="Запуск инференса",
    request=PredictRequestSerializer,
    responses={501: {"type": "object"}},
)
@api_view(["POST"])
def predict(request: Request) -> Response:
    """Заглушка для запуска инференса.

    TODO: подключить expds.infer.predict_probs или собственную функцию-обёртку,
    принимающую параметры из request.data.
    """
    return Response(
        {
            "detail": "predict endpoint is not implemented yet",
            "received": request.data,
        },
        status=501,
    )
