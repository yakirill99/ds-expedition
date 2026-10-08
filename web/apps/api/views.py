"""API-представления для ML-инференса."""

from __future__ import annotations

from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response


@api_view(["GET"])
def health(request: Request) -> Response:
    """Проверка работоспособности сервиса."""
    return Response({"status": "ok"})


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
