"""Представления для визуализации результатов."""

from django.http import JsonResponse
from django.views.decorators.http import require_GET


@require_GET
def index(request):
    """Заглушка главной страницы визуализации."""
    return JsonResponse({"detail": "viz index"})
