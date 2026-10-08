#!/usr/bin/env python
"""Утилита командной строки Django."""

import os
import sys


def main():
    """Точка входа для manage.py."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Не удалось импортировать Django. Убедитесь, что он установлен и "
            "доступен в PYTHONPATH. Возможно, забыли активировать окружение?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
