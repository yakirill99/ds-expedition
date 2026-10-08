"""Базовые проверки настроек Django."""


def test_settings_import():
    """Настройки должны импортироваться без ошибок."""
    from web.config import settings

    assert settings.ROOT_URLCONF == "web.config.urls"
