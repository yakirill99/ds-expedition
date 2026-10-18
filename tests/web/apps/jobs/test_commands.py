"""Тесты management-команд."""

import io

from django.core.management import call_command


def test_train_command_stub():
    out = io.StringIO()
    call_command("train", "--config", "configs/baseline.yaml", stdout=out)
    assert "Запуск обучения" in out.getvalue()


def test_predict_command_stub():
    out = io.StringIO()
    call_command(
        "predict",
        "--config",
        "configs/baseline.yaml",
        "--site",
        "site_001",
        stdout=out,
    )
    assert "Запуск инференса" in out.getvalue()
