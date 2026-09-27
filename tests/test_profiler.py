"""Тесты профилировщика."""

from __future__ import annotations

import logging
import time

import pytest

from expds.utils.profiler import Profiler


class TestProfiler:
    """Тесты Profiler."""

    def test_single_stage_records(self) -> None:
        profiler = Profiler()
        with profiler.stage("test"):
            time.sleep(0.01)
        assert len(profiler.records) == 1
        record = profiler.records[0]
        assert record.name == "test"
        assert record.duration_sec >= 0.01
        assert record.success is True

    def test_multiple_stages(self) -> None:
        profiler = Profiler()
        with profiler.stage("a"):
            pass
        with profiler.stage("b"):
            pass
        assert len(profiler.records) == 2
        assert [r.name for r in profiler.records] == ["a", "b"]

    def test_failed_stage_recorded(self) -> None:
        profiler = Profiler()
        with pytest.raises(ValueError):
            with profiler.stage("failing"):
                raise ValueError("boom")
        assert len(profiler.records) == 1
        assert profiler.records[0].success is False
        assert profiler.records[0].name == "failing"

    def test_total_sec(self) -> None:
        profiler = Profiler()
        with profiler.stage("a"):
            time.sleep(0.01)
        with profiler.stage("b"):
            time.sleep(0.02)
        assert profiler.total_sec() >= 0.03

    def test_report_format(self) -> None:
        profiler = Profiler()
        with profiler.stage("a"):
            pass
        report = profiler.report()
        assert "ПРОФИЛЬ СТАДИЙ" in report
        assert "a" in report
        assert "ИТОГО" in report

    def test_empty_report(self) -> None:
        profiler = Profiler()
        report = profiler.report()
        assert "нет записей" in report

    def test_reset(self) -> None:
        profiler = Profiler()
        with profiler.stage("a"):
            pass
        assert len(profiler.records) == 1
        profiler.reset()
        assert len(profiler.records) == 0

    def test_stage_logs(self, caplog) -> None:
        profiler = Profiler()
        with caplog.at_level(logging.INFO):
            with profiler.stage("test_stage"):
                pass
        assert "test_stage" in caplog.text
        assert "Начало" in caplog.text
        assert "Конец" in caplog.text
