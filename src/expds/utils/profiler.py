"""Профилировщик стадий пайплайна.

Простой контекстный менеджер для замера времени. Пишет в лог начало,
конец и длительность. В конце можно получить сводную таблицу.

Использование:
    profiler = Profiler()
    with profiler.stage("чтение LAS"):
        points = read_points(...)
    with profiler.stage("фильтр земли"):
        ground = filter_ground(...)
    profiler.report()
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class StageRecord:
    """Запись о стадии: имя, длительность, статус."""

    name: str
    duration_sec: float
    success: bool


@dataclass
class Profiler:
    """Профилировщик стадий.

    Хранит список записей. Каждая стадия — через context manager.
    Если стадия падает с исключением, она записывается как неуспешная
    и исключение пробрасывается дальше.
    """

    _records: list[StageRecord] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Контекстный менеджер для замера времени стадии.

        Args:
            name: имя стадии для лога.

        Yields:
            None.

        Raises:
            Любое исключение из тела — записывается как неуспешная стадия
            и пробрасывается дальше.
        """
        logger.info("▶ Начало стадии: %s", name)
        start = time.perf_counter()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            self._records.append(StageRecord(name=name, duration_sec=duration, success=success))
            status = "OK" if success else "FAIL"
            logger.info(
                "■ Конец стадии: %s | %.2f сек | %s",
                name,
                duration,
                status,
            )

    @property
    def records(self) -> list[StageRecord]:
        """Список записей о стадиях."""
        return list(self._records)

    def total_sec(self) -> float:
        """Суммарное время всех стадий."""
        return sum(r.duration_sec for r in self._records)

    def report(self) -> str:
        """Формирует сводную таблицу.

        Returns:
            Многострочная строка с таблицей.
        """
        if not self._records:
            return "Профилировщик: нет записей"

        lines = [
            "",
            "=" * 60,
            "ПРОФИЛЬ СТАДИЙ",
            "=" * 60,
            f"{'Стадия':<40} {'Время, сек':>10} {'Статус':>8}",
            "-" * 60,
        ]
        for record in self._records:
            status = "OK" if record.success else "FAIL"
            lines.append(f"{record.name:<40} {record.duration_sec:>10.2f} {status:>8}")
        lines.append("-" * 60)
        lines.append(f"{'ИТОГО':<40} {self.total_sec():>10.2f}")
        lines.append("=" * 60)
        result = "\n".join(lines)
        logger.info(result)
        return result

    def reset(self) -> None:
        """Очищает записи."""
        self._records.clear()
