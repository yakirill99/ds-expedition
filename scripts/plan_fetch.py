"""Что копировать локально, если места меньше, чем весь датасет.

Сортирует участки по плотности разметки (объектов на гигабайт) и набирает,
пока не упрётся в бюджет. Участки, которые уже есть в data/raw, бесплатны
и попадают в план всегда.

    uv run python scripts/plan_fetch.py --budget-gb 50
    uv run python scripts/plan_fetch.py --budget-gb 50 --exclude 093,187,190
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ranges(nums: list[int]) -> str:
    """[14,15,16,30] -> '014-016 030' — для fetch_sites.sh."""
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]:03d}" if i == j else f"{nums[i]:03d}-{nums[j]:03d}")
        i = j + 1
    return " ".join(out)


def main() -> int:
    """Точка входа."""
    ap = argparse.ArgumentParser(description="План копирования под бюджет места")
    ap.add_argument("--survey", type=Path, default=Path("data/survey.json"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--budget-gb", type=float, default=50.0)
    ap.add_argument("--exclude", default="", help="номера через запятую")
    args = ap.parse_args()

    skip = {s.strip() for s in args.exclude.split(",") if s.strip()}
    rows = json.loads(args.survey.read_text(encoding="utf-8"))
    if not any((r.get("src_objects") or r.get("objects") or 0) > 0 for r in rows):
        print(
            f"{args.survey}: ни у одного участка нет числа объектов.\n"
            "Отчёт снят без src_objects — перегенерируй:\n"
            "  uv run python scripts/survey_dataset.py --root <каталог> --out data/survey.json"
        )
        return 2

    have, cand = [], []
    for r in rows:
        num = r["name"][:3]
        obj = r.get("src_objects") or r.get("objects") or 0
        mb = float(r.get("mb") or 0)
        if num in skip or obj <= 0:
            continue
        if (args.raw / r["name"]).is_dir():
            have.append((num, r["name"], mb, obj))
        else:
            cand.append((num, r["name"], mb, obj))

    cand.sort(key=lambda t: t[3] / (t[2] / 1024) if t[2] else 0.0, reverse=True)

    budget_mb = args.budget_gb * 1024
    have_mb = sum(t[2] for t in have)
    used, picked = 0.0, []
    for row in cand:
        if used + row[2] > budget_mb:
            continue
        picked.append(row)
        used += row[2]

    print(f"{'участок':38} {'МБ':>7} {'объектов':>9} {'об/ГБ':>7}")
    print("--- уже на месте, бесплатно")
    for _num, name, mb, obj in sorted(have):
        d = obj / (mb / 1024) if mb else 0.0
        print(f"{name:38} {mb:7.0f} {obj:9d} {d:7.0f}")
    print("--- к копированию, по убыванию отдачи")
    for _num, name, mb, obj in picked:
        d = obj / (mb / 1024) if mb else 0.0
        print(f"{name:38} {mb:7.0f} {obj:9d} {d:7.0f}")

    total_mb = have_mb + used
    total_obj = sum(t[3] for t in have) + sum(t[3] for t in picked)
    print(
        f"\nна месте {len(have)} ({have_mb / 1024:.1f} ГБ) | "
        f"копировать {len(picked)} ({used / 1024:.1f} ГБ)"
    )
    print(f"итого {total_mb / 1024:.1f} ГБ, {total_obj} объектов")

    rest = [t for t in cand if t not in picked]
    print(
        f"не влезло {len(rest)} участков ({sum(t[2] for t in rest) / 1024:.1f} ГБ) — "
        f"обрабатывать прямо с /mnt/d"
    )
    print(f"\nbash scripts/fetch_sites.sh {_ranges([int(t[0]) for t in picked])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
