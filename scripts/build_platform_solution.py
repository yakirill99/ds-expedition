#!/usr/bin/env python3
"""Собирает архив для платформы: solution.py + metadata.json.

Платформа (Приложение 2 Технического регламента КОЗ №3) ждёт zip из двух
файлов В КОРНЕ:

    solution.py      точка входа: python solution.py <input_dir> <output.geojson>
    metadata.json    {"image": "<образ>", "entry_point": "python solution.py"}

Код (expds), config.yaml и weights.pt лежат в Docker-образе под /app.
Поэтому архив весит килобайты, а не 87 МБ — не путать с `make_submission.py`,
который собирает self-contained архив для локального прогона.

Зачем пост-патч
---------------
В разметке шесть классов из семи полигональные. `postprocess` гоняет для
каждого класса обе ветки — полигональную и точечную. Точечное предсказание
по полигональному классу не проходит IoU >= 0.5 с полигоном GT, то есть
даёт только FP. Замер на oracle (идеальные таргеты, 9 участков, 437 GT):

    обе ветки   tp 232  fp  52
    полигоны    tp 226  fp   4      <- heatmap = 92% FP при 2.6% TP

Образ собран до этой правки, поэтому config.yaml из архива не помог бы:
`PostConfig` в образе не знает ключа `branches` и падает на unknown keys.
Поэтому патчим не конфиг, а имя, которое `postprocess` берёт из модуля в
момент вызова. Патч обёрнут в try/except: если в образе другая структура,
он тихо пропустится и пайплайн отработает как раньше — хуже не станет.

Использование
-------------
    uv run python scripts/build_platform_solution.py
    uv run python scripts/build_platform_solution.py --out dist/sub.zip
    uv run python scripts/build_platform_solution.py --no-post-patch
    uv run python scripts/build_platform_solution.py --point-classes finds_points,object_poly
    uv run python scripts/build_platform_solution.py --check dist/solution_platform.zip

Скрипт идемпотентен: повторный запуск перезаписывает блок патча, а не
дублирует его. Зависимостей нет — только стандартная библиотека.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BEGIN = "# === PLATFORM POST-PATCH: сгенерировано build_platform_solution.py ==="
END = "# === END PLATFORM POST-PATCH ==="

PATCH_TEMPLATE = '''{begin}
# Блок перезаписывается scripts/build_platform_solution.py. Руками не править.
_PATCH_VERSION = "1"
_POINT_CLASSES = {point_classes!r}


def _apply_platform_post_patch() -> None:
    """Гасит heatmap-ветку для полигональных классов.

    Точечные предсказания по классам, размеченным полигонами, не проходят
    IoU >= 0.5 и дают только FP. На oracle ветка отдаёт 92% FP при 2.6% TP.
    Классы из _POINT_CLASSES (по умолчанию finds_points) сохраняют прежнее
    поведение: у них разметка точечная, и heatmap для них — единственная ветка.

    Патч не роняет пайплайн: любая ошибка печатается и проглатывается.
    """
    try:
        _ensure_expds()
    except Exception as exc:  # noqa: BLE001
        print(f"post-patch: SKIPPED, expds не найден ({{exc}})", flush=True)
        return
    try:
        import expds.post.vectorize as _V
    except Exception as exc:  # noqa: BLE001
        print(f"post-patch: SKIPPED, нет expds.post.vectorize ({{exc}})", flush=True)
        return

    original = getattr(_V, "heatmap_to_points", None)
    if original is None:
        print("post-patch: SKIPPED, heatmap_to_points отсутствует", flush=True)
        return
    if getattr(original, "_koz_patched", False):
        print("post-patch: уже применён", flush=True)
        return

    def heatmap_to_points(prob, grid, cls, cfg, *args, **kwargs):
        if cls in _POINT_CLASSES:
            return original(prob, grid, cls, cfg, *args, **kwargs)
        return []

    heatmap_to_points._koz_patched = True  # type: ignore[attr-defined]
    _V.heatmap_to_points = heatmap_to_points
    print(
        f"post-patch v{{_PATCH_VERSION}}: heatmap выключен, "
        f"точечные классы {{_POINT_CLASSES}}",
        flush=True,
    )


_apply_platform_post_patch()
{end}
'''


def _strip_patch(src: str) -> str:
    """Убирает ранее вставленный блок патча (идемпотентность)."""
    while True:
        a = src.find(BEGIN)
        if a < 0:
            return src
        b = src.find(END, a)
        if b < 0:
            return src[:a]
        b = src.find("\n", b) + 1
        # съедаем пустые строки вокруг блока
        while a > 0 and src[a - 1] == "\n" and src[:a].endswith("\n\n"):
            a -= 1
        src = src[:a] + src[b:]


def _normalize_eol(src: str) -> str:
    """Приводит переводы строк к LF.

    Архив распаковывается на Linux, но solution.py мог быть сохранён в
    CRLF. Без нормализации `rstrip("\\n")` оставляет висящий `\\r`, и
    повторная сборка даёт другой файл — идемпотентность ломается.
    """
    return src.replace("\r\n", "\n").replace("\r", "\n")


def _inject(src: str, point_classes: tuple[str, ...]) -> str:
    """Вставляет блок патча перед `if __name__`, иначе в конец файла."""
    body = _strip_patch(_normalize_eol(src)).rstrip("\n") + "\n"
    block = PATCH_TEMPLATE.format(
        begin=BEGIN,
        end=END,
        point_classes=point_classes,
    )
    anchor = 'if __name__ == "__main__":'
    idx = body.find(anchor)
    if idx < 0:
        return body + "\n\n" + block
    return body[:idx] + block + "\n\n" + body[idx:]


def build(
    solution: Path,
    out: Path,
    image: str,
    entry_point: str,
    point_classes: tuple[str, ...],
    post_patch: bool,
) -> int:
    if not solution.is_file():
        print(f"нет файла: {solution}", file=sys.stderr)
        return 1

    src = solution.read_text(encoding="utf-8")
    if post_patch:
        new_src = _inject(src, point_classes)
    else:
        new_src = _strip_patch(_normalize_eol(src)).rstrip("\n") + "\n"

    try:
        compile(new_src, str(solution), "exec")
    except SyntaxError as exc:
        print(f"патченный solution.py не компилируется: {exc}", file=sys.stderr)
        return 1

    metadata = {"image": image, "entry_point": entry_point}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("solution.py", new_src)
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))

    print(f"готово: {out} ({out.stat().st_size} байт)")
    print(f"  image        {image}")
    print(f"  entry_point  {entry_point}")
    print(f"  post-patch   {'включён' if post_patch else 'ВЫКЛЮЧЕН'}")
    if post_patch:
        print(f"  точечные     {', '.join(point_classes) if point_classes else '(нет)'}")
    return 0


def check(path: Path) -> int:
    """Проверяет готовый архив: состав, metadata, компилируемость, патч."""
    if not path.is_file():
        print(f"нет файла: {path}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        ok = True
        extra = sorted(set(names) - {"solution.py", "metadata.json"})
        if extra:
            print(f"FAIL: лишние файлы в корне архива: {extra}")
            ok = False
        if "solution.py" not in names or "metadata.json" not in names:
            print(f"FAIL: нет обязательных файлов, есть только {names}")
            return 1
        src = zf.read("solution.py").decode("utf-8")
        meta = json.loads(zf.read("metadata.json"))
        try:
            compile(src, "solution.py", "exec")
            print("OK: solution.py компилируется")
        except SyntaxError as exc:
            print(f"FAIL: не компилируется: {exc}")
            ok = False
    print(f"OK: файлы {names}")
    print(f"OK: metadata {meta}")
    patched = BEGIN in src
    print(f"{'OK' if patched else 'WARN'}: пост-патч {'на месте' if patched else 'отсутствует'}")
    if patched:
        print(f"     вызов: {src.count('_apply_platform_post_patch()')} (ожидается 2)")
    print(f"OK: размер {path.stat().st_size} байт")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Архив для платформы: solution.py + metadata.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--solution", type=Path, default=REPO / "inference" / "solution.py")
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "solution_platform.zip")
    ap.add_argument(
        "--image",
        default="yakirill99/ds-expedition:latest",
        help="образ с кодом, config.yaml и weights.pt",
    )
    ap.add_argument("--entry-point", default="python solution.py")
    ap.add_argument(
        "--point-classes",
        default="finds_points",
        help="классы, для которых heatmap остаётся включённой (через запятую)",
    )
    ap.add_argument(
        "--no-post-patch",
        action="store_true",
        help="не вставлять рантайм-патч (архив как есть)",
    )
    ap.add_argument("--check", type=Path, help="только проверить готовый архив")
    args = ap.parse_args(argv)

    if args.check:
        return check(args.check)

    classes = tuple(c.strip() for c in args.point_classes.split(",") if c.strip())
    return build(
        solution=args.solution,
        out=args.out,
        image=args.image,
        entry_point=args.entry_point,
        point_classes=classes,
        post_patch=not args.no_post_patch,
    )


if __name__ == "__main__":
    sys.exit(main())
