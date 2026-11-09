"""Архив для платформы: состав, metadata, идемпотентность пост-патча."""

import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "build_platform_solution.py"
SOLUTION = REPO / "inference" / "solution.py"

pytestmark = pytest.mark.skipif(
    not (SCRIPT.is_file() and SOLUTION.is_file()), reason="нет скрипта или solution.py"
)


def _load_script():
    """Импортирует scripts/build_platform_solution.py по пути."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_platform_solution", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build(tmp_path: Path, *args: str) -> Path:
    mod = _load_script()
    out = tmp_path / "sol.zip"
    rc = mod.main(["--out", str(out), *args])
    assert rc == 0
    return out


def test_archive_has_only_platform_files(tmp_path):
    """Платформа ждёт ровно два файла в корне — никаких подпапок."""
    out = _build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
    assert names == ["metadata.json", "solution.py"]


def test_metadata_contract(tmp_path):
    out = _build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        meta = json.loads(zf.read("metadata.json"))
    assert meta["entry_point"] == "python solution.py"
    assert meta["image"]
    assert set(meta) == {"image", "entry_point"}


def test_solution_compiles(tmp_path):
    out = _build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        src = zf.read("solution.py").decode("utf-8")
    compile(src, "solution.py", "exec")


def test_patch_is_idempotent(tmp_path):
    """Три прогона подряд дают побайтово одинаковый архив."""
    a = _build(tmp_path / "a")
    b = _build(tmp_path / "b")
    md5 = lambda p: __import__("hashlib").md5(p.read_bytes()).hexdigest()  # noqa: E731
    assert md5(a) == md5(b)


def test_patch_block_appears_once(tmp_path):
    out = _build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        src = zf.read("solution.py").decode("utf-8")
    mod = _load_script()
    assert src.count(mod.BEGIN) == 1
    assert src.count("_apply_platform_post_patch()") == 2  # определение + вызов


def test_no_post_patch_removes_block(tmp_path):
    """--no-post-patch выдаёт solution.py без патча."""
    out = _build(tmp_path, "--no-post-patch")
    with zipfile.ZipFile(out) as zf:
        src = zf.read("solution.py").decode("utf-8")
    mod = _load_script()
    assert mod.BEGIN not in src
    assert "def _apply_platform_post_patch" not in src


def test_point_classes_are_embedded(tmp_path):
    out = _build(tmp_path, "--point-classes", "finds_points,object_poly")
    with zipfile.ZipFile(out) as zf:
        src = zf.read("solution.py").decode("utf-8")
    assert "_POINT_CLASSES = ('finds_points', 'object_poly')" in src
    compile(src, "solution.py", "exec")


def test_check_reports_ok(tmp_path):
    out = _build(tmp_path)
    assert _load_script().main(["--check", str(out)]) == 0


def test_check_fails_on_foreign_zip(tmp_path):
    """Архив с лишними файлами не проходит --check."""
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("solution.py", "print(1)\n")
        zf.writestr("metadata.json", json.dumps({"image": "x", "entry_point": "y"}))
        zf.writestr("extra.txt", "лишний")
    assert _load_script().main(["--check", str(bad)]) == 1
