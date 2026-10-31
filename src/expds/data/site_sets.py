"""Наборы участков из одного манифеста (configs/sites.yaml).

Участки лежат в data/raw один раз. train/val/test/smoke — это фильтры над
каталогом, а не копии и не симлинки.

Сплит по группам и по хешу: группа ландшафта (Гнёздово, Осечки, …) целиком
попадает в одну часть, а какая это часть — решает sha1(seed:группа).
Поэтому докопирование новых участков не перетасовывает уже распределённые,
в отличие от жадной балансировки, которая пересчитывает всё заново.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MANIFEST = Path("configs/sites.yaml")
SPLITS = ("train", "val", "test")
KINDS = ("lidar", "optic", "geo", "none")
FLAGS = ("labels", "layers", "rgb", "built")
_SET_KEYS = {"description", "split", "kind", "require", "missing", "only", "exclude"}
_NUM_RE = re.compile(r"^(\d{3})_")
_ID_RE = re.compile(r"^\d{1,3}(-\d{1,3})?$")


@dataclass(frozen=True)
class SiteInfo:
    """Участок в каталоге и то, что о нём знает манифест."""

    name: str
    path: Path
    number: int
    kind: str
    group: str
    split: str
    has_labels: bool
    has_layers: bool
    has_rgb: bool
    n_objects: int

    def has(self, flag: str) -> bool:
        """Флаг для require/missing."""
        return {
            "labels": self.has_labels and self.n_objects > 0,
            "layers": self.has_layers,
            "rgb": self.has_rgb,
            "built": self.has_layers or self.has_rgb,
        }[flag]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def parse_refs(items: Any) -> tuple[set[int], set[str]]:
    """Ссылки на участки -> (номера, имена).

    Строка "014-024,030" или список строк. Элемент — номер, диапазон
    номеров или полное имя папки. Числа без кавычек запрещены:
    YAML 1.1 читает 014 как восьмеричное 12.
    """
    if items is None:
        return set(), set()
    if isinstance(items, str):
        items = items.split(",")
    ids: set[int] = set()
    names: set[str] = set()
    for raw in items:
        if isinstance(raw, int | float):
            raise ValueError(
                f'номер {raw!r} без кавычек: YAML читает 014 как восьмеричное — пишите "{raw}"'
            )
        s = str(raw).strip()
        if not s:
            continue
        if _ID_RE.match(s):
            lo, _, hi = s.partition("-")
            a, b = int(lo), int(hi or lo)
            ids.update(range(min(a, b), max(a, b) + 1))
        else:
            names.add(s)
    return ids, names


def _kind(path: Path) -> str:
    codes = {p.name[:2] for p in path.iterdir() if p.is_dir()}
    if (path / "layers" / "dtm.tif").exists() or codes & {"01", "02"}:
        return "lidar"
    if codes & {"03", "04", "05"}:
        return "optic"
    if codes & {"07", "08", "09"}:
        return "geo"
    return "none"


def _group(name: str, prefixes: list[str]) -> str:
    rest = _NUM_RE.sub("", name)
    for prefix in prefixes:
        if rest.startswith(prefix):
            return prefix
    return name


def _bucket(group: str, seed: int, fractions: dict[str, float]) -> str:
    digest = hashlib.sha1(f"{seed}:{group}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for split in ("test", "val"):
        edge += float(fractions.get(split, 0.0))
        if u < edge:
            return split
    return "train"


def _n_objects(labels: Path) -> int:
    if not labels.is_file():
        return 0
    try:
        return len(json.loads(labels.read_text(encoding="utf-8")).get("features") or [])
    except (OSError, ValueError):
        return 0


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Читает и проверяет манифест."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fractions = (data.get("split") or {}).get("fractions") or {}
    if fractions:
        total = sum(float(fractions.get(s, 0.0)) for s in SPLITS)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split.fractions в сумме {total}, нужно 1.0")
    for name, spec in (data.get("sets") or {}).items():
        spec = spec or {}
        unknown = set(spec) - _SET_KEYS
        if unknown:
            raise ValueError(f"набор {name}: неизвестные ключи {sorted(unknown)}")
        for key in ("require", "missing"):
            bad = set(_as_list(spec.get(key))) - set(FLAGS)
            if bad:
                raise ValueError(f"набор {name}.{key}: {sorted(bad)}, допустимо {FLAGS}")
        bad = set(_as_list(spec.get("kind"))) - set(KINDS)
        if bad:
            raise ValueError(f"набор {name}.kind: {sorted(bad)}, допустимо {KINDS}")
        parse_refs(spec.get("only"))
        parse_refs(spec.get("exclude"))
    return data


def scan(root: Path, manifest: dict[str, Any]) -> list[SiteInfo]:
    """Все участки каталога с группой и сплитом; глобальный exclude применён."""
    split_cfg = manifest.get("split") or {}
    seed = int(split_cfg.get("seed", 0))
    fractions = split_cfg.get("fractions") or {"train": 0.8, "val": 0.1, "test": 0.1}
    pins = split_cfg.get("pin") or {}
    prefixes = [str(p) for p in manifest.get("groups") or []]
    ex_ids, ex_names = parse_refs(manifest.get("exclude"))

    out: list[SiteInfo] = []
    for path in sorted(Path(root).iterdir()):
        m = _NUM_RE.match(path.name)
        if not path.is_dir() or not m:
            continue
        number = int(m.group(1))
        if number in ex_ids or path.name in ex_names:
            continue
        group = _group(path.name, prefixes)
        split = pins.get(group) or _bucket(group, seed, fractions)
        if split not in SPLITS:
            raise ValueError(f"pin {group}: {split!r}, допустимо {SPLITS}")
        labels = path / "labels.geojson"
        out.append(
            SiteInfo(
                name=path.name,
                path=path,
                number=number,
                kind=_kind(path),
                group=group,
                split=split,
                has_labels=labels.is_file(),
                has_layers=(path / "layers" / "dtm.tif").exists(),
                has_rgb=(path / "layers" / "rgb_r.tif").exists(),
                n_objects=_n_objects(labels),
            )
        )
    return out


def select(sites: list[SiteInfo], spec: dict[str, Any] | None) -> list[SiteInfo]:
    """Фильтр набора над результатом scan."""
    spec = spec or {}
    splits = set(_as_list(spec.get("split")))
    kinds = set(_as_list(spec.get("kind")))
    require = _as_list(spec.get("require"))
    missing = _as_list(spec.get("missing"))
    only_ids, only_names = parse_refs(spec.get("only"))
    ex_ids, ex_names = parse_refs(spec.get("exclude"))
    has_only = bool(only_ids or only_names)

    out = []
    for s in sites:
        if splits and s.split not in splits:
            continue
        if kinds and s.kind not in kinds:
            continue
        if has_only and s.number not in only_ids and s.name not in only_names:
            continue
        if s.number in ex_ids or s.name in ex_names:
            continue
        if not all(s.has(f) for f in require):
            continue
        if any(s.has(f) for f in missing):
            continue
        out.append(s)
    return out


def resolve(
    set_name: str, manifest_path: Path = DEFAULT_MANIFEST, root: Path | None = None
) -> list[SiteInfo]:
    """Участки набора set_name — то, что зовёт train.py."""
    manifest = load_manifest(manifest_path)
    sets = manifest.get("sets") or {}
    if set_name not in sets:
        raise KeyError(f"нет набора {set_name!r}; есть: {sorted(sets)}")
    root = Path(root or manifest.get("root", "data/raw"))
    return select(scan(root, manifest), sets[set_name])
