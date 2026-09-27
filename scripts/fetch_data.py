"""Скачивает файлы из data/README.md по публичным ссылкам Яндекс Диска.

Использование:
    python scripts/fetch_data.py            # всё
    python scripts/fetch_data.py --only id1 id2
Уже скачанные файлы пропускаются.
"""

import argparse
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "README.md"
API = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="


def parse_registry() -> dict[str, tuple[Path, str]]:
    items = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 6 and cells[5].startswith("http"):
            items[cells[0]] = (ROOT / cells[4], cells[5])
    return items


def download(url: str, dst: Path) -> None:
    import json

    href = json.load(urllib.request.urlopen(API + urllib.parse.quote(url)))["href"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(href, tmp)
    tmp.rename(dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="id из реестра")
    args = p.parse_args()
    items = parse_registry()
    ids = args.only or list(items)
    for i in ids:
        if i not in items:
            print(f"[skip] {i}: нет в реестре")
            continue
        dst, url = items[i]
        if dst.exists():
            print(f"[ok]   {i}: уже есть")
            continue
        print(f"[get]  {i} -> {dst.relative_to(ROOT)}")
        download(url, dst)


if __name__ == "__main__":
    main()
