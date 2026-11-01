#!/usr/bin/env bash
# Копирование участков с Яндекс.Диска в data/raw по номерам.
#
#   DRY=1 bash scripts/fetch_sites.sh 014-024      # только план и объём
#   bash scripts/fetch_sites.sh 014-024
#   bash scripts/fetch_sites.sh 014-024 030 200-212
#   bash scripts/fetch_sites.sh all
#
# Повторный запуск безопасен: rsync докачивает и пропускает готовое.
# Участок, который Яндекс ещё пишет, пропускается — повтори ту же команду позже.
# fuser -m здесь бесполезен: процесс Яндекса живёт в Windows и из WSL не виден,
# поэтому занятость определяется по временным файлам и меняющемуся размеру.

set -uo pipefail

SRC="${KOZ_SRC:-/mnt/d/Yandex_Disk/Yandex.Disk-yakirill1999/Учеба/Хакатоны/2026-09-ds-expedition/04-data/train-koz-3}"
DST="${KOZ_DST:-$HOME/ds-expedition/data/raw}"
STABLE_SEC="${STABLE_SEC:-2}"
EXCL=(--exclude='*.~*' --exclude='.~*' --exclude='*.tmp' --exclude='*.part'
      --exclude='*.sync' --exclude='*.crdownload')

gb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1024 / 1024 / 1024 }'; }

if [ $# -eq 0 ]; then sed -n '2,12p' "$0"; exit 2; fi
for r in "$@"; do
    [[ $r == all || $r =~ ^[0-9]{1,3}(-[0-9]{1,3})?$ ]] \
        || { echo "не понял диапазон: '$r' (нужно 014-024, 030 или all)"; exit 2; }
done
[ -d "$SRC" ] || { echo "нет источника: $SRC"; exit 2; }
mkdir -p "$DST"

in_ranges() {
    local n=$((10#$1)); shift
    local r a b t
    for r in "$@"; do
        [ "$r" = all ] && return 0
        if [[ $r == *-* ]]; then
            a=$((10#${r%-*})); b=$((10#${r#*-}))
            (( a > b )) && { t=$a; a=$b; b=$t; }
            (( n >= a && n <= b )) && return 0
        else
            (( n == 10#$r )) && return 0
        fi
    done
    return 1
}

sites=()
for d in "$SRC"/[0-9][0-9][0-9]_*/; do
    name=$(basename "$d")
    in_ranges "${name:0:3}" "$@" && sites+=("$name")
done
[ ${#sites[@]} -gt 0 ] || { echo "под диапазон ничего не попало"; exit 1; }

echo "участков: ${#sites[@]}  ($SRC -> $DST)"
echo "считаю объём..."
need=0
for s in "${sites[@]}"; do
    a=$(du -sb "$SRC/$s" 2>/dev/null | cut -f1); a=${a:-0}
    b=$(du -sb "$DST/$s" 2>/dev/null | cut -f1); b=${b:-0}
    (( a > b )) && need=$(( need + a - b ))
done
avail=$(df --output=avail -B1 "$DST" | tail -1)
cavail=$(df --output=avail -B1 /mnt/c 2>/dev/null | tail -1); cavail=${cavail:-$avail}
lim=$(( avail < cavail ? avail : cavail ))
echo "докачать ~$(gb "$need") ГБ   свободно ~$(gb "$lim") ГБ (минимум из ~ и C:)"
if (( need > lim * 9 / 10 )) && [ -z "${FORCE:-}" ]; then
    echo "!! места не хватит с запасом 10%. Сузь диапазон или FORCE=1"; exit 1
fi
if [ -n "${DRY:-}" ]; then printf '  %s\n' "${sites[@]}"; exit 0; fi

ok=(); skip=(); fail=(); t0=$SECONDS; i=0
for s in "${sites[@]}"; do
    i=$((i + 1))
    echo; echo "[$i/${#sites[@]}] $s   ($(( (SECONDS - t0) / 60 )) мин)"

    tmp=$(find "$SRC/$s" \( -name '*.~*' -o -name '.~*' -o -name '*.tmp' -o -name '*.part' \
          -o -name '*.sync' -o -name '*.crdownload' \) -print -quit 2>/dev/null)
    if [ -n "$tmp" ]; then
        echo "  качается: $(basename "$tmp") — пропуск"; skip+=("$s:temp"); continue
    fi
    s1=$(du -sb "$SRC/$s" | cut -f1); sleep "$STABLE_SEC"; s2=$(du -sb "$SRC/$s" | cut -f1)
    if [ "$s1" != "$s2" ]; then
        echo "  размер меняется ($s1 -> $s2) — Яндекс пишет, пропуск"; skip+=("$s:growing"); continue
    fi

    mkdir -p "$DST/$s"
    if ! nice -n 10 rsync -a --partial --append-verify --info=progress2 \
            "${EXCL[@]}" "$SRC/$s/" "$DST/$s/"; then
        fail+=("$s:rsync"); continue
    fi

    missing=$(comm -23 \
        <(cd "$SRC/$s" && find . -type f ! -name '*.~*' ! -name '.~*' ! -name '*.tmp' \
              ! -name '*.part' ! -name '*.sync' -printf '%s %p\n' | LC_ALL=C sort) \
        <(cd "$DST/$s" && find . -type f -printf '%s %p\n' | LC_ALL=C sort))
    if [ -n "$missing" ]; then
        echo "  !! не сошлось:"; echo "$missing" | head -5 | sed 's/^/     /'
        fail+=("$s:verify"); continue
    fi
    ok+=("$s")
done

echo
echo "за $(( (SECONDS - t0) / 60 )) мин: ok ${#ok[@]} | пропущено ${#skip[@]} | ошибок ${#fail[@]}"
for x in "${skip[@]}"; do echo "  SKIP $x"; done
for x in "${fail[@]}"; do echo "  FAIL $x"; done
[ ${#skip[@]} -gt 0 ] && echo "пропущенные: запусти ту же команду позже"
exit $(( ${#fail[@]} > 0 ))
