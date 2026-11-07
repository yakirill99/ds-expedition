#!/usr/bin/env bash
# Очередь без тяжёлых участков: до ~30 млн точек на участок.
# Тяжёлые ломают WSL по памяти (vstack держит две копии массива точек).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

HEAVY="190_Муравлево 028_Вышегжа_2 186_Семибратнее_1 007_Армиево 004_Боровичи_1 037_Никола_Ленивец 024_Стрекаловка 093_Старый_Барах 002_Тешилово 021_Новенькое 106_Верхмарево 017_Юшково 019_Селяне"

is_heavy() {
    local n=$1 h
    for h in $HEAVY; do [ "$n" = "$h" ] && return 0; done
    return 1
}

ok=0; fail=0; skip=0
for s in $(uv run python scripts/sites.py list todo_lidar --paths 2>/dev/null); do
    n=$(basename "$s")
    if is_heavy "$n"; then
        echo "SKIP $n (тяжёлый, отдельно)"
        skip=$((skip + 1)); continue
    fi
    log="logs/build_${n}.log"; t0=$SECONDS
    if bash scripts/prepare_sites.sh "$s" >"$log" 2>&1; then
        echo "OK   $n  $((SECONDS - t0))s"; ok=$((ok + 1))
    else
        echo "FAIL $n  $((SECONDS - t0))s -> $log"; fail=$((fail + 1))
    fi
done
echo; echo "ok $ok | fail $fail | отложено $skip"
