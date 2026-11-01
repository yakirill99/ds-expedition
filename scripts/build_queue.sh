#!/usr/bin/env bash
# Сборка слоёв по очереди манифеста: мелкие участки первыми.
#
#   bash scripts/build_queue.sh optic    # только rgb_* (быстро)
#   bash scripts/build_queue.sh lidar    # только prepare_sites.sh (долго)
#   bash scripts/build_queue.sh both
#
# Параллелить нельзя: prepare_sites.sh пишет в общий data/cache/raw.
# Логи по участкам: logs/build_<имя>.log.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

mode="${1:-both}"
ok=0; fail=0

queue() {
    uv run python scripts/sites.py list "$1" --paths 2>/dev/null \
        | while read -r s; do printf '%s %s\n' "$(du -sm "$s" 2>/dev/null | cut -f1)" "$s"; done \
        | sort -n | cut -d' ' -f2-
}

run_one() {
    local s=$1 kind=$2 n log t0 rc took
    n=$(basename "$s"); log="logs/build_${n}.log"; t0=$SECONDS
    if [ "$kind" = lidar ]; then
        bash scripts/prepare_sites.sh "$s" >"$log" 2>&1
    else
        uv run python scripts/build_layers_raster.py "$s" >>"$log" 2>&1
    fi
    rc=$?; took=$((SECONDS - t0))
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] OK   $n  ${took}s"; ok=$((ok + 1))
    else
        echo "[$(date +%H:%M:%S)] FAIL $n  ${took}s -> $log"; fail=$((fail + 1))
    fi
}

if [ "$mode" = optic ] || [ "$mode" = both ]; then
    echo "=== rgb_* (оптика)"
    while read -r s; do [ -n "$s" ] && run_one "$s" optic; done < <(queue todo_optic)
fi

if [ "$mode" = lidar ] || [ "$mode" = both ]; then
    echo "=== слои (лидар)"
    while read -r s; do [ -n "$s" ] && run_one "$s" lidar; done < <(queue todo_lidar)
fi

echo; echo "ok $ok | fail $fail"
