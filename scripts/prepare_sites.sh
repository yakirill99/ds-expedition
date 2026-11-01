#!/usr/bin/env bash
# Подготовка реальных участков: CRS -> слои -> разметка -> сводка.
#
#   bash scripts/prepare_sites.sh data/raw/00[6-9]_* data/raw/01[0-3]_*
#
# Падение одного участка не останавливает остальные.

set -uo pipefail

CONFIG_DIR="configs/sites"
CACHE_DIR="data/cache/raw"
mkdir -p "$CONFIG_DIR"

ok=(); fail=(); skip=()

for site in "$@"; do
    name=$(basename "$site")
    echo
    echo "=============================================================="
    echo "  $name"
    echo "=============================================================="

    if [ ! -d "$site" ]; then echo "  НЕТ КАТАЛОГА"; fail+=("$name:no-dir"); continue; fi
    if [ ! -f "$site/UTM.json" ]; then
        echo "  НЕТ UTM.json"; fail+=("$name:no-utm"); continue
    fi
    echo "--- UTM.json ---"; cat "$site/UTM.json"; echo

    epsg=$(grep -o 'EPSG::[0-9]*' "$site/UTM.json" | head -1 | cut -d: -f3)
    case "$epsg" in
        4326|4258|"")
            echo "  CRS $epsg не метрическая — нужна перепроекция"; skip+=("$name:geographic"); continue ;;
    esac
    echo "  EPSG:$epsg"

    # Лидар обязателен для diag/build; участки на аэрофото — отдельная задача
    if ! compgen -G "$site/01_*_La_точки/*.las" >/dev/null 2>&1 \
       && ! compgen -G "$site/01_*_La_точки/*.laz" >/dev/null 2>&1; then
        echo "  ЛИДАРА НЕТ (только растровые модальности) — пропускаю, нужен raster-путь"
        skip+=("$name:no-lidar"); continue
    fi

    echo "--- diag_crs ---"
    if ! uv run python scripts/diag_crs.py "$site" 2>&1 | tail -12; then
        echo "  diag_crs упал"; fail+=("$name:diag"); continue
    fi

    cfg="$CONFIG_DIR/${name}.yaml"
    echo "--- конфиг -> $cfg ---"
    uv run python scripts/make_config_from_site.py --site "$site" --out "$cfg" \
        || { echo "  конфиг не создан"; fail+=("$name:config"); continue; }

    echo "--- build_layers ---"
    rm -rf "$CACHE_DIR"
    uv run python scripts/build_layers.py --config "$cfg" --force \
        || { echo "  слои не собрались"; fail+=("$name:layers"); continue; }

    # build_layers кладёт результат в общий кэш — переносим к участку
    if [ -d "$CACHE_DIR/layers" ]; then
        mkdir -p "$site/layers" "$site/samples"
        mv "$CACHE_DIR"/layers/*.tif "$site/layers/" 2>/dev/null
        if [ -f "$CACHE_DIR/samples/sample.npz" ]; then
            cp -f "$CACHE_DIR/samples/sample.npz" "$site/samples/sample.npz"
        else
            echo "  внимание: sample.npz не создан (для load_site не нужен)"
        fi
        rm -rf "$CACHE_DIR"
    fi
    n_layers=$(ls "$site/layers"/*.tif 2>/dev/null | wc -l)
    echo "  слоёв у участка: $n_layers"
    if [ "$n_layers" -eq 0 ]; then
        echo "  слои не легли к участку"; fail+=("$name:layers-empty"); continue
    fi

    echo "--- merge labels ---"
    if ls "$site"/0*_*разметка* >/dev/null 2>&1; then
        uv run python scripts/merge_labels.py --site "$site" --out-name labels.geojson \
            || { echo "  разметка не слилась"; fail+=("$name:merge"); continue; }
    else
        echo "  папки разметки нет — участок без GT"
    fi

    ok+=("$name")
done

echo
echo "готово: ${#ok[@]}   пропущено: ${#skip[@]}   проблемы: ${#fail[@]}"
for s in "${ok[@]}";   do echo "  OK    $s"; done
for s in "${skip[@]}"; do echo "  SKIP  $s"; done
for s in "${fail[@]}"; do echo "  FAIL  $s"; done
