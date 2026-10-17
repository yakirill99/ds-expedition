# Модуль viz — визуализация карт

Модуль отвечает за отображение геопространственных данных: локаций и слоёв карт.

## Хранилище тайлов

Тайлы хранятся по схеме:

```
<MAP_STORAGE_ROOT>/<location_slug>/<layer_slug>/<z>/<x>/<y>.png
```

### Настройки (`web/config/settings.py` и `.env`)

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `MAP_STORAGE_BACKEND` | Бэкенд хранилища: `filesystem` или `s3` | `filesystem` |
| `MAP_STORAGE_ROOT` | Корневая папка с тайлами (для файлового бэкенда) | `web/tiles` |
| `MAP_TILE_URL_PREFIX` | URL-префикс для тайлов | `/tiles/` |
| `MAP_S3_BUCKET` | S3-бакет (будущее) | — |
| `MAP_S3_ENDPOINT_URL` | S3 endpoint (будущее) | — |

### Формат данных локации

Рекомендуемая структура описана в `docs/LOCATION_FORMAT.md`. Кратко:

```text
data/<location>/
  manifest.json          # slug, name, center, zoom
  tiles/
    satellite/{z}/{x}/{y}.png
    lidar/{z}/{x}/{y}.png
```

### Подготовка тайлов из исходных данных

```bash
uv run python scripts/prepare_tiles.py \\
    --location data/001_Гнёздово_Нивленское \\
    --output web/tiles \\
    --zooms 13 14 15 16 \\
    --crs 32636
```

Скрипт поддерживает:
- GeoTIFF (`*.tif`) → нарезка в тайлы;
- LAS-облака точек (`*.las`) → растрирование по высоте `z` → тайлы;
- GeoJSON-разметку → копируется рядом с тайлами и отображается поверх карты.

GeoJSON автоматически перепроецируется из своей CRS (например, `EPSG:3857`) в `WGS84` для Leaflet.

### Импорт локации

```bash
# С manifest.json (центр и название подтянутся автоматически)
uv run python web/manage.py import_location \\
    --tiles-dir web/tiles \\
    --slug 001-gnezdovo-nivlenskoe

# Без manifest.json — параметры передаются явно
uv run python web/manage.py import_location \\
    --name "Участок 1" --slug uchastok-1 \\
    --tiles-dir web/tiles --lat 55.75 --lon 37.61 --zoom 14
```

После импорта тайлы доступны по URL:

```
/viz/tiles/<location_slug>/<layer_slug>/<z>/<x>/<y>.png
```

### API

- `GET /viz/api/locations/` — список локаций.
- `GET /viz/api/locations/<slug>/layers/` — слои локации с шаблонами тайловых URL.

### Переход на S3

Для перехода достаточно:
1. Установить `MAP_STORAGE_BACKEND=s3`.
2. Заполнить `MAP_S3_BUCKET` и `MAP_S3_ENDPOINT_URL`.
3. Реализовать методы в `web/apps/viz/storage/s3.py`.

Бизнес-логика (`views.py`, `models.py`) останется без изменений.
