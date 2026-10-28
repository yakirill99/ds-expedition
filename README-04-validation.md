# Модуль 04-validation: внешняя валидация на AHN + регламент метрики

**Issue:** `docs/issues/04-validation.md`. **DoD:** отчёт F1 по классам на AHN; тест метрики
на примере из регламента. Вход: `configs/dataset_*.yaml`, `eval/metric.py`, `scripts/train.py`.

## Что уже есть в репо (проверено, код не менялся)

- **Метрика** (`src/expds/eval/metric.py`): IoU-полигоны (порог `eval.iou_thr`), точки —
  центроид pred в радиусе `eval.r_tol`; `r_tol_units: crs|meters` (для 3857 «meters»
  умножает допуск на `cosh(y/R)` — множитель Web Mercator); жадное сопоставление по
  убыванию score, один GT — один TP; по-классовый разбор (`evaluate`), отчёт по файлам
  (`evaluate_files`); 0/0 в precision/recall/F1 = 1.0.
- **Пространственный сплит** (`src/expds/train.py`): `train_sites` / `val_sites` — участки
  ЦЕЛИКОМ (не тайлы), пересечение множеств запрещено (`ValueError`). Требование issue
  выполнено скелетом — менять ничего не нужно.
- **Отчёт** (`scripts/evaluate.py`): таблица по классам + `--json`, `--validate`
  (валидатор формата платформы), `--errors` (FP/FN в GeoJSON для QGIS).
- **Конфиг AHN** (`configs/dataset_ahn.yaml`): полный pipeline-конфиг; `targets.classes:
  [mound, rampart, ditch]`; каналы rgb/mag у AHN отсутствуют — заполняются нулями
  (channel_mask); `train_sites`/`val_sites` заполняются после инвентаризации.

## Что добавлено в этом модуле

1. **`src/expds/data/build_site.py` — `build_site_from_dtm`**: растровый DTM-участок
   (лист/вырезка AHN, `dtm*.tif` в корне) → `layers/` + `samples/`. Сетка и CRS — из
   самого GeoTIFF (AHN: EPSG:28992, 0.5 м/px); фильтр земли/биннинг/облако не нужны
   (DTM уже земля, каналы облака дозаполнятся нулями); AppConfig роли 1 не требуется —
   дефолтные производные рельефа (`_DEFAULT_RELIEF`) дают тот же набор слоёв, что
   собирается из LAS на участке 013.
2. **`resolve_sites`**: растровые DTM-участки теперь собираются автоматически и в
   инференсе (`predict --data`), причём даже без конфига роли 1 (cfg=None). Растровые
   БЕЗ `dtm*.tif` (карты SpOr/Li) — по-прежнему ошибка/пропуск, как раньше.
3. **`scripts/build_layers.py` — pipeline-режим** (детект по YAML автоматически):
   `--config configs/dataset_ahn.yaml` обходит `data.sites_dir` и собирает каждый
   DTM-участок. Комментарий в шапке `dataset_ahn.yaml` теперь соответствует
   действительности.
4. **`scripts/fetch_ahn.py`**: скачивание вырезки AHN (DTM/DSM 0.5 м) официальным
   ImageServer Esri NL без авторизации → `data/ahn/raw/<site>/dtm.tif`.

## Полный цикл AHN (DoD)

```bash
# 1. Вырезки. EPSG:28992 (RD New, метры). Точки на местности — взять в QGIS/LoD AHN-viewer.
#    Пример (Vaals, юг Лимбурга, есть курганы):
uv run python scripts/fetch_ahn.py --site ahn_vaals_1 \
    --bbox 180000 318000 180512 318512
uv run python scripts/fetch_ahn.py --site ahn_vaals_2 \
    --center 180256 318256 --size-m 512 --dsm   # + dsm.tif вторым слоем
#    AHN5 покрывает пока юг страны (extent X 10k-225k, Y 306k-618k);
#    для севера: --ahn 4.

# 2. Слои + Sample для всех участков сразу (pipeline-режим):
uv run python scripts/build_layers.py --config configs/dataset_ahn.yaml

# 3. Разметка: data/ahn/raw/<site>/labels.geojson, EPSG:3857, свойства:
#    class = mound|rampart|ditch; геометрия Polygon (вал/насыпь) или Point.
#    Удобно оцифровывать в QGIS по hillshade_az_315.tif (курганы читаются лучше всего).

# 4. Инвентаризация и сплит (участки целиком!):
uv run python scripts/inventory.py --config configs/dataset_ahn.yaml
#    -> вписать train_sites: [ahn_vaals_1] / val_sites: [ahn_vaals_2] в configs/dataset_ahn.yaml

# 5. Обучение и инференс (CPU/GPU — без правок кода):
uv run python scripts/train.py --config configs/dataset_ahn.yaml --arch unet
uv run python scripts/predict.py --data data/ahn/raw/ahn_vaals_2 \
    --config configs/dataset_ahn.yaml --weights runs/<run>/best.pt \
    --out runs/pred_ahn.geojson

# 6. Отчёт F1 по классам (DoD):
uv run python scripts/evaluate.py --pred runs/pred_ahn.geojson \
    --gt data/ahn/raw/ahn_vaals_2/labels.geojson \
    --config configs/dataset_ahn.yaml --json runs/report_ahn.json
```

Профиль стадий сборки AHN-выреза 512x512 м (0.5 м/px): чтение растра 0.03 с,
производные рельефа 0.45 с, сохранение слоёв 0.37 с, Sample 1.69 с — итого ~2.5 с.

## Регламент: чек-лист сверки (БЛОК: документ ещё не получен)

Как только появится текст регламента — сверить и зафиксировать в
`configs/dataset_ahn.yaml` + дефолтах `EvalConfig` (`src/expds/eval/metric.py`):

| Пункт регламента | Ключ конфига | Сейчас |
|---|---|---|
| R_tolerance: значение и единицы | `eval.r_tol`, `eval.r_tol_units` | 15.0 crs (~9 м на 52° с.ш.) |
| IoU-порог для полигонов | `eval.iou_thr` | 0.5 |
| Список классов | `targets.classes` | [mound, rampart, ditch] |
| Один ли GeoJSON на всё / конвенция entrypoint | `predict.run`, submission | --out один файл |
| Пример-разбор из регламента | `tests/test_metric.py::test_regulation_*` | ждём документ |

Правило: если регламент задаёт допуск в метрах — переключить `r_tol_units: meters`
(код уже умеет), не трогая `r_tol`. Дефолты костяка меняются только по регламенту
(пункт «Заменяет в костяке: дефолты EvalConfig, targets.classes»).
