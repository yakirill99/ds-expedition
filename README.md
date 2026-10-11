================================================================================
DS-EXPEDITION — КОЗ №3 «КОМПЛЕКСИРОВАНИЕ»
================================================================================

Команда: 5 человек. Правила работы — в docs/REPO_SETUP.md.

Задача: автоматическое распознавание археологических объектов (курганы,
валы, рвы, городища, архитектуры) на данных дистанционного и глубинного
зондирования. Комплексирование нескольких источников (лидар, оптика,
геофизика).

Репозиторий: https://github.com/yakirill99/ds-expedition


================================================================================
БЫСТРЫЙ СТАРТ
================================================================================

git clone <url> && cd ds-expedition
make setup      # uv sync + pre-commit
make test


================================================================================
МОДУЛИ (src/expds/)
================================================================================

| Модуль      | Владелец    | Назначение                                     |
|-------------|-------------|------------------------------------------------|
| data        | Data/Geo    | чтение LAS/LAZ, GeoTIFF, кэш слоёв             |
| features    | Data/Geo    | контракты, фильтр земли, DTM, рельеф,          |
|             |             | признаки из облака, выравнивание               |
| models      | Model       | модели по источникам (U-Net, stub)             |
| fusion      | Fusion      | комплексирование (сшивка, фьюжн)               |
| evaluation  | Validation  | метрики, валидация, сабмит                     |
| utils       | Data/Geo    | профилировщик, seed                            |
| io          | Post/Vector | GeoJSON I/O                                    |
| targets     | Validation  | растеризация разметки                          |
| tiles       | Validation  | тайлер, датасет                                |
| post        | Post/Vector | векторизация, NMS                              |
| infer       | MLOps       | ядро инференса                                 |
| train       | MLOps       | обучение                                       |
| predict     | MLOps       | entrypoint предсказания                        |
| submission  | MLOps       | сборка архива сабмита                          |


================================================================================
СТРУКТУРА ПРОЕКТА
================================================================================

configs/              YAML-конфиги датасетов
data/
  raw/                исходные данные (в .gitignore)
  cache/              результат пайплайна: layers/*.tif, samples/*.npz
  models/             веса (в .gitignore)
  README.md           реестр данных
docs/
  CONTRACTS.md        контракты данных (Data/Geo)
  SKELETON.md         пайплайн, API модулей, команды
  REPO_SETUP.md       правила работы с репозиторием
  issues/             задачи по ролям
scripts/              CLI-скрипты
src/expds/            основной пакет
tests/                pytest (300+ тестов)
inference/            entrypoint для платформы


================================================================================
ПАЙПЛАЙН
================================================================================

Стек 1 (baseline): лидар → DTM → производные рельефа → модель → полигоны.
Стек 2 (модальные эксперты + фьюжн): добавляются оптика и геофизика.

Полный пайплайн (от сырья до предсказания):

  LAS/LAZ ──[io/las.py]──> точки (X, Y, Z, intensity, return_number,
                            classification) в internal_crs
       │
       ├──[features/ground.py]──> фильтр земли (preset: soft/strict)
       │
       ├──[features/dtm.py]──> DTM (mean/min/median + IDW)
       │
       ├──[features/relief.py]──> 15 производных: slope, aspect,
       │                          hillshade (4 азимута), SLRM (3 σ),
       │                          openness, sky-view, curvature,
       │                          TPI, TRI
       │
       ├──[features/cloud_features.py]──> 7 признаков из облака:
       │                                  point_density, mean_intensity,
       │                                  std_intensity, non_first_return_ratio,
       │                                  mean_return_number, mean_z, z_std
       │
       └──[scripts/build_layers.py]──> layers/*.tif + samples/sample.npz


================================================================================
РЕЗУЛЬТАТ ДЛЯ РОЛИ 2 (MODEL)
================================================================================

После прогона build_layers в data/cache/<site>/ появляются:

layers/*.tif
    26 отдельных GeoTIFF (для отладки, QGIS, визуального контроля).
    CRS = internal, nodata = -9999, float32.

samples/sample.npz + sample_meta.json
    Готовый тензор для модели. Один файл, один контракт.

    Порядок каналов (фиксирован):
      1. dtm — всегда канал 0.
      2. Рельеф по алфавиту: aspect, curvature_sigma_2.0,
         hillshade_az_135, hillshade_az_45, negative_openness,
         positive_openness, sky_view_factor, slope, slrm_sigma_8.0,
         tpi_radius_5, tri_radius_3, ... (зависит от конфига)
      3. Cloud features по алфавиту: mean_intensity,
         mean_return_number, mean_z, non_first_return_ratio,
         point_density, std_intensity, z_std.

    Точный список — в sample_meta.json, поле channel_names.

    Как читать:
        import json
        import numpy as np
        meta = json.load(open("data/cache/<site>/samples/sample_meta.json"))
        data = np.load("data/cache/<site>/samples/sample.npz")
        tensor = data["tensor"]              # [C, H, W] float32
        channel_mask = data["channel_mask"]  # [C] bool

    Соглашение о значениях:
        nodata заменён на 0 (через stack_layers).
        Валидные значения — как есть.
        channel_mask — все True в стеке 1.


================================================================================
ПАКЕТНАЯ ОБРАБОТКА УЧАСТКОВ
================================================================================

Все участки обрабатываются одной командой:

    uv run python scripts/process_sites.py --data-dir data/raw

Скрипт:
  - находит все папки с UTM.json (участки);
  - для каждого определяет CRS (LAS header или UTM.json);
  - генерирует конфиг в памяти (без файла на диске);
  - запускает полный пайплайн;
  - сохраняет layers/ + samples/ в data/cache/<name>/;
  - печатает отчёт: сколько успешно, сколько упало.

Опции:
  --only 006 017 035    обработать только эти участки
  --skip 048            пропустить эти участки
  --pixel-size 1.0      размер пикселя (по умолчанию 1.0 м)

Обработать один участок вручную:

    uv run python scripts/build_layers.py --config configs/dataset_013.yaml

Сгенерировать конфиг по папке участка:

    uv run python scripts/make_config_from_site.py \
        --site data/raw/013_Нора_Виннон/013_Нора_Виннон \
        --out configs/dataset_013.yaml

Для растровых участков (без лидара):

    uv run python scripts/prepare_raster_site.py \
        --site data/raw/048_Kalmykia \
        --out data/cache/048_Kalmykia \
        --pixel-size 1.0


================================================================================
ВИЗУАЛЬНЫЙ АУДИТ
================================================================================

Проверить препроцессинг глазами:

    uv run python scripts/visual_audit.py \
        --site 013_Nora_Vinnon \
        --labels "/path/to/разметка"

Строит PNG: слои из layers/ + контуры разметки. Отдельно — общие
картинки и зум на каждый объект разметки.

Результат в data/cache/<site>/audit/*.png.


================================================================================
ИНФЕРЕНС И САБМИТ
================================================================================

Обучение (пример на синтетике):

    uv run python scripts/train.py --config configs/dataset_synthetic.yaml \
        --arch unet --epochs 15

Инференс участка:

    uv run python scripts/predict.py --data data/synthetic/site_3 \
        --config configs/dataset_synthetic.yaml \
        --weights runs/<run>/best.pt \
        --out pred.geojson

Сборка архива сабмита:

    uv run python scripts/make_submission.py \
        --config configs/dataset_synthetic.yaml \
        --weights runs/<run>/best.pt \
        --out dist/submission.zip \
        --smoke-data data/synthetic/site_3

Архив: inference/{entrypoint.py, expds/, config.yaml, weights.pt,
MANIFEST.json, requirements.txt}.

Платформа: python inference/entrypoint.py --data <участок|папка>
--out pred.geojson.


================================================================================
ДАННЫЕ
================================================================================

Реестр данных — в data/README.md.

На текущий момент обработаны:

Лидарные участки (La):
  013_Нора_Виннон   2.8 точек/м²   EPSG:25833   37 курганов
  006_Осечки        5.1 точек/м²   EPSG:32637    3 кургана
  017_Юшково       47.9 точек/м²   EPSG:32636   85 курганов
  035_Шумгора       4.8 точек/м²   EPSG:32636   15 курганов

Растровые участки (SpOr):
  048_Калмыкия     3 канала       EPSG:32638   88 курганов

Все данные — локально (не в git). Участки в data/raw/<name>/.
Результаты в data/cache/<name>/{layers,samples}/.


================================================================================
ВЕТКИ
================================================================================

main      ← develop ← feature/<модуль>-<задача>
                       exp/<автор>-<идея>

Текущие:
  develop                          основная интеграция
  feature/data-config-generator    PR #19 (генератор конфигов, техдолг,
                                   инвентаризация LAS, фикс relief nodata,
                                   полигоны в аудите, пакетная обработка)
  feature/models-baseline          baseline-модель
  feature/skeleton                 костяк пайплайна (смержен)


================================================================================
ТЕСТЫ И ЛИНТЕР
================================================================================

make lint     # ruff check + ruff format --check
make test     # pytest (~300 тестов)

Правила стиля:
  - строка ≤ 100 символов;
  - type hints везде;
  - логирование через logging (не print);
  - константы модуля с префиксом _;
  - конфиг только через YAML (не хардкод).


================================================================================
ДОКУМЕНТАЦИЯ
================================================================================

docs/CONTRACTS.md    контракты данных (Grid, Layer, Sample, DataSpec)
docs/SKELETON.md     схема пайплайна, API модулей, встроенные проверки
docs/REPO_SETUP.md   правила работы с репозиторием
docs/issues/         задачи по ролям (Model, Post/Vector, Data/Geo,
                     Validation, MLOps)


================================================================================
КОНТАКТЫ
================================================================================

Капитан команды — MLOps. Вся коммуникация с оргкомитетом через капитана.
Репозиторий: https://github.com/yakirill99/ds-expedition
