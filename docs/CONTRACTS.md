# Контракты данных (роль 1, develop) — справка для костяка

Источник: `src/expds/features/grid.py`, `config.py`, `src/expds/data/raster.py`,
`cache.py`, `scripts/build_layers.py`, `configs/dataset_raleigh.yaml`.
Правило: контракты не переписываем, только расширяем.

## 1. Grid — сетка и координаты

- `Grid(crs, pixel_size, x_min, y_min, width, height)`, frozen, валидация в `__post_init__`.
- `shape = (height, width)`; `bounds = (x_min, y_min, x_max, y_max)`.
- Центр пикселя: `x = x_min + (col + 0.5) * ps`, `y = y_min + (row + 0.5) * ps`.
- `transform_to_pixel(x, y) -> (col, row)` (float), `transform_to_world(col, row) -> (x, y)`.
- `Grid.from_bounds(crs, ps, bounds)`: width/height через ceil, origin = (x_min, y_min).
- По docstring Grid: **row 0 — снизу (y_min)**.

### GeoTIFF ↔ Grid (raster.py)

- `grid_to_affine(grid)` = `Affine(ps, 0, x_min, 0, -ps, y_max)` — north-up, row 0 файла = **верх** (y_max).
- `_build_grid_from_raster(transform)`: `y_min = y_max - height * ps`.
- `read_raster`/`write_raster` делают `flipud` на границе: в памяти всегда конвенция Grid (row 0 = юг), на диске — north-up. South-up tif → ValueError.
- `read_raster(path, grid)` — чтение с ресэмплингом в grid при несовпадении.
- `write_raster(...)` — пишет `layer.data` с transform из `grid_to_affine`.

**⚠ Баг ориентации Y исправлен** (PR `fix/raster-y-orientation`):
`flipud` на границе в `write_raster`/`read_raster` и вокруг `reproject` в `align.py`;
`relief.py`: `arctan2(-gx, -gy)`. Кэш до фикса пересобрать.

## 2. Layer, DataSpec, Sample, stack_layers

- `Layer(name, data[H,W] float32, grid, nodata=-9999, source, license, unit)`;
  `valid_mask = data != nodata`. dtype строго float32, shape строго = grid.shape.
- `DataSpec(grid, channel_names, nodata, metadata)`: имена уникальны,
  `channel_index(name)`. Порядок каналов = контракт модели.
- `Sample(tensor[C,H,W], channel_names, channel_mask[C], grid, target[H,W] | None,
  valid_mask[H,W] | None)`:
  - `valid_mask` — попиксельная валидность (True там, где хотя бы один
    присутствующий канал валиден). `None` — маска не запрошена
    (обратная совместимость).
- `stack_layers(layers, spec, compute_valid_mask=True) -> Sample`:
  - nodata → 0 в `tensor`;
  - `channel_mask` — наличие канала целиком, не попиксельно;
  - `valid_mask` — попиксельно: `True`, если хоть один канал в пикселе валиден;
  - отсутствующий канал → нули + mask False; лишние слои → warning, игнор;
  - слой с другим Grid → ValueError (сначала `align`).

### Решения костяка поверх контракта

- `Sample.valid_mask` — новое поле, критично для U-Net/сегментации:
  модель не должна учиться на выдуманных нулях в дырах DTM.
- `Sample.target` (`[H,W]`) костяк не использует: таргеты `(K,H,W)` —
  `y_seg` и `y_hm` — идут отдельно в `KozDataset`. `target=None`.

## 3. Слои build_layers и порядок каналов

Порядок (`_order_channel_names`): `dtm` → рельеф → cloud → остальное.
Группы по префиксам, **внутри группы строковая сортировка** (не порядок префиксов).

Шаблоны имён (relief.py, cloud_features.py, dtm.py):

| Группа | Имена |
|---|---|
| dtm | `dtm` |
| рельеф | `slope`, `aspect`, `hillshade_az_{int(az)}`, `slrm_sigma_{sigma}`, `positive_openness`, `negative_openness`, `sky_view_factor`, `curvature_sigma_{sigma}`, `tpi_radius_{r}`, `tri_radius_{r}` |
| cloud | `point_density`, `mean_intensity`, `std_intensity`, `non_first_return_ratio`, `mean_return_number`, `mean_z`, `z_std`, `height_range`, `skewness_z`, `kurtosis_z` |
| остальное | `hole_mask`, `count` (служебные), `rgb_r`, `rgb_g`, `rgb_b`, `mag` (синтетика), `or_*`, `sp_or_*`, `ae_*`, `maggr_*`, `etgr_*`, `terfl_*` (растры), спектральные индексы `*_ndvi`, `*_sr`, `*_ari`, `*_gndvi`, `*_ndwi`, `*_savi` |

`hole_mask` и `count` **не выделяются** — идут в «остальное» по алфавиту.
Это сохраняет контракт с ролью 5 (порядок каналов в синтетике не меняется).

Следствия:
- набор каналов зависит от конфига (азимуты, σ, радиусы, модальности) →
  `channel_names` сохраняются в чекпойнт, на инференсе сверяются с DataSpec,
  при расхождении — ошибка;
- каналы костяка для других модальностей: `rgb_r`, `rgb_g`, `rgb_b`, `mag`
  (Layer строго 2D) — попадают в группу «остальное», в конец по алфавиту.

## 4. Кэш и sample

- `LayerCache(root)`: `<root>/<key>.tif` + `<root>/<key>.json`,
  `key = compute_cache_key(name, ...)` — хэш параметров, **не имя слоя**.
  Имена файлов кэша для костяка не контракт; читать слои через `LayerCache` или
  `sample.npz`, не по glob `*.tif`.
- `data/cache/samples/sample.npz`: ключи `tensor` (`[C,H,W]` float32),
  `channel_mask` (`[C]` bool), `valid_mask` (`[H,W]` bool — новое, если
  строился через `_save_sample`/`_save_site`). Сжатый (`savez_compressed`).
  Старый код, читающий только `tensor`/`channel_mask`, продолжает работать.
- `data/cache/samples/sample_meta.json`: `channel_names`, `channel_mask`,
  `grid` {crs, pixel_size, x_min, y_min, width, height}, `nodata`,
  `n_channels`, `n_available_channels`, `has_valid_mask`.
- Grid восстанавливается как `Grid(**meta["grid"])`.

## 5. CRS

- `crs.source` — CRS исходника (CRS из LAS приоритетнее конфига).
- `crs.internal` — метрическая CRS: Grid, все слои, все расчёты (raleigh: EPSG:32617).
- `crs.output` — EPSG:3857: вход разметки и выход GeoJSON, перепроекция только
  на границе (`align.reproject_layer_to_crs` / `reproject_layers_to_crs`,
  флаг `--output-crs` в скриптах). По умолчанию слои остаются в internal
  (модель работает в метрах).
- Метрика считается в EPSG:3857. `r_tol` задаётся в единицах 3857
  (≠ метры: 1 м на местности ≈ 1/cos φ единиц 3857).
- TODO из регламента: значение R_tolerance и его единицы, критерий матчинга
  полигонов (IoU-порог), список классов.

## 6. Конфиг

- `load_config(path) -> AppConfig`, обязательные секции:
  dataset, lidar, crs, grid, ground, dtm, relief, cloud_features, align, logging.
- Опциональные секции (обратная совместимость): `spectral`, `modalities`,
  `output_crs`. Если их нет — берутся дефолты в `config.py`.
- `dataset.lidar_file` обязателен → в конфигах без LAS (синтетика) — заглушка.
- Лишние секции игнорируются → секции костяка (`targets`, `tiles`, `post`, `eval`)
  читает отдельный загрузчик костяка; `config.py` не трогаем.
- `setup_logging(cfg.logging)` — консоль + файл.

### Секции для мультимодальности и спектральных индексов

- `modalities.enabled` — какие модальности обрабатывать (li, or, sp_or, ae,
  maggr, etgr, terfl).
- `modalities.priority` — порядок для выбора целевой сетки, если лидара нет.
- `spectral.enabled` — какие индексы строить (ndvi, sr, ari, gndvi, ndwi, savi).
- `spectral.red_channel` / `nir_channel` / `green_channel` / `blue_channel` —
  0-based индексы каналов в снимке. `None` → индекс не строится (graceful skip).
- `spectral.channel_std_min` — минимальное std канала (по валидным пикселям),
  чтобы признать его информативным. Константные каналы (alpha-маска=255,
  nodata-заливка) исключаются из индексов и из `layers/`: NDVI/ARI на них
  дают мусор. Значение по умолчанию 0.01.
- `output_crs.enabled` / `output_crs.epsg` — экспорт в EPSG:3857 (по умолчанию
  выкл; включается флагом `--output-crs`).

### Соглашения spectral.py

- `SR` клиппится в `[0, _SR_MAX=20]`: при `Red→0` отношение взрывается,
  для модели это выброс.
- `ARI` клиппится в `[-_ARI_ABS_MAX, +_ARI_ABS_MAX]` (`_ARI_ABS_MAX=10`):
  `1/green - 1/red` при яркостях, близких к 0, даёт выбросы.
- `SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)`, `L=0.5` (Huete, 1988):
  устойчив к почвенному фону — важно для распаханных курганов. Клиппинг
  в `[-1.5, 1.5]`.
- `NDVI`, `GNDVI`, `NDWI` — нормированные разности, клиппинг в `[-1, 1]`.
- Канал считается информативным, если по валидным пикселям
  `std >= channel_std_min`. Иначе индекс не строится (`skipped`),
  `channel_mask=False`.
- `blue_channel` в текущих индексах не используется: NDBI требует SWIR,
  которого в наших данных нет. Поле зарезервировано на будущее.

### Соглашения align.reproject_layers_to_crs

- Сетка строится по bounds ПЕРВОГО слоя. Если слой выходит за bounds
  целевой сетки — внешняя часть обрезается, в лог пишется `WARNING`.
  Первым передавай слой с наибольшим покрытием (обычно лидар, если
  он шире, или общий bbox, построенный заранее).

## 7. Проверки, закрывающие открытые вопросы

- Ориентация Y: закрыто, регрессионные тесты `tests/test_*_orientation.py`.
- `valid_mask` в Sample: закрыто, `tests/test_spectral.py`, `test_grid.py`.
- Спектральные индексы: закрыто, `tests/test_spectral.py` (20+ тестов).
- Универсальный адаптер `build_site`: закрыто, `src/expds/features/build_site.py`,
  `scripts/build_layers.py` и `scripts/process_sites.py` на нём.
- Точный список каналов: `sample_meta.json` после прогона `build_layers.py`
  или `process_sites.py`.
- Регламент: R_tolerance, IoU-порог, классы.

## Костяк

Пайплайн, API модулей, команды: [SKELETON.md](SKELETON.md).
