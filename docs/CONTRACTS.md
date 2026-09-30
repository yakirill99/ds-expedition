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

- `_build_transform(grid)` = `Affine(ps, 0, x_min, 0, -ps, y_max)` — north-up, row 0 файла = **верх** (y_max).
- `_build_grid_from_raster(transform)`: `y_min = y_max - height * ps`.
- Флипа массива (`flipud`, `[::-1]`) в raster.py нет.
- `read_raster(path, grid)` — чтение с ресэмплингом в grid при несовпадении.
- `write_raster(...)` — пишет `layer.data` с transform из `_build_transform`.

**⚠ Баг ориентации Y подтверждён и исправлен** (PR `fix/raster-y-orientation`):
`flipud` на границе в `write_raster`/`read_raster` и вокруг `reproject` в `align.py`;
`relief.py`: `arctan2(-gx, -gy)`. В памяти всегда конвенция Grid (row 0 = юг),
на диске — north-up. South-up tif → ValueError. Кэш до фикса пересобрать.

## 2. Layer, DataSpec, Sample, stack_layers

- `Layer(name, data[H,W] float32, grid, nodata=-9999, source, license, unit)`;
  `valid_mask = data != nodata`. dtype строго float32, shape строго = grid.shape.
- `DataSpec(grid, channel_names, nodata, metadata)`: имена уникальны,
  `channel_index(name)`. Порядок каналов = контракт модели.
- `Sample(tensor[C,H,W], channel_names, channel_mask[C], grid, target[H,W] | None)`.
- `stack_layers(layers, spec) -> Sample`:
  - nodata → 0;
  - `channel_mask` — наличие канала целиком, не попиксельно;
  - отсутствующий канал → нули + mask False; лишние слои → warning, игнор;
  - слой с другим Grid → ValueError (сначала `align`).

### Решения костяка поверх контракта

- Попиксельная валидность теряется в `stack_layers` → valid-маску берём из
  `Layer("dtm").valid_mask` до стака и храним отдельно.
- `Sample.target` (`[H,W]`) костяк не использует: таргеты `(K,H,W)` —
  `y_seg` и `y_hm` — идут отдельно в `KozDataset`. `target=None`.

## 3. Слои build_layers и порядок каналов

Порядок (`_order_channel_names`): `dtm` → рельеф → cloud → остальное.
Группы по префиксам, **внутри группы строковая сортировка** (не порядок префиксов).

Шаблоны имён (relief.py, cloud_features.py, dtm.py):

| Группа | Имена |
|---|---|
| dtm | `dtm` (+ `hole_mask`, `count` из dtm.py — входят ли в 19, проверить) |
| рельеф | `slope`, `aspect`, `hillshade_az_{int(az)}`, `slrm_sigma_{sigma}`, `positive_openness`, `negative_openness`, `sky_view_factor`, `curvature_sigma_{sigma}`, `tpi_radius_{r}`, `tri_radius_{r}` |
| cloud | `point_density`, `mean_intensity`, `std_intensity`, `non_first_return_ratio`, `mean_return_number`, `mean_z`, `z_std` |

Ожидаемый порядок для raleigh (σ — float → `8.0`, `2.0`; строковая сортировка → `135` < `45`):

```
dtm,
aspect, curvature_sigma_2.0, hillshade_az_135, hillshade_az_45,
negative_openness, positive_openness, sky_view_factor, slope,
slrm_sigma_8.0, tpi_radius_5, tri_radius_3,
mean_intensity, mean_return_number, mean_z, non_first_return_ratio,
point_density, std_intensity, z_std
```

Итого 19 при условии, что openness/SVF есть, а `hole_mask`/`count` в sample не входят.
Подтвердить по `sample_meta.json` после первого реального прогона.

Следствия:
- набор каналов зависит от конфига (азимуты, σ, радиусы) → `channel_names`
  сохраняются в чекпойнт, на инференсе сверяются с DataSpec, при расхождении — ошибка;
- каналы костяка для других модальностей: `rgb_r`, `rgb_g`, `rgb_b`, `mag`
  (Layer строго 2D) — попадают в группу «остальное», в конец по алфавиту.

## 4. Кэш и sample

- `LayerCache(root)`: `<root>/<key>.tif` + `<root>/<key>.json`,
  `key = compute_cache_key(name, ...)` — хэш параметров, **не имя слоя**.
  Имена файлов кэша для костяка не контракт; читать слои через `LayerCache` или
  `sample.npz`, не по glob `*.tif`.
- `data/cache/samples/sample.npz`: ключи `tensor` (`[C,H,W]` float32),
  `channel_mask` (`[C]` bool). Сжатый (`savez_compressed`).
- `data/cache/samples/sample_meta.json`: `channel_names`, `channel_mask`,
  `grid` {crs, pixel_size, x_min, y_min, width, height}, `nodata`,
  `n_channels`, `n_available_channels`.
- Grid восстанавливается как `Grid(**meta["grid"])`.

## 5. CRS

- `crs.source` — CRS исходника (CRS из LAS приоритетнее конфига).
- `crs.internal` — метрическая CRS: Grid, все слои, все расчёты (raleigh: EPSG:32617).
- `crs.output` — EPSG:3857: вход разметки и выход GeoJSON, перепроекция только на границе.
- Метрика считается в EPSG:3857. `r_tol` задаётся в единицах 3857
  (≠ метры: 1 м на местности ≈ 1/cos φ единиц 3857).
- TODO из регламента: значение R_tolerance и его единицы, критерий матчинга
  полигонов (IoU-порог), список классов.

## 6. Конфиг

- `load_config(path) -> AppConfig`, все секции обязательны:
  dataset, lidar, crs, grid, ground, dtm, relief, cloud_features, align, logging.
- `dataset.lidar_file` обязателен → в конфигах без LAS (синтетика) — заглушка.
- Лишние секции игнорируются → секции костяка (`targets`, `tiles`, `post`, `eval`)
  читает отдельный загрузчик костяка; `config.py` не трогаем.
- `setup_logging(cfg.logging)` — консоль + файл.

## 7. Проверки, закрывающие открытые вопросы

- Ориентация Y: закрыто, регрессионные тесты `tests/test_*_orientation.py`.
  строки 0 с `Grid.transform_to_world(0, 0)`.
- Точный список 19 каналов: `sample_meta.json` после прогона `build_layers.py`.
- Регламент: R_tolerance, IoU-порог, классы.

## Костяк

Пайплайн, API модулей, команды: [SKELETON.md](SKELETON.md).
