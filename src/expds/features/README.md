================================================================================
МОДУЛЬ FEATURES
================================================================================

Владелец: Data/Geo

================================================================================
НАЗНАЧЕНИЕ
================================================================================

Превращает сырьё в чистые растровые слои в общей сетке: фильтр земли,
DTM, производные рельефа, признаки из облака, спектральные индексы,
выравнивание, универсальный адаптер build_site.

================================================================================
ПУБЛИЧНЫЕ МОДУЛИ
================================================================================

grid.py — контракты.
  Grid(crs, pixel_size, x_min, y_min, width, height).
  Layer(name, data, grid, nodata, source, license, unit).
  Sample(tensor, channel_names, channel_mask, grid, target, valid_mask).
  DataSpec(grid, channel_names, nodata, metadata).
  stack_layers(layers, spec, compute_valid_mask=True) -> Sample.
  grid_to_affine(grid) -> Affine.

config.py — загрузка YAML в типизированные dataclass'ы.
  load_config(path) -> AppConfig.
  setup_logging(config).
  Секции spectral, modalities, output_crs опциональны (дефолты).

ground.py — фильтр земли.
  filter_ground(x, y, z, config, crs) -> GroundResult.
  Пресеты strict (убирает всё) и soft (сохраняет курганы).

dtm.py — построение DTM.
  build_dtm(x, y, z, grid, config, source) -> DtmResult.
  DtmResult(dtm, hole_mask, count).

relief.py — производные рельефа.
  build_relief_layers(dtm, pixel_size, ...) -> dict[str, np.ndarray].
  slope, aspect, hillshade, SLRM, openness, sky-view, curvature, TPI, TRI.
  Все compute_* принимают nodata, маскируют nodata в результате.

cloud_features.py — растровые признаки из облака.
  build_cloud_features(x, y, z, intensity, return_number, grid, enabled,
                       nodata) -> CloudFeaturesResult.
  point_density, mean_intensity, std_intensity, non_first_return_ratio,
  mean_return_number, mean_z, z_std.

spectral.py — спектральные индексы из оптики.
  build_spectral_indices(channels, valid_mask, grid, config, source)
      -> SpectralResult(layers, skipped).
  compute_ndvi, compute_sr, compute_ari, compute_gndvi, compute_ndwi.
  Индексы каналов 0-based. Если NIR/Red нет — слой не строится
  (graceful skip), channel_mask=False.

align.py — выравнивание и перепроецирование.
  resample_layer(layer, target_grid, method) -> Layer.
  estimate_shift(layer_a, layer_b, max_shift) -> ShiftEstimate (NCC).
  apply_shift(layer, shift) -> Layer.
  check_alignment(layers, reference_name, ...) -> AlignmentReport.
  reproject_layer_to_crs(layer, target_crs, ...) -> Layer.
  reproject_layers_to_crs(layers, target_crs, ...) -> dict[str, Layer].

build_site.py — универсальный адаптер участка.
  inventory_site(site_dir, config) -> SiteInventory.
  build_site(raw_dir, out_dir, config, max_points, output_crs) -> SiteResult.
  Лидар + растры (or, sp_or, ae, maggr, etgr, terfl) -> layers/ +
  samples/sample.npz (с valid_mask).

================================================================================
ВХОД
================================================================================

Точки земли (x, y, z) в метрах, CRS = internal.
GeoTIFF-модальности (оптика, геофизика) в своих CRS.
Grid — целевая сетка.
Конфиг (AppConfig) с секциями spectral, modalities, output_crs.

================================================================================
ВЫХОД
================================================================================

Layer — растровые слои в Grid.
Sample — тензор [C, H, W] + channel_names, channel_mask, valid_mask.
GroundResult, DtmResult, CloudFeaturesResult, SpectralResult,
ShiftEstimate, SiteResult.

================================================================================
СОГЛАШЕНИЯ
================================================================================

Все производные — float32.
Nodata -9999.0.
Порядок каналов в Sample: dtm → рельеф → cloud → остальное (по алфавиту).
hole_mask и count — в «остальное», не выделяются (контракт с ролью 5).
align.estimate_shift использует NCC, порог 0.3.
Спектральные индексы — graceful skip: нет каналов → нет слоя, mask False.

================================================================================
ПРИМЕР
================================================================================

Лидар:

    from expds.features.config import load_config
    from expds.features.build_site import build_site

    config = load_config("configs/dataset_013.yaml")
    result = build_site(
        raw_dir=config.dataset.raw_dir,
        out_dir=config.dataset.cache_dir,
        config=config,
    )
    # result.layers, result.sample, result.grid, result.inventory

Только индексы:

    from expds.features.spectral import build_spectral_indices

    result = build_spectral_indices(
        channels={0: red, 1: green, 3: nir},
        valid_mask=mask,
        grid=grid,
        config=config.spectral,
    )
    # result.layers: {"ndvi": Layer, "sr": Layer, ...}

Перепроецирование в EPSG:3857:

    from expds.features.align import reproject_layers_to_crs

    layers_3857 = reproject_layers_to_crs(
        layers=layers,
        target_crs="EPSG:3857",
    )

================================================================================
КАК ЗАПУСТИТЬ
================================================================================

uv run python scripts/build_layers.py --config configs/dataset_013.yaml

Или пакетно:

uv run python scripts/process_sites.py --data-dir data/raw
