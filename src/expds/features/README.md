================================================================================
МОДУЛЬ FEATURES
================================================================================

Владелец: Data/Geo

================================================================================
НАЗНАЧЕНИЕ
================================================================================

Превращает сырьё в чистые растровые слои в общей сетке: фильтр земли,
DTM, производные рельефа, признаки из облака, выравнивание.

================================================================================
ПУБЛИЧНЫЕ МОДУЛИ
================================================================================

grid.py — контракты.
  Grid(crs, pixel_size, x_min, y_min, width, height).
  Layer(name, data, grid, nodata, source, license, unit).
  Sample(tensor, channel_names, channel_mask, grid, target).
  DataSpec(grid, channel_names, nodata, metadata).
  stack_layers(layers, spec) -> Sample.
  grid_to_affine(grid) -> Affine.

config.py — загрузка YAML в типизированные dataclass'ы.
  load_config(path) -> AppConfig.
  setup_logging(config).

ground.py — фильтр земли.
  filter_ground(x, y, z, config, crs) -> GroundResult.
  Пресеты strict (убирает всё) и soft (сохраняет курганы).

dtm.py — построение DTM.
  build_dtm(x, y, z, grid, config, source) -> DtmResult.
  DtmResult(dtm, hole_mask, count).

relief.py — производные рельефа.
  build_relief_layers(dtm, pixel_size, ...) -> dict[str, np.ndarray].
  slope, aspect, hillshade, SLRM, openness, sky-view, curvature, TPI, TRI.

cloud_features.py — растровые признаки из облака.
  build_cloud_features(x, y, z, intensity, return_number, grid, enabled,
                       nodata) -> CloudFeaturesResult.
  point_density, mean_intensity, std_intensity, non_first_return_ratio,
  mean_return_number, mean_z, z_std.

align.py — выравнивание.
  resample_layer(layer, target_grid, method) -> Layer.
  estimate_shift(layer_a, layer_b, max_shift) -> ShiftEstimate (NCC).
  apply_shift(layer, shift) -> Layer.
  check_alignment(layers, reference_name, ...) -> AlignmentReport.

================================================================================
ВХОД
================================================================================

Точки земли (x, y, z) в метрах, CRS = internal.
Grid — целевая сетка.
Конфиг (AppConfig).

================================================================================
ВЫХОД
================================================================================

Layer — растровые слои в Grid.
Sample — тензор [C, H, W] + channel_names, channel_mask.
GroundResult, DtmResult, CloudFeaturesResult, ShiftEstimate.

================================================================================
СОГЛАШЕНИЯ
================================================================================

Все производные — float32.
Nodata -9999.0.
Порядок каналов в Sample: dtm, потом рельеф, потом cloud.
align.estimate_shift использует NCC, порог 0.3.

================================================================================
ПРИМЕР
================================================================================

from expds.features.config import load_config
from expds.features.ground import filter_ground
from expds.features.dtm import build_dtm
from expds.features.relief import build_relief_layers

config = load_config("configs/dataset_013.yaml")
ground = filter_ground(x, y, z, config.ground, config.crs.internal)
dtm = build_dtm(
    x[ground.ground_mask], y[ground.ground_mask], z[ground.ground_mask],
    grid, config.dtm,
)
relief = build_relief_layers(dtm.dtm.data, grid.pixel_size, ...)

================================================================================
КАК ЗАПУСТИТЬ
================================================================================

uv run python scripts/build_layers.py --config configs/dataset_013.yaml
