================================================================================
МОДУЛЬ DATA
================================================================================

Владелец: Data/Geo

================================================================================
НАЗНАЧЕНИЕ
================================================================================

Чтение сырья (LAS/LAZ, GeoTIFF), кэш слоёв.

================================================================================
ПУБЛИЧНЫЕ МОДУЛИ
================================================================================

las.py — чтение LAS/LAZ.
  read_header(path) -> LasHeaderInfo.
  read_points(path, chunk_size, source_crs, target_crs, source_z_unit,
              max_points) -> Iterator[np.ndarray] — чанки (N, 5):
              x, y, z, intensity, return_number.
  summarize(...) -> LasSummary — сводка: bounds, диапазон Z, единицы.

raster.py — чтение/запись GeoTIFF.
  read_raster(path, grid, nodata, resampling) -> Layer.
  write_raster(layer, path, compress, dtype).
  read_raster_meta(path) -> RasterMeta.
  ВАЖНО: не перепроецирует между CRS. Это задача features/align.py.

cache.py — кэш слоёв по хэшу.
  LayerCache(root).get_or_compute(name, params, grid, compute_fn,
                                  use_cache).
  compute_cache_key(name, params, grid) -> str.
  Ключ зависит от _CACHE_VERSION, _get_code_version(), name, params, grid.

================================================================================
ВХОД
================================================================================

LAS/LAZ — лидар. X, Y, Z + intensity, return_number, classification.
GeoTIFF — растры. CRS, nodata, dtype.

================================================================================
ВЫХОД
================================================================================

Layer — растровый слой в общей сетке Grid.
GeoTIFF — тот же Layer на диске.

================================================================================
СОГЛАШЕНИЯ
================================================================================

row 0 = юг (конвенция Grid). На границе с GeoTIFF делается flipud.
Nodata по умолчанию -9999.0.
Dtype float32.
CRS перепроецируется при чтении LAS (в internal_crs).

================================================================================
ПРИМЕР
================================================================================

from pathlib import Path
from expds.data.las import read_points

for chunk in read_points(
    path=Path("data/raw/site.las"),
    chunk_size=1_000_000,
    source_crs="EPSG:25833",
    target_crs="EPSG:25833",
    source_z_unit="meters",
):
    x, y, z, intensity, return_number = chunk.T
    # ...

================================================================================
КАК ЗАПУСТИТЬ
================================================================================

uv run python scripts/build_layers.py --config configs/dataset_013.yaml
