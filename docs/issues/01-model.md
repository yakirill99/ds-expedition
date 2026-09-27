# [Model] U-Net под реальные данные: аугментации, кропы, warmup

**Вход:** `KozDataset` (тайлы `x[C,T,T]`, `channel_mask`, `valid`, `y_seg`, `y_hm`), `ModelConfig`,
`KozLoss`, `scripts/train.py`.
**Выход:** обученный `best.pt` (через `save_checkpoint`), конфиг `model`/`loss`/`train` в YAML.

**Задачи**
- Аугментации: flip/rot90 синхронно для `x`, `y_seg`, `y_hm`, `valid` (row 0 = юг — для flip неважно,
  но проверить тестом), channel dropout с учётом `channel_mask`.
- Случайные кропы вместо фиксированных окон `TileIndex` для train.
- Warmup lr (в 1-й эпохе выброс hm-лосса), подбор `hm_min_norm`, весов seg/hm.
- Обучение под 12 ГБ (5070): batch, AMP, `tiles_per_epoch`.
- Свой U-Net (без smp) до точки ветвления — отдельной веткой.

**DoD:** val F1 на синтетике ≥ 0.71 (текущий baseline), oracle 1.0, e2e зелёный,
`tests/test_model_factory.py` расширен.
**Заменяет в костяке:** `train.fit` (dataloader/аугментации), дефолты `ModelConfig`/`LossConfig`.
