# Костяк КОЗ №3: пайплайн, модули, команды

Один код на синтетику и реальные данные. Участок = папка `layers/<канал>.tif` +
`samples/sample_meta.json` (+ `labels.geojson` в EPSG:3857 для обучения) — формат `build_layers` /
`make_synthetic`. Все параметры — в YAML датасета (`configs/dataset_*.yaml`), не в коде.

```
layers/*.tif ──load_site──> SiteData x[C,H,W], valid, (y_seg, y_hm)[K,H,W]
   └─KozDataset (тайлы tile_px/overlap_px, паддинг valid=False)
        └─model (BaseSegmenter): логиты [B, 2K, T, T] = [seg_0..K-1, hm_0..K-1]
             └─Stitcher (окно Ханна) -> probs [2K, H, W]
                  └─postprocess: seg -> контуры -> Дуглас–Пекер; hm -> пики -> круги; NMS; top-N
                       └─write_detections (internal -> EPSG:3857) -> pred.geojson
                            └─evaluate_files: IoU (полигоны), центроид в r_tol (точки) -> F1
```

## Модули

| Модуль | Главное | Владелец дальше |
|---|---|---|
| `features/*`, `data/*`, `io/geojson`, `targets/rasterize`, `tiles/*` | контракты роли 1 + 1.3/2.1 | Data/Geo |
| `models/factory`, `stub`, `koz_loss`, `checkpoint` | `build_model`, 2K выходов, `KozLoss`, чекпойнт со сверкой каналов/классов | Model |
| `fusion/stitch` | `Stitcher`, окно Ханна | Post/Vector |
| `post/vectorize`, `post/nms` | `postprocess`, `simplify_ring`, `nms_centroids` | Post/Vector |
| `eval/metric` | `match`, `evaluate`, `evaluate_files`, `polygon_iou` | Валидация |
| `infer`, `train`, `predict`, `submission` | `predict_probs`, `fit` (+oracle), `run`, `build_submission` | MLOps |
| `pipeline_config` | секции YAML -> dataclass, неизвестные ключи = ошибка | все |

Секции YAML: `data` (channels, normalize, labels_crs), `targets`, `tiles` — обязательны;
`model`, `loss`, `stitch`, `post`, `eval`, `train` — необязательны (дефолты dataclass).

## Команды

```bash
# синтетика
uv run python tools/make_synthetic.py --n-sites 3 --size 1024 --seed 0
# обучение (печатает oracle F1 и val F1 по эпохам) -> runs/<ts>_<arch>/
uv run python scripts/train.py --config configs/dataset_synthetic.yaml --arch unet --epochs 15
# инференс участка или папки участков
uv run python scripts/predict.py --data data/synthetic/site_3 \
    --config configs/dataset_synthetic.yaml --weights runs/<run>/best.pt --out pred.geojson
# архив сабмита + smoke из распакованного архива без сети
uv run python scripts/make_submission.py --config configs/dataset_synthetic.yaml \
    --weights runs/<run>/best.pt --out dist/submission.zip --smoke-data data/synthetic/site_3
```

Архив: `inference/{entrypoint.py, expds/, config.yaml, weights.pt, MANIFEST.json, requirements.txt}`.
Платформа: `python inference/entrypoint.py --data <участок|папка> --out pred.geojson`.

## Встроенные проверки

- **Oracle** в начале обучения: таргеты val через post + метрику. Должно быть 1.0; меньше —
  ошибка в цепочке post / метрика / разметка, а не в модели.
- **E2E** (`tests/test_end2end_synthetic.py`, CPU, ~20 с): TP/FP/FN entrypoint == val лучшей
  эпохи. Расхождение = разные пути train-val и инференса (нормализация, каналы, тайлинг, CRS).
- **Smoke сабмита**: entrypoint из распакованного архива, сеть заблокирована, `expds` из архива.
- Чекпойнт хранит `channel_names`, `classes`, `model_cfg`; инференс сверяет их с YAML.

## Результаты на синтетике (3 × 1024², train site_1-2, val site_3)

oracle F1 1.00; U-Net resnet34, 15 эпох × 512 тайлов (RTX 3050, ~7.5 с/эпоха): val F1 0.71
(p 0.62, r 0.84), в основном FP; инференс 1024² — 1.6 с. Stub (e2e, CPU): F1 0.14.

## Известные ограничения

- Дыры полигонов отбрасываются; все детекции — Polygon (точки — круги `point_radius_px`).
- `Stitcher` держит 2K×H×W float32 (10k² при K=3 ≈ 2.4 ГБ).
- Entrypoint принимает только готовые `layers/`; сырой LAS -> слои пока отдельно (`build_layers`).
- Регламент не получен: R_tolerance (значение и единицы: `eval.r_tol_units` crs|meters),
  IoU-порог, классы, конвенция вызова entrypoint на платформе.
