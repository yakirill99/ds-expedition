# [Валидация] AHN без правок кода, пространственный сплит, регламент метрики

**Вход:** `configs/dataset_*.yaml`, `eval/metric.py`, `scripts/train.py`.
**Выход:** `configs/dataset_ahn.yaml`, отчёт метрик, уточнённые параметры `eval`.

**Задачи**
- `configs/dataset_ahn.yaml` и прогон train/predict/eval на AHN **без правок кода** (только YAML).
- Сплит по территориям (участки целиком, не тайлы) — уже так в `train_sites`/`val_sites`.
- По регламенту: R_tolerance и единицы (`eval.r_tol_units`: crs|meters для 3857), IoU-порог,
  список классов, один ли GeoJSON на всё. Тест метрики на примере из регламента, если он есть.

**DoD:** отчёт F1 по классам на AHN; `test_metric.py` с кейсом из регламента.
**Заменяет в костяке:** дефолты `EvalConfig`, `targets.classes`.
