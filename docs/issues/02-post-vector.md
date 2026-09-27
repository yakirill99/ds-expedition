# [Post/Vector] Пороги по val, дыры, свой marching squares, валидатор формата

**Вход:** `probs[2K,H,W]` (`expds.infer.predict_probs`), `PostConfig`, `evaluate`.
**Выход:** `post/*` с теми же сигнатурами `postprocess(...) -> list[Detection]`.

**Задачи**
- Скрипт подбора `seg_thr`, `hm_thr`, `min_area_px`, `nms_radius_px` по val (кэш probs на диск,
  перебор без повторного инференса). Сейчас ошибки U-Net — в основном FP.
- Дыры полигонов (сейчас отбрасываются; `Detection` хранит только внешнее кольцо — согласовать с Data/Geo).
- Свой marching squares вместо `skimage.find_contours` (или обосновать оставить: BSD).
- Валидатор формата платформы (типы геометрий, CRS, свойства) + `eval/errors.py`: разбор FP/FN,
  экспорт в GeoJSON для QGIS.

**DoD:** рост val F1 от подбора порогов на синтетике зафиксирован в PR; oracle 1.0; тесты `test_post.py`.
**Заменяет в костяке:** дефолты `PostConfig`, `post/vectorize.py`.
