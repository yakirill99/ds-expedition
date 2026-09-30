# [Data/Geo] Сырой вход платформы -> layers/, реальный датасет

**Вход:** сырые данные участка (LAS/LAZ, hillshade, снимки, магнитка, георадар), `build_layers`.
**Выход:** папка участка в формате костяка (`layers/*.tif`, `samples/sample_meta.json`).

**Задачи**
- Функция `build_site(raw_dir, out_dir, cfg)` (без YAML-путей внутри) и вызов из `expds.predict.run`,
  если у участка нет `layers/` (сейчас — FileNotFoundError с подсказкой).
- Каналы `point_density`, `mean_intensity` (сейчас в синтетике отсутствуют -> нули, mask False).
- Футы -> метры, режимы фильтра земли (уже в issue), единый `grid_to_affine`, версия кода в ключе кэша.
- Бюджет: слои для участка платформы укладываются в 30 мин вместе с инференсом на 4090.

**DoD:** `python inference/entrypoint.py --data <сырой участок>` работает; профиль стадий в PR.
**Заменяет в костяке:** `predict.find_sites` / загрузку участка.
