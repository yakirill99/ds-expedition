# 03-data-geo: build_site, автосборка сырых лидарных участков, профиль стадий

Распаковывать в корень репо на ветке `feat/data-geo` (после мержа PR #33).
`unzip -o` перезапишет 5 изменённых файлов и добавит 2 новых.

## Что внутри

Новые:
- `src/expds/data/build_site.py` — `build_site(raw_dir, out_dir, cfg)`: сырой
  лидарный участок (LAS/LAZ) -> `layers/*.tif` + `samples/` (формат костяка).
  Без YAML-путей внутри: все параметры в AppConfig (секции роли 1 того же
  конфига). `resolve_sites(data, cfg)` — замена `predict.find_sites` для
  инференса: готовые участки проходят, сырые лидарные собираются in-place,
  растровые без лидара -> `RawSiteUnsupportedError` с объяснением (семантика
  каналов не согласована — docs/issues/03, статус 04.10.2026).
- `tests/test_build_site.py` — 10 тестов (torch-free): kind-классификация,
  e2e сборка из синтетического LAS (laspy), порядок каналов dtm->рельеф->cloud
  (CONTRACTS.md §3), idempotentность/force, растровые/пустые ошибки,
  resolve_sites (одиночный/папка/смешанная/вложенный LAS).

Изменённые (перезаписать):
- `scripts/build_layers.py` — 528 -> 100 строк: тонкий CLI над
  `build_site` (пайплайн переехал в библиотеку). Убраны мёртвые `--no-cache`,
  `--only` и ДУБЛЬ `_save_sample` (второе определение молча затеняло первое
  и писало sample без channel_mask). Флаг `--force` вместо `--no-cache`.
- `src/expds/predict.py` — `run()` использует `resolve_sites` (стадия
  «участки» в профиле); `main()` перехватывает `RawSiteUnsupportedError`
  -> код 2 и чистое сообщение; find_sites не тронут (совместимость).
- `tools/make_synthetic.py` — вместо importlib-хака на scripts/build_layers.py
  импортирует `expds.data.build_site._save_sample` (тот же формат sample).
- `docs/SKELETON.md`, `docs/CONTRACTS.md` — актуализация строк про entrypoint
  и про место `_order_channel_names`.

## Проверки

    make lint
    uv run pytest -q        # ожидаем 443 passed (433 + 10)

## DoD-смок на реальном участке (013)

    # сырой участок -> автосборка слоёв -> инференс -> GeoJSON:
    uv run python inference/entrypoint.py \
        --data <путь/к/013_Нора_Виннон> \
        --config <конфиг_роли1_для_013.yaml> \
        --weights <runs/.../best.pt> \
        --out runs/pred_013.geojson

Правило раскладки: LAS должен лежать ПРЯМО в папке участка (если глубже —
укажите --data на папку с LAS). Повторный запуск слои не пересобирает
(idempotent), `scripts/build_layers.py --force` пересобирает.

Профиль стадий entrypoint печатает таблицей в конце — перенести в PR.

## Замечание

`inference/config.yaml` и `inference/weights.pt` — артефакты сабмита, в git
не входят: при смоке указывайте --config/--weights явно.
