# Архив для платформы

Платформа (Приложение 2 Технического регламента КОЗ №3) принимает zip из **двух
файлов в корне**:

```
solution.py      python solution.py <input_dir> <output.geojson>
metadata.json    {"image": "<образ>", "entry_point": "python solution.py"}
```

Код (`expds`), `config.yaml` и `weights.pt` живут в Docker-образе под `/app`.
Поэтому архив весит килобайты. Это **не** `make_submission.py` — тот собирает
self-contained архив на 87 МБ для локального прогона.

## Быстрый старт

```bash
uv run python scripts/build_platform_solution.py
uv run python scripts/build_platform_solution.py --check dist/solution_platform.zip
```

Готовый файл — `dist/solution_platform.zip`, его и заливать.

## Зачем пост-патч

`postprocess` гоняет для каждого класса обе ветки: полигональную и точечную.
Шесть классов из семи размечены полигонами, и точечное предсказание по ним не
проходит IoU ≥ 0.5 с полигоном GT — то есть даёт только FP. Замер на oracle
(идеальные таргеты, 9 участков, 437 GT):

| ветки | tp | fp | P | R | F1 |
|---|---|---|---|---|---|
| обе | 232 | 52 | 0.817 | 0.531 | 0.644 |
| полигоны | 226 | 4 | 0.983 | 0.517 | 0.678 |

Heatmap-ветка = **92.3% всех FP при 2.6% всех TP**.

Правку нельзя сделать через `config.yaml` в архиве: `PostConfig` в образе не
знает ключа `branches` и падает на unknown keys. Поэтому скрипт вставляет в
`solution.py` блок, который подменяет `heatmap_to_points` в момент импорта —
до первого вызова. Патч обёрнут в `try/except`: если структура образа другая,
он печатает `post-patch: SKIPPED` и пайплайн отрабатывает как раньше.

По логу платформы видно, сработал ли патч:

```
post-patch v1: heatmap выключен, точечные классы ('finds_points',)
```

## Флаги

| флаг | по умолчанию | смысл |
|---|---|---|
| `--out` | `dist/solution_platform.zip` | куда положить архив |
| `--image` | `yakirill99/ds-expedition:latest` | образ с кодом и весами |
| `--entry-point` | `python solution.py` | команда запуска в `metadata.json` |
| `--point-classes` | `finds_points` | классы, где heatmap остаётся включённой |
| `--no-post-patch` | — | архив без патча, `solution.py` как есть |
| `--check` | — | только проверить готовый архив |

Отключить heatmap везде (включая `finds_points`) — `--point-classes ""`.
Исключить её и для `object_poly` — `--point-classes finds_points,object_poly`.

## Проверка перед отправкой

```bash
uv run python scripts/build_platform_solution.py --check dist/solution_platform.zip
```

Ожидаемый вывод:

```
OK: solution.py компилируется
OK: файлы ['solution.py', 'metadata.json']
OK: metadata {'image': '...', 'entry_point': 'python solution.py'}
OK: пост-патч на месте
     вызов: 2 (ожидается 2)
OK: размер 4627 байт
```

`WARN: пост-патч отсутствует` — значит архив собран с `--no-post-patch`.

## Когда образ надо пересобирать

Пост-патч — временная мера. Он живёт до тех пор, пока в образе нет правки
`vectorize.py` с маршрутизацией ветвей по классам. После пересборки образа
скрипт можно запускать с `--no-post-patch`, и тогда `solution.py` уходит как
есть.

Патч не видит поля `PostConfig`, поэтому не конфликтует с будущей правкой: если
`postprocess` начнёт сам пропускать ненужные ветки, подмена останется
безвредной (лишний вызов вернёт тот же пустой список).

## Тесты

```bash
uv run pytest -q tests/test_build_platform_solution.py
```

Покрыто: состав архива, контракт `metadata.json`, компилируемость,
идемпотентность (три прогона → одинаковый md5), однократность блока,
`--no-post-patch`, встраивание `--point-classes`, `--check` на хорошем и на
чужом архиве.
