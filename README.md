# ds-expedition — КОЗ №3 «Комплексирование»

Команда: 5 человек. Правила работы — в `docs/REPO_SETUP.md`.

## Быстрый старт
```bash
git clone <url> && cd ds-expedition
make setup      # uv sync + pre-commit
make test
```

## Модули (`src/expds/`)
| Модуль | Владелец | Назначение |
|---|---|---|
| data | @ | загрузка, валидация |
| features | @ | препроцессинг, признаки |
| models | @ | модели по источникам |
| fusion | @ | комплексирование |
| evaluation | @ | метрики, сабмит |

## Ветки
`main` ← `develop` ← `feature/<модуль>-<задача>`, `exp/<автор>-<идея>`.
