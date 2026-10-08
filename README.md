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
| web | @ | Django-интерфейс: API, задачи, визуализация |

## Django-веб (`web/`)

```bash
cp .env.example .env
uv sync --all-groups
make migrate
make runserver
```

- `web/config/` — настройки проекта.
- `web/apps/api/` — REST API (`/api/health/`, `/api/predict/`).
- `web/apps/jobs/` — management-команды (`train`, `predict`) и модель `InferenceJob`.
- `web/apps/viz/` — заглушка под визуализацию.
- Тесты — в `tests/web/`.

Полезные команды:
- `make runserver` — запуск dev-сервера.
- `make migrate` / `make makemigrations` — миграции.
- `make djcheck` — проверка настроек.
- `make test-web` — Django-тесты.

## Ветки
`main` ← `develop` ← `feature/<модуль>-<задача>`, `exp/<автор>-<идея>`.
