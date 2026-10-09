# ds-expedition — КОЗ №3 «Комплексирование»

Команда: 5 человек. Правила работы — в `docs/REPO_SETUP.md`.

## Быстрый старт
```bash
git clone <url> && cd ds-expedition
make setup      # uv sync + pre-commit
make test
```

## Требования к окружению
- Linux/Windows: сборка PyTorch cu130, нужен драйвер NVIDIA ≥ 580 и GPU не старше Turing (RTX 20xx и новее).
- macOS: CPU-сборка, обучение на GPU недоступно.

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
