# Репозиторий хакатона «Экспедиция. Data Science» — КОЗ №3

Команда: 5 человек, модульная разработка, Python, DevOps-подход (CI, защита веток, воспроизводимость).

---

## 1. Создание репозитория

1. GitHub → **New repository** → лучше создать **Organization** (бесплатно) — удобнее права и CODEOWNERS.
2. Имя: `expds-koz3`, **Private**, добавить `README`, `.gitignore (Python)`, лицензию — по правилам конкурса.
3. Settings → Collaborators/Teams → добавить 4 участников с ролью **Write**, 1–2 мейнтейнера — **Maintain/Admin**.

## 2. Структура проекта

```
expds-koz3/
├── .github/
│   ├── workflows/ci.yml          # линтер + тесты на каждый PR
│   ├── CODEOWNERS                # кто ревьюит какой модуль
│   └── pull_request_template.md
├── configs/                      # yaml-конфиги экспериментов
├── data/                         # в .gitignore! (raw/, interim/, processed/)
├── notebooks/                    # EDA, имена: 01_ivanov_eda.ipynb
├── src/expds/
│   ├── data/        # М1: загрузка, чтение, валидация данных
│   ├── features/    # М2: препроцессинг, признаки
│   ├── models/      # М3: модели по отдельным источникам
│   ├── fusion/      # М4: комплексирование (объединение моделей/источников)
│   └── evaluation/  # М5: метрики, валидация, формирование сабмита
├── tests/                        # pytest, зеркально src/
├── scripts/                      # train.py, predict.py, make_submission.py
├── Dockerfile
├── Makefile                      # make lint / test / train / submit
├── pyproject.toml                # зависимости (uv или poetry), настройки ruff
├── .pre-commit-config.yaml
└── README.md
```

**Модуль = владелец.** Каждый отвечает за свой каталог в `src/expds/`. Модули общаются через **чёткие интерфейсы** (функции/классы с зафиксированными входами-выходами, описанными в `README` модуля). Сначала договоритесь об интерфейсах — потом пишите код.

## 3. Ветки

| Ветка | Назначение | Правила |
|---|---|---|
| `main` | Только рабочие версии / сабмиты | Защищена, merge только из `develop` через PR, каждый релиз — тег |
| `develop` | Интеграция модулей | Защищена, merge через PR + 1 approve + зелёный CI |
| `feature/<модуль>-<задача>` | Разработка | От `develop`, живёт 1–2 дня, напр. `feature/fusion-stacking` |
| `exp/<автор>-<идея>` | Эксперименты, гипотезы | Можно не мержить; удачное → переносится в `feature/` |
| `hotfix/<суть>` | Срочная правка перед дедлайном | От `main`, мержится в `main` и `develop` |

Теги сабмитов: `v0.1-baseline`, `v0.2-lb0.83` — всегда можно откатиться к лучшему решению.

## 4. Защита веток (Settings → Branches → Rulesets)

Для `main` и `develop`:
- Require pull request, **1 approval** (для `main` — 2);
- Require status checks: `ci`;
- Запрет force push и удаления;
- Automatically delete head branches — включить.

## 5. CI — `.github/workflows/ci.yml`

```yaml
name: ci
on:
  pull_request:
    branches: [develop, main]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -q
```

GPU в CI не нужен: тесты гоняют на маленьких фикстурах и CPU. Обучение — локально (RTX 5070).

## 6. CODEOWNERS

```
/src/expds/data/        @user1
/src/expds/features/    @user2
/src/expds/models/      @user3
/src/expds/fusion/      @user4
/src/expds/evaluation/  @user5
```

## 7. Pre-commit (локально у каждого)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{id: ruff, args: [--fix]}, {id: ruff-format}]
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks: [{id: nbstripout}]      # чистит выводы ноутбуков → меньше конфликтов
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks: [{id: check-added-large-files, args: [--maxkb=5000]}]
```

Установка: `uv run pre-commit install`.

## 8. Данные, модели, эксперименты

- Данные и веса **не коммитим**. Варианты: **DVC** + общее хранилище (Google Drive/S3) или общий диск с фиксированными путями в `configs/`.
- Эксперименты логируем в **MLflow** или **W&B**: конфиг, seed, метрика, git-коммит.
- Фиксируем `seed` и версии зависимостей (`uv.lock` коммитится).
- Время обучения на RTX 5070 записывать — для пересчёта на 4090 через коэффициент.

## 9. Рабочий цикл (каждый участник)

```bash
git switch develop && git pull
git switch -c feature/models-catboost
# ... код + тесты ...
git add -p && git commit -m "feat(models): catboost baseline"
git push -u origin feature/models-catboost
# → PR в develop, ревью владельцем соседнего модуля, CI зелёный → Squash merge
```

Коммиты по **Conventional Commits**: `feat`, `fix`, `refactor`, `test`, `docs`, `exp`.

## 10. Процесс (DevOps-минимум для хакатона)

- **Issues + Projects (Kanban)**: To Do / In Progress / Review / Done; каждая задача → issue → ветка → PR (`Closes #12`).
- Короткий синк раз в день: что влито в `develop`, что блокирует.
- Интеграция в `develop` — **минимум раз в день**, не копить большие PR.
- Перед сабмитом: `develop` → PR в `main` → тег → `make submit`.

## 11. Чек-лист первого дня

- [ ] Репозиторий, участники, rulesets на `main`/`develop`
- [ ] Структура каталогов + `pyproject.toml` + `uv.lock`
- [ ] CI, pre-commit, CODEOWNERS, PR-шаблон
- [ ] Интерфейсы между модулями описаны в README
- [ ] Бейзлайн end-to-end (пусть слабый) → тег `v0.1-baseline`
