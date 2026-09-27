# Git в проекте ds-expedition: как это работает и что делать

## 1. Модель: три места, где живёт код

```
рабочая папка  --git add-->  индекс (staging)  --git commit-->  локальный репозиторий  --git push-->  GitHub (origin)
   ~/ds-expedition                                                  .git/                          github.com/yakirill99/ds-expedition
```

- **Рабочая папка** — файлы, которые вы редактируете.
- **Индекс** — список того, что войдёт в *следующий* коммит. `git add` кладёт файл сюда. Если после `git add` файл снова изменился (как это сделал pre-commit), в коммит попадёт **старая** версия — нужно `git add` повторно.
- **Локальный репозиторий** — история коммитов на вашей машине. `git commit` создаёт снимок из индекса.
- **origin** — копия на GitHub. `git push` отправляет туда ваши коммиты, `git pull` забирает чужие.

`git status` всегда показывает, в каком из этих мест что лежит. Читайте его после каждого шага.

## 2. Ветки

Ветка — указатель на цепочку коммитов. Переключение (`git switch`) меняет содержимое рабочей папки.

| Ветка | Что это | Кто пишет напрямую |
|---|---|---|
| `main` | Только рабочие версии, каждая с тегом сабмита | Никто. Только PR из `develop` |
| `develop` | Общая интеграционная ветка, «текущее состояние проекта» | Никто. Только PR из feature/exp |
| `feature/<модуль>-<задача>` | Одна задача одного человека | Автор |
| `exp/<автор>-<идея>` | Эксперимент, может умереть | Автор |
| `hotfix/<суть>` | Срочная правка перед дедлайном | Автор, от `main` |

Правило «никто напрямую» сейчас держится **только на дисциплине** — rulesets на приватном Free-репо не работают. Технически `git push origin develop` пройдёт, но так не делаем.

## 3. Полный цикл задачи (наизусть)

```bash
# 0. Свежий develop
git switch develop
git pull

# 1. Своя ветка
git switch -c feature/models-catboost

# 2. Работа. Периодически:
git status                       # что изменилось
git diff                         # построчно
git add src/expds/models/cat.py tests/test_cat.py
git commit -m "feat(models): catboost baseline"

#    Если pre-commit что-то починил и написал "Failed ... files were modified":
git add -u                       # добавить все изменённые отслеживаемые файлы
git commit -m "feat(models): catboost baseline"   # повторить ту же команду

# 3. Отправить ветку
git push -u origin feature/models-catboost        # -u только в первый раз, дальше просто git push

# 4. PR в вебе: ссылка из вывода push  → base: develop → описание → Create pull request
#    Ждём зелёный ci (1–2 мин). Ревью. Squash and merge. Delete branch.

# 5. Прибраться
git switch develop
git pull
git branch -d feature/models-catboost
```

### Формат коммита (Conventional Commits)

`<тип>(<модуль>): <что сделано>` — на английском, повелительное наклонение, без точки.

Типы: `feat` новое, `fix` починка, `refactor` без изменения поведения, `test`, `docs`, `style` только форматирование, `chore` инфраструктура, `exp` эксперимент.

Примеры: `feat(fusion): stacking over modality experts`, `fix(data): wrong CRS in raster loader`, `exp(models): try focal loss`.

## 4. Pre-commit: что это и почему прерывает коммит

`uv run pre-commit install` положил в `.git/hooks/pre-commit` скрипт. Он запускается перед каждым `git commit` и гоняет:

| Хук | Что делает | Может изменить файлы |
|---|---|---|
| `ruff` | линтер: неиспользуемые импорты, ошибки | да (авто-фикс) |
| `ruff-format` | форматирование (как black) | да |
| `nbstripout` | чистит выводы из `.ipynb` | да |
| `check-added-large-files` | блокирует файлы > 5 МБ | нет |
| `end-of-file-fixer` | пустая строка в конце файла | да |

Поведение: если хук **изменил** файл — коммит отменяется, чтобы вы увидели правку. Действие всегда одно: `git add -u && git commit` повторно. Если хук **не смог** починить (например, файл > 5 МБ) — уберите проблему и повторите.

Прогнать вручную на всех файлах: `uv run pre-commit run --all-files`.
Пропустить хук (только для аварий): `git commit --no-verify`. CI всё равно поймает.

## 5. CI (GitHub Actions)

Файл `.github/workflows/ci.yml`. Запускается на каждый PR в `develop`/`main` и на пуш в них. Шаги: `uv sync` → `ruff check` → `ruff format --check` → `pytest`.

Красный CI → открыть job на GitHub, найти шаг с ✗, прочитать лог. Почти всегда это одно из:
- `ruff format --check` упал → локально `uv run ruff format .` → commit → push.
- `ruff check` упал → `uv run ruff check --fix .`, остальное править руками.
- `pytest` упал → `uv run pytest -q` локально, чинить.

Перед пушем полезно прогнать всё сразу: `make lint && make test`.

## 6. Pull Request

PR — это предложение «влить ветку X в ветку Y», с diff'ом, обсуждением и статусом проверок. Смысл для команды из пяти: (а) второй человек видит код до того, как он попал в общую ветку; (б) CI прогнал тесты; (в) есть связь с задачей (`Closes #12` в описании закроет issue автоматически).

Как открыть: после `git push` в терминале строка `remote: https://github.com/.../pull/new/<ветка>` — открыть в браузере. Либо на странице репо кнопка «Compare & pull request».

**Squash and merge** — все коммиты ветки схлопываются в один в `develop`. История чистая, откат простой. Merge commit и Rebase не используем.

Обновить PR: просто `git commit` + `git push` в ту же ветку. PR подхватит сам, CI перезапустится.

## 7. Теги (сабмиты)

Каждая отправка на платформу — тег на `main`:
```bash
git switch main && git pull
git tag -a v0.2-lb0.83 -m "second submit, stacking"
git push origin v0.2-lb0.83
```
Вернуться к коду сабмита: `git switch --detach v0.2-lb0.83` (посмотреть) или `git switch -c hotfix/from-v0.2 v0.2-lb0.83` (править).

## 8. Данные

В git — только `data/README.md` (реестр). Файлы на Яндекс Диске по публичным ссылкам, `make data` скачивает всё, `uv run python scripts/fetch_data.py --only <id>` — одну позицию. `data/**` в `.gitignore`; хук не даст закоммитить > 5 МБ.

## 9. Шпаргалка команд

**Смотреть**
```bash
git status                     # состояние
git log --oneline --graph -20  # история
git diff                       # незакоммиченные изменения
git diff --staged              # что войдёт в коммит
git branch -a                  # все ветки
git show <hash>                # что в коммите
git blame <файл>               # кто менял строку
```
**Ветки**
```bash
git switch <ветка>             # переключиться
git switch -c <новая>          # создать и переключиться
git branch -d <ветка>          # удалить локально (слитую)
git branch -D <ветка>          # удалить принудительно
git push origin --delete <ветка>   # удалить на GitHub
git fetch --prune              # подтянуть список веток, убрать удалённые
```
**Синхронизация**
```bash
git pull                       # = fetch + merge текущей ветки
git push                       # отправить
git push -u origin <ветка>     # первый пуш новой ветки
git fetch                      # скачать, не применять
```
**Откаты**
```bash
git restore <файл>             # отменить незакоммиченные правки файла
git restore --staged <файл>    # убрать из индекса, правки оставить
git reset --soft HEAD~1        # отменить последний коммит, изменения в индексе
git reset --hard HEAD~1        # отменить последний коммит и изменения (необратимо)
git revert <hash>              # новый коммит, отменяющий старый (безопасно для общих веток)
git stash / git stash pop      # спрятать правки / вернуть
```
**Обновить ветку из develop** (если develop ушёл вперёд, пока вы работали)
```bash
git switch feature/x
git merge develop              # или git rebase develop — для своей неопубликованной ветки
```

## 10. Троблшутинг

**«files were modified by this hook», коммит не создался**
Норма. `git add -u && git commit -m "..."`.

**`git push` → `rejected ... fetch first`**
Кто-то запушил в вашу ветку (или вы с другой машины). `git pull`, разрешить конфликты, `git push`.

**Конфликт при merge/pull**
В файле появятся `<<<<<<<`, `=======`, `>>>>>>>`. Открыть, оставить нужный вариант, удалить маркеры, `git add <файл>`, `git commit`. Отменить всё: `git merge --abort`.

**Закоммитил не в ту ветку (например, в develop)**
```bash
git switch -c feature/oops      # унести коммит в новую ветку
git switch develop
git reset --hard origin/develop # вернуть develop к состоянию на GitHub
```

**Закоммитил большой файл / данные**
Если ещё не пушили: `git reset --soft HEAD~1`, убрать файл из индекса `git restore --staged <файл>`, добавить в `.gitignore`, закоммитить заново. Если запушили — сказать в чат, история чистится через `git filter-repo`, лучше вместе.

**`Permission denied (publickey)`**
Нет SSH-ключа на этой машине или он не добавлен в GitHub. `ssh -T git@github.com` для проверки; ключ: `ssh-keygen -t ed25519`, публичный `~/.ssh/id_ed25519.pub` → GitHub → Settings → SSH keys.

**`Connection timed out` port 22**
Сеть WSL. `wsl --shutdown` в PowerShell; проверить `.wslconfig` — `networkingMode=mirrored` в секции `[wsl2]`. Запасной путь: `git remote set-url origin https://github.com/yakirill99/ds-expedition.git` + Personal Access Token вместо пароля.

**Ноутбук даёт огромный diff**
nbstripout должен чистить выводы. Если не сработал: `uv run nbstripout notebooks/*.ipynb`. Ноутбуки — только для EDA, код переносить в `src/`.

**`uv: command not found` после установки**
`source ~/.local/bin/env` или перезапустить терминал.

**Нужно посмотреть чужую ветку**
`git fetch && git switch feature/ivanov-x` — ветка появится локально.

**Всё сломалось, хочу чистый develop**
```bash
git stash            # спрятать своё, если жалко
git switch develop
git fetch origin
git reset --hard origin/develop
```

**Удалил файл, нужно вернуть**
`git restore <файл>` (если был закоммичен), либо `git checkout <hash> -- <файл>` из старого коммита.

## 11. Настройки git один раз на машину

```bash
git config --global user.name "Кирилл Ястребов"
git config --global user.email "<email из GitHub>"
git config --global pull.rebase false
git config --global init.defaultBranch main
git config --global core.autocrlf input      # важно в WSL: не тащить CRLF из Windows
```
