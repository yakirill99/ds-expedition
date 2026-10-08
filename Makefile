.PHONY: data setup lint test test-web train predict submit runserver migrate makemigrations collectstatic djcheck
setup:
	uv sync --all-groups && uv run pre-commit install
lint:
	uv run ruff check . && uv run ruff format --check .
test:
	uv run pytest -q

test-web:
	uv run pytest tests/web -q
train:
	uv run python scripts/train.py --config configs/baseline.yaml
predict:
	uv run python scripts/predict.py --config configs/baseline.yaml
submit:
	uv run python scripts/make_submission.py --config configs/baseline.yaml
data:
	uv run python scripts/fetch_data.py

runserver:
	uv run python web/manage.py runserver 0.0.0.0:8000

migrate:
	uv run python web/manage.py migrate

makemigrations:
	uv run python web/manage.py makemigrations

collectstatic:
	uv run python web/manage.py collectstatic --noinput

djcheck:
	uv run python web/manage.py check
