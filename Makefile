.PHONY: data setup lint test train predict submit
setup:
	uv sync --all-groups && uv run pre-commit install
lint:
	uv run ruff check . && uv run ruff format --check .
test:
	uv run pytest -q
train:
	uv run python scripts/train.py --config configs/baseline.yaml
predict:
	uv run python scripts/predict.py --config configs/baseline.yaml
submit:
	uv run python scripts/make_submission.py --config configs/baseline.yaml
data:
	uv run python scripts/fetch_data.py
