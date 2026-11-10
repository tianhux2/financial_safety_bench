.PHONY: install data validate test lint

install:
	uv sync --dev

data:
	uv run python safety_bench/build/build_dataset.py

validate:
	uv run python safety_bench/build/validate_dataset.py

test:
	uv run pytest tests

lint:
	uv run ruff check .
