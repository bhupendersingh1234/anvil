.PHONY: install test lint format deploy app run clean

install:
	pip install -e ".[all]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

deploy:
	modal deploy -m anvil.service

run:
	anvil run

app:
	anvil app

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
