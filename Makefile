.PHONY: install dev test lint format build clean help

help:
	@echo "Available commands:"
	@echo "  make install     - Install the package"
	@echo "  make dev         - Install development dependencies"
	@echo "  make test        - Run tests with coverage"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code with black and isort"
	@echo "  make build       - Build the package"
	@echo "  make clean       - Clean build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=ming --cov-report=term-missing --cov-report=html

test-quick:
	pytest tests/ -v -x

lint:
	ruff check ming/
	black --check ming/
	isort --check-only ming/

format:
	black ming/
	isort ming/

typecheck:
	mypy ming/ --ignore-missing-imports

build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check: lint test
	@echo "All checks passed!"

pre-commit: format lint test
	@echo "Pre-commit checks passed!"
