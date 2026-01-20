.PHONY: install test train lint clean all

# Variables
PYTHON = poetry run python
PYTEST = poetry run pytest
BLACK = poetry run black
ISORT = poetry run isort
FLAKE8 = poetry run flake8

# Install dependencies
install:
	poetry install

# Run tests
test:
	$(PYTEST) tests/

# Train the model
train:
	$(PYTHON) scripts/train.py

# Run linting and formatting
lint:
	$(BLACK) src/ scripts/ tests/
	$(ISORT) src/ scripts/ tests/
	$(FLAKE8) src/ scripts/ tests/

# Clean up cache and build artifacts
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf build
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Run everything
all: install lint test train
