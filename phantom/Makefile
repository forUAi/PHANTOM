.PHONY: help install dev test lint format clean docker docs

PYTHON := python3.11
VENV := venv
BIN := $(VENV)/bin

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $1, $2}'

$(VENV)/bin/activate:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel

install: $(VENV)/bin/activate  ## Install PHANTOM
	$(BIN)/pip install -e .

dev: $(VENV)/bin/activate  ## Install with dev dependencies
	$(BIN)/pip install -e ".[dev,vision,docs]"
	$(BIN)/pre-commit install

test:  ## Run tests
	$(BIN)/pytest tests/ -v --cov=phantom_core --cov-report=term-missing

test-cuda:  ## Run CUDA tests
	$(BIN)/pytest tests/ -v -m cuda

test-mps:  ## Run MPS tests
	$(BIN)/pytest tests/ -v -m mps

lint:  ## Run linters
	$(BIN)/ruff check phantom_core phantom_cli tests
	$(BIN)/mypy phantom_core phantom_cli

format:  ## Format code
	$(BIN)/black phantom_core phantom_cli tests
	$(BIN)/ruff check --fix phantom_core phantom_cli tests

clean:  ## Clean build artifacts
	rm -rf build dist *.egg-info
	rm -rf reports/* runs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker:  ## Build Docker image
	docker build -t phantom:latest --target production .
	docker build -t phantom:dev --target development .

docker-run:  ## Run PHANTOM in Docker
	docker run --rm --gpus all \
		-v $(PWD)/examples:/home/phantom/examples \
		-v $(PWD)/reports:/home/phantom/reports \
		phantom:latest phantom trace examples/resnet_demo.py --target=cuda

docs:  ## Build documentation
	$(BIN)/sphinx-build -b html docs docs/_build

release:  ## Create release
	$(BIN)/python -m build
	$(BIN)/twine check dist/*
	@echo "Ready to upload with: twine upload dist/*"

benchmark:  ## Run benchmarks
	@echo "Running CPU benchmark..."
	$(BIN)/phantom profile examples/resnet_demo.py --targets cpu --iterations 100
	@echo "\nRunning CUDA benchmark..."
	$(BIN)/phantom profile examples/resnet_demo.py --targets cuda --iterations 100
	@echo "\nRunning MPS benchmark (if available)..."
	$(BIN)/phantom profile examples/resnet_demo.py --targets mps --iterations 100 || true