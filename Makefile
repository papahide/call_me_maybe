PYTHON  = python3
PACKAGE    = src
UV      = uv
PIP		= pip

.PHONY: install run debug lint lint-strict clean

install:
	@echo "Setting up environment with uv..."
	$(UV) sync
	@echo "Done."

run:
	$(UV) run $(PYTHON) -m $(PACKAGE)

debug:
	$(UV) run $(PYTHON) -m pdb -m $(PACKAGE)

lint:

	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores \
	                 --ignore-missing-imports \
	                 --disallow-untyped-defs \
	                 --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .venv dist
	@echo "Cleaned."