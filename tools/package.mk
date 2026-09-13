# Shared quality targets for packages in this workspace.
# The including Makefile must define PACKAGE_NAME.

REPO_ROOT := $(abspath ../..)
UV_BASE := uv --cache-dir $(REPO_ROOT)/.uv-cache
UV := $(UV_BASE) --project $(REPO_ROOT)
TOOL_CONFIG := $(REPO_ROOT)/pyproject.toml

.PHONY: check coverage format lint lock sync test type

check: lock lint type coverage

coverage:
	$(UV) run --locked coverage erase
	$(UV) run --locked coverage run -m pytest $(TEST_ARGS)
	$(UV) run --locked coverage report

lock:
	$(UV) lock --check

sync:
	$(UV) sync --locked --all-packages --no-default-groups --group dev

lint:
	$(UV) run --locked ruff check --config $(TOOL_CONFIG) .
	$(UV) run --locked ruff format --config $(TOOL_CONFIG) --check .

format:
	$(UV) run --locked ruff check --config $(TOOL_CONFIG) --fix .
	$(UV) run --locked ruff format --config $(TOOL_CONFIG) .

test:
	$(UV) run --locked pytest $(TEST_ARGS)

type:
	$(UV) run --locked mypy --config-file $(TOOL_CONFIG) src tests
