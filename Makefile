UV ?= uv
NPM ?= npm
PYTHON ?= python3
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 5173
API_BASE_URL ?= http://$(API_HOST):$(API_PORT)/api/v1

.PHONY: help doctor setup sync lock dev backend serve frontend preview lint format format-check build check health

help: ## Show available commands
	@printf '\nEnerCol development commands\n\n'
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  make %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\nOverride hosts or ports, for example: make dev API_PORT=8010 WEB_PORT=5180\n\n'

doctor: ## Check local tools and required environment files
	@command -v $(UV) >/dev/null || { echo "Missing uv: https://docs.astral.sh/uv/"; exit 1; }
	@command -v $(NPM) >/dev/null || { echo "Missing npm (install Node.js 18+)."; exit 1; }
	@command -v $(PYTHON) >/dev/null || { echo "Missing $(PYTHON)."; exit 1; }
	@test -f backend/.env || { echo "Missing backend/.env with OPENROUTER_API_KEY."; exit 1; }
	@echo "Tools and backend/.env are ready."

setup: ## Install backend and frontend dependencies
	$(UV) sync --directory backend --locked --group dev
	$(NPM) ci --prefix frontend

sync: ## Sync Python dependencies from uv.lock
	$(UV) sync --directory backend --locked --group dev

lock: ## Resolve and update uv.lock
	$(UV) lock --directory backend

dev: ## Start backend and frontend together; Ctrl-C stops both
	UV="$(UV)" NPM="$(NPM)" API_HOST="$(API_HOST)" API_PORT="$(API_PORT)" WEB_HOST="$(WEB_HOST)" WEB_PORT="$(WEB_PORT)" API_BASE_URL="$(API_BASE_URL)" $(PYTHON) scripts/dev.py

backend: ## Start the FastAPI development server
	$(UV) run --directory backend fastapi dev --host $(API_HOST) --port $(API_PORT)

serve: ## Start the FastAPI production server
	$(UV) run --directory backend fastapi run --host $(API_HOST) --port $(API_PORT)

frontend: ## Start only the Vite frontend
	VITE_API_BASE_URL="$(API_BASE_URL)" $(NPM) --prefix frontend run dev -- --host $(WEB_HOST) --port $(WEB_PORT)

preview: ## Preview the production frontend build
	$(NPM) --prefix frontend run preview -- --host $(WEB_HOST) --port $(WEB_PORT)

lint: ## Lint the backend with Ruff
	$(UV) run --directory backend ruff check app alembic

format: ## Format backend Python files
	$(UV) run --directory backend ruff format app alembic

format-check: ## Check backend formatting without changing files
	$(UV) run --directory backend ruff format --check app alembic

build: ## Build the frontend for production
	$(NPM) --prefix frontend run build

check: ## Check lockfile, backend lint/format, and frontend production build
	$(UV) lock --directory backend --check
	$(UV) run --directory backend ruff check app alembic
	$(UV) run --directory backend ruff format --check app alembic
	$(UV) run --directory backend python -m compileall -q app alembic
	$(PYTHON) -m py_compile scripts/dev.py
	$(NPM) --prefix frontend run build

health: ## Check the local backend health endpoint
	curl --fail --silent --show-error http://$(API_HOST):$(API_PORT)/api/v1/health
	@printf '\n'
