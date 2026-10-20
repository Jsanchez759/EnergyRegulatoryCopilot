UV ?= uv
NPM ?= npm
PYTHON ?= python3
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 5173
API_BASE_URL ?= http://$(API_HOST):$(API_PORT)/api/v1

.PHONY: help doctor setup sync lock db migrate dev backend serve frontend preview lint format format-check test e2e up down build check health

help: ## Show available commands
	@printf '\nEnerCol development commands\n\n'
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  make %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\nOverride hosts or ports, for example: make dev API_PORT=8010 WEB_PORT=5180\n\n'

doctor: ## Check local tools and required environment files
	@command -v $(UV) >/dev/null || { echo "Missing uv: https://docs.astral.sh/uv/"; exit 1; }
	@command -v $(NPM) >/dev/null || { echo "Missing npm (install Node.js 18+)."; exit 1; }
	@command -v $(PYTHON) >/dev/null || { echo "Missing $(PYTHON)."; exit 1; }
	@command -v docker >/dev/null || { echo "Missing Docker (needed for Postgres)."; exit 1; }
	@test -f backend/.env || { echo "Missing backend/.env with OPENROUTER_API_KEY."; exit 1; }
	@echo "Tools and backend/.env are ready."

setup: ## Install backend and frontend dependencies
	$(UV) sync --directory backend --locked --group dev
	$(NPM) ci --prefix frontend

sync: ## Sync Python dependencies from uv.lock
	$(UV) sync --directory backend --locked --group dev

lock: ## Resolve and update uv.lock
	$(UV) lock --directory backend

db: ## Start local Postgres with pgvector (docker compose service `db`)
	docker compose up -d --wait db

migrate: db ## Apply database migrations (a deploy step, never run by the app itself)
	$(UV) run --directory backend alembic upgrade head

dev: migrate ## Start backend and frontend together; Ctrl-C stops both
	UV="$(UV)" NPM="$(NPM)" API_HOST="$(API_HOST)" API_PORT="$(API_PORT)" WEB_HOST="$(WEB_HOST)" WEB_PORT="$(WEB_PORT)" API_BASE_URL="$(API_BASE_URL)" $(PYTHON) scripts/dev.py

backend: migrate ## Start the FastAPI development server
	$(UV) run --directory backend fastapi dev --host $(API_HOST) --port $(API_PORT)

serve: ## Start the FastAPI production server
	$(UV) run --directory backend fastapi run --host $(API_HOST) --port $(API_PORT)

frontend: ## Start only the Vite frontend
	VITE_API_BASE_URL="$(API_BASE_URL)" $(NPM) --prefix frontend run dev -- --host $(WEB_HOST) --port $(WEB_PORT)

preview: ## Preview the production frontend build
	$(NPM) --prefix frontend run preview -- --host $(WEB_HOST) --port $(WEB_PORT)

lint: ## Lint backend (Ruff) and frontend (ESLint)
	$(UV) run --directory backend ruff check app alembic tests
	$(NPM) --prefix frontend run lint

format: ## Format backend Python files
	$(UV) run --directory backend ruff format app alembic tests

format-check: ## Check backend formatting without changing files
	$(UV) run --directory backend ruff format --check app alembic tests

test: db ## Run backend and frontend tests
	$(UV) run --directory backend pytest
	$(NPM) --prefix frontend test

up: ## Build and start the Docker stack (web :8080, api :8000)
	docker compose up -d --build

down: ## Stop the Docker stack
	docker compose down

e2e: ## Run Playwright end-to-end tests against the running stack (make up first)
	$(NPM) --prefix frontend run e2e

build: ## Build the frontend for production
	$(NPM) --prefix frontend run build

check: ## Lockfile, lint, format, tests, and frontend production build
	$(UV) lock --directory backend --check
	$(MAKE) lint format-check test
	$(PYTHON) -m py_compile scripts/dev.py
	$(NPM) --prefix frontend run build

health: ## Check the local backend health endpoint
	curl --fail --silent --show-error http://$(API_HOST):$(API_PORT)/api/v1/health
	@printf '\n'
