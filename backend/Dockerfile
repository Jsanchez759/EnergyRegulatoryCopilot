FROM python:3.13-slim AS build

# chroma-hnswlib has no prebuilt wheel for every platform, so a C++ toolchain is needed here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY --from=build /opt/venv /opt/venv

COPY pyproject.toml alembic.ini ./
COPY alembic ./alembic
COPY app ./app

RUN useradd --create-home --uid 10001 enercol \
    && mkdir -p /app/storage \
    && chown -R enercol:enercol /app/storage
USER enercol

ENV DATABASE_PATH=/app/storage/enercol.db \
    VECTOR_DB_PATH=/app/storage/chroma
VOLUME ["/app/storage"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000"]
