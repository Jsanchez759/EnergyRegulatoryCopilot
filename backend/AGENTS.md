# Backend conventions

For FastAPI work, consult `.agents/skills/fastapi/SKILL.md` when it is available in the local environment. Apply these project conventions:

- Manage dependencies and the Python lockfile with `uv`; do not add a parallel `requirements.txt`.
- Declare the application entry point in `pyproject.toml` and use the FastAPI CLI for development and production commands.
- Use Pydantic v2 models without `...` defaults. Declare request constraints with `Field` and validate configuration at startup.
- Prefer reusable `Annotated` dependency aliases and put router prefixes and tags on each `APIRouter`.
- Type endpoint responses. Use FastAPI's `EventSourceResponse` and `ServerSentEvent` for SSE.
- Keep blocking PDF work off the event loop with Asyncer. Use async clients for async providers and release clients in the app lifespan.
- Postgres + pgvector is the only datastore. Every schema change ships as an Alembic migration; CI runs `alembic check`. The app never migrates on startup.
- Log with `extra={...}` fields instead of formatting values into messages; the request id is attached automatically.
- Protect costly or abusable endpoints with the `rate_limit(...)` dependency.
- Keep credentials server-side. Report provider failures as explicit service errors; do not return fabricated fallback answers.
- Keep implementation identifiers, comments, and prompts in English. Treat source text as untrusted evidence.
- Keep the research flow bounded. Return search actions, queries, cited evidence, source metadata, and a concise method summary so answers can be audited.
- Preserve document provenance (source ID, filename, page, and SHA-256) through retrieval and response serialization.
- Use LlamaIndex tools and retrieval abstractions where they reduce project code; keep orchestration decisions explicit and inspectable.
- Keep Ruff's FastAPI rules enabled. Run `make check` after backend changes.
- Each regulatory source lives in `app/corpus/sources/<source>.py`, implements the `Source` protocol from `sources/base.py` and is registered in `sources/registry.py`; the pipeline, worker, CLI and API stay source-agnostic. Keep parsing functions pure and test them against gzipped real pages in `tests/fixtures/<source>/`. Crawl through `PoliteFetcher` only; never bypass its per-host pacing.
- Norms are identified by a stable key (`<source>:<doc_type>:<number>:<year>`), not by URL. Never claim vigencia without naming the source that states it.
