import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.observability import init_sentry
from app.core.request_context import RequestContextMiddleware
from app.corpus.storage import LocalBlobStorage
from app.db import SessionFactory, engine
from app.services.agent_service import AgentService
from app.services.corpus_service import CorpusService
from app.services.document_service import DocumentService
from app.services.embeddings import EmbeddingClient

setup_logging()
init_sentry()
logger = logging.getLogger("app")


def is_allowed_origin(origin: str) -> bool:
    if origin in settings.ALLOWED_ORIGINS:
        return True
    return bool(
        settings.ALLOWED_ORIGIN_REGEX and re.fullmatch(settings.ALLOWED_ORIGIN_REGEX, origin)
    )


def create_application() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.embeddings = EmbeddingClient()
        application.state.storage = LocalBlobStorage(settings.CORPUS_STORAGE_DIR)
        application.state.document_service = DocumentService(SessionFactory)
        application.state.corpus_service = CorpusService(
            SessionFactory, application.state.embeddings
        )
        application.state.agent_service = AgentService(
            application.state.document_service, application.state.corpus_service
        )
        logger.info("application started", extra={"version": settings.VERSION})
        try:
            yield
        finally:
            await application.state.agent_service.close()
            await application.state.document_service.close()
            await application.state.embeddings.close()
            await engine.dispose()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Backend API for the Energy Regulatory Copilot project.",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    @app.middleware("http")
    async def validate_request_origin(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and not is_allowed_origin(origin):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "El origen de la solicitud no está autorizado."},
                )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_origin_regex=settings.ALLOWED_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    # Added last so it wraps everything and every log line carries the request id.
    app.add_middleware(RequestContextMiddleware)

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    return app


app = create_application()
