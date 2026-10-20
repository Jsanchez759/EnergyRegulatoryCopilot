import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db import engine, initialize_database
from app.services.agent_service import AgentService
from app.services.document_service import DocumentService

setup_logging()
logger = logging.getLogger("app")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        await initialize_database()
        application.state.document_service = DocumentService()
        application.state.agent_service = AgentService(application.state.document_service)
        try:
            yield
        finally:
            await application.state.agent_service.close()
            await application.state.document_service.close()
            await engine.dispose()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Backend API for the Energy Regulatory Copilot project.",
        lifespan=lifespan,
    )

    # Allow frontend clients to call this backend during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_origin_regex=settings.ALLOWED_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def validate_request_origin(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if (
                origin
                and origin not in settings.ALLOWED_ORIGINS
                and not re.fullmatch(settings.ALLOWED_ORIGIN_REGEX, origin)
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "El origen de la solicitud no está autorizado."},
                )
        return await call_next(request)

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Log HTTP request/response metadata with latency and request id."""
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"

        logger.info(
            "request.start id=%s method=%s path=%s client=%s", request_id, method, path, client_host
        )
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request.error id=%s method=%s path=%s duration_ms=%.2f",
                request_id,
                method,
                path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request.end id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # Mount versioned API routes under a common prefix.
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()
