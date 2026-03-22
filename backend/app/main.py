import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("app")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Backend API for the Energy Regulatory Copilot project.",
    )

    # Allow frontend clients to call this backend during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_origin_regex=settings.ALLOWED_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Log HTTP request/response metadata with latency and request id."""
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"

        logger.info("request.start id=%s method=%s path=%s client=%s", request_id, method, path, client_host)
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
