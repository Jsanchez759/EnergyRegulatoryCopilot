import logging
import logging.config

from app.core.config import settings


def setup_logging() -> None:
    """Configure application-wide structured logging."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
            },
            "loggers": {
                "app": {
                    "handlers": ["console"],
                    "level": settings.LOG_LEVEL,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": settings.LOG_LEVEL,
                },
                "uvicorn.access": {
                    "level": settings.LOG_LEVEL,
                },
                # Chroma telemetry occasionally logs noisy internal errors even when
                # anonymized telemetry is disabled. Suppress those in production logs.
                "chromadb.telemetry.product.posthog": {
                    "level": "CRITICAL",
                    "propagate": False,
                },
            },
            "root": {
                "handlers": ["console"],
                "level": settings.LOG_LEVEL,
            },
        }
    )
