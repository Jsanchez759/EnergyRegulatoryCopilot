import json
import logging
import logging.config
from contextvars import ContextVar
from datetime import UTC, datetime

from app.core.config import settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "request_id"}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": settings.ENVIRONMENT,
        }
        if getattr(record, "request_id", None):
            payload["request_id"] = record.request_id
        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _RESERVED_ATTRS}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {
            key: value for key, value in record.__dict__.items() if key not in _RESERVED_ATTRS
        }
        if extras:
            line += " " + " ".join(f"{key}={value}" for key, value in extras.items())
        if getattr(record, "request_id", None):
            line += f" request_id={record.request_id}"
        return line


def setup_logging() -> None:
    """Configure application-wide logging as JSON lines or readable console output."""
    formatter = (
        {"()": JsonFormatter}
        if settings.LOG_FORMAT == "json"
        else {
            "()": ConsoleFormatter,
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        }
    )
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {"default": formatter},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                },
            },
            "loggers": {
                # Uvicorn's own handlers would print a second, unstructured line per event.
                "uvicorn": {
                    "handlers": ["console"],
                    "level": settings.LOG_LEVEL,
                    "propagate": False,
                },
                "uvicorn.error": {"handlers": [], "level": settings.LOG_LEVEL, "propagate": True},
                # The request middleware already logs every response with its request id.
                "uvicorn.access": {"handlers": [], "level": "WARNING", "propagate": False},
                "httpx": {"level": "WARNING"},
            },
            "root": {"handlers": ["console"], "level": settings.LOG_LEVEL},
        }
    )
