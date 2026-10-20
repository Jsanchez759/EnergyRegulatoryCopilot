import logging
import re
import time
import uuid

import sentry_sdk
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging_config import request_id_var

logger = logging.getLogger("app.request")
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")


class RequestContextMiddleware:
    """Assign a request id, expose it in logs and responses, and log one line per request.

    Implemented as pure ASGI so streaming responses are timed until their last byte.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        sentry_sdk.set_tag("request_id", request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception(
                "request failed",
                extra=self._fields(scope, status_code, started),
            )
            raise
        else:
            level = logging.WARNING if status_code >= 500 else logging.INFO
            logger.log(level, "request completed", extra=self._fields(scope, status_code, started))
        finally:
            request_id_var.reset(token)

    @staticmethod
    def _fields(scope: Scope, status_code: int, started: float) -> dict[str, object]:
        client = scope.get("client")
        return {
            "method": scope["method"],
            "path": scope["path"],
            "status": status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "client_ip": client[0] if client else None,
        }
