import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.core.logging_config import JsonFormatter, RequestIdFilter, request_id_var
from app.core.rate_limit import get_limiter
from tests.conftest import PASSWORD


@pytest.fixture
def fresh_rate_limits():
    asyncio.run(get_limiter().storage.reset())
    yield
    asyncio.run(get_limiter().storage.reset())


def test_readiness_reports_migrated_schema(client: TestClient) -> None:
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["schema"] == "0003_pplx_embeddings_1024"


def test_responses_carry_generated_or_forwarded_request_id(client: TestClient) -> None:
    generated = client.get("/api/v1/health").headers["x-request-id"]
    assert len(generated) == 32
    forwarded = client.get("/api/v1/health", headers={"X-Request-ID": "edge-request-1234"})
    assert forwarded.headers["x-request-id"] == "edge-request-1234"
    rejected = client.get("/api/v1/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert rejected.headers["x-request-id"] != "bad id\nwith newline"


def test_json_logs_include_request_id_and_extra_fields() -> None:
    record = logging.makeLogRecord(
        {"name": "app", "levelname": "INFO", "msg": "request completed", "status": 200}
    )
    token = request_id_var.set("abc123")
    try:
        RequestIdFilter().filter(record)
    finally:
        request_id_var.reset(token)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] == "abc123"
    assert payload["status"] == 200
    assert payload["message"] == "request completed"


def test_login_is_rate_limited_per_ip(client, make_user, monkeypatch, fresh_rate_limits) -> None:
    email = make_user().get("/api/v1/auth/me").json()["email"]
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH", "3/minute")
    anonymous = TestClient(client.app)
    statuses = [
        anonymous.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        ).status_code
        for _ in range(4)
    ]
    assert statuses == [200, 200, 200, 429]
    limited = anonymous.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert int(limited.headers["retry-after"]) > 0


def test_agent_is_rate_limited_per_user(make_user, monkeypatch, fresh_rate_limits) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_AGENT", "1/minute")
    user, other = make_user(), make_user()
    payload = {"conversation_id": "missing", "question": "¿Qué es la CREG?"}
    assert user.post("/api/v1/agent/stream", json=payload).status_code == 404
    assert user.post("/api/v1/agent/stream", json=payload).status_code == 429
    assert other.post("/api/v1/agent/stream", json=payload).status_code == 404


def test_production_rejects_unsafe_settings() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(
            ENVIRONMENT="production",
            SESSION_COOKIE_SECURE=False,
            OPENROUTER_API_KEY="",
            ALLOWED_ORIGINS=["http://localhost:5173"],
        )
    message = str(error.value)
    assert "SESSION_COOKIE_SECURE" in message
    assert "OPENROUTER_API_KEY" in message
    assert "localhost" in message


def test_production_defaults_are_strict() -> None:
    production = Settings(
        ENVIRONMENT="production",
        SESSION_COOKIE_SECURE=True,
        OPENROUTER_API_KEY="key",
        ALLOWED_ORIGINS=["https://app.enercol.co"],
    )
    assert production.ALLOWED_ORIGIN_REGEX is None
    assert production.LOG_FORMAT == "json"


def test_samesite_none_requires_secure_cookie() -> None:
    with pytest.raises(ValidationError):
        Settings(SESSION_COOKIE_SAMESITE="none", SESSION_COOKIE_SECURE=False)
