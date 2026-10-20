import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Settings and the database engine are built at import time, so isolate them before importing app.
_STORAGE = Path(tempfile.mkdtemp(prefix="enercol-tests-"))
os.environ.update(
    {
        "DATABASE_PATH": str(_STORAGE / "test.db"),
        "VECTOR_DB_PATH": str(_STORAGE / "chroma"),
        "OPENROUTER_API_KEY": "",
        "TAVILY_API_KEY": "",
        "LOG_LEVEL": "WARNING",
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PASSWORD = "correct-horse-battery"


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_user(client: TestClient):
    created = 0

    def _make_user() -> TestClient:
        nonlocal created
        created += 1
        user_client = TestClient(app)
        response = user_client.post(
            "/api/v1/auth/register",
            json={
                "display_name": "Analista",
                "email": f"user-{os.urandom(4).hex()}-{created}@example.com",
                "password": PASSWORD,
            },
        )
        assert response.status_code == 201, response.text
        return user_client

    return _make_user
