import asyncio
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Settings and the database engine are built at import time, so isolate them before importing app.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://enercol:enercol@localhost:5432/enercol_test"
)
os.environ.update(
    {
        "ENVIRONMENT": "test",
        "CORPUS_STORAGE_DIR": tempfile.mkdtemp(prefix="enercol-corpus-"),
        "DATABASE_URL": TEST_DATABASE_URL,
        "OPENROUTER_API_KEY": "",
        "TAVILY_API_KEY": "",
        "SENTRY_DSN": "",
        "LOG_LEVEL": "WARNING",
        "RATE_LIMIT_AUTH": "10000/minute",
        "RATE_LIMIT_AGENT": "10000/minute",
        "RATE_LIMIT_UPLOAD": "10000/minute",
        "RATE_LIMIT_PUBLIC": "10000/minute",
        "REGISTRATION_MODE": "open",
    }
)

from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from alembic import command  # noqa: E402
from app.main import app  # noqa: E402

PASSWORD = "correct-horse-battery"
BACKEND_ROOT = Path(__file__).resolve().parents[1]


async def _reset_schema() -> None:
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def migrate_test_database() -> None:
    asyncio.run(_reset_schema())
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.attributes["configure_logger"] = False
    config.attributes["database_url"] = TEST_DATABASE_URL
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    migrate_test_database()
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


async def _promote(email: str) -> None:
    from sqlalchemy import update

    from app.db import SessionFactory
    from app.models import User

    async with SessionFactory() as db:
        await db.execute(update(User).where(User.email == email).values(role="admin"))
        await db.commit()


@pytest.fixture
def make_admin(make_user):
    def _make_admin() -> TestClient:
        admin = make_user()
        asyncio.run(_promote(admin.get("/api/v1/auth/me").json()["email"]))
        return admin

    return _make_admin
