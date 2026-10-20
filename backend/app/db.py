import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from app.core.config import settings

settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = settings.DATABASE_URL or f"sqlite+aiosqlite:///{settings.DATABASE_PATH.resolve()}"
engine = create_async_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def configure_sqlite(dbapi_connection, connection_record) -> None:
    if "sqlite" not in DATABASE_URL:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


def _upgrade_schema() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.attributes["configure_logger"] = False
    config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("+aiosqlite", "+pysqlite"))
    command.upgrade(config, "head")


async def initialize_database() -> None:
    """Apply checked-in schema migrations before serving requests."""
    await asyncio.to_thread(_upgrade_schema)
