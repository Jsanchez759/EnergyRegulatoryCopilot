from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

# asyncpg connections are bound to the event loop that opened them, and test clients run
# requests on short-lived loops, so tests must not reuse pooled connections.
engine = (
    create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    if settings.ENVIRONMENT == "test"
    else create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
    )
)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session
