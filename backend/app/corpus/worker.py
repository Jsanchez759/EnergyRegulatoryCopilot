"""Background worker: PgQueuer jobs and cron schedules that keep the regulatory corpus fresh.

Run with `python -m app.corpus.worker`.
"""

import asyncio
import json
import logging
import signal
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import asyncpg
from pgqueuer import AsyncpgDriver, PgQueuer
from pgqueuer.errors import DuplicateJobError
from pgqueuer.models import Job, Schedule
from pgqueuer.queries import Queries

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.observability import init_sentry
from app.corpus.fetch import PoliteFetcher
from app.corpus.pipeline import PROCESS_ENTRYPOINT, CorpusPipeline
from app.corpus.sources import creg
from app.corpus.storage import LocalBlobStorage
from app.db import SessionFactory, engine
from app.services.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

DISCOVER_ENTRYPOINT = "corpus.discover"


def asyncpg_dsn() -> str:
    return settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)


def make_enqueue(queries: Queries):
    async def enqueue(entrypoint: str, payload: dict[str, Any], dedupe_key: str) -> None:
        try:
            await queries.enqueue(
                entrypoint, json.dumps(payload).encode(), dedupe_key=f"{entrypoint}:{dedupe_key}"
            )
        except DuplicateJobError:
            pass  # the same work is already queued

    return enqueue


@asynccontextmanager
async def corpus_pipeline(queries: Queries):
    fetcher = PoliteFetcher(min_interval_by_host={creg.HOST: creg.MIN_INTERVAL_SECONDS})
    embeddings = EmbeddingClient()
    try:
        yield CorpusPipeline(
            sessions=SessionFactory,
            fetcher=fetcher,
            storage=LocalBlobStorage(settings.CORPUS_STORAGE_DIR),
            embeddings=embeddings,
            enqueue=make_enqueue(queries),
        )
    finally:
        await fetcher.close()
        await embeddings.close()


def register(pgq: PgQueuer, pipeline: CorpusPipeline) -> None:
    @pgq.entrypoint(DISCOVER_ENTRYPOINT, concurrency_limit=1)
    async def discover(job: Job) -> None:
        payload = json.loads(job.payload or b"{}")
        await pipeline.discover_creg(years=payload["years"], mode=payload["mode"])

    # The shared fetcher still paces each host; concurrency only overlaps embedding waits.
    @pgq.entrypoint(PROCESS_ENTRYPOINT, concurrency_limit=3)
    async def process(job: Job) -> None:
        payload = json.loads(job.payload or b"{}")
        outcome = await pipeline.process_norm(payload["norm_id"])
        logger.info("norm processed", extra={"norm_id": payload["norm_id"], "outcome": outcome})

    @pgq.schedule("corpus.creg.incremental", "0 */3 * * *")
    async def creg_incremental(_schedule: Schedule) -> None:
        year = datetime.now(UTC).year
        await pipeline.discover_creg(years=[year - 1, year], mode="incremental")

    # Amendments rewrite existing pages; conditional requests make this weekly pass cheap.
    @pgq.schedule("corpus.creg.recheck", "30 4 * * 0")
    async def creg_recheck(_schedule: Schedule) -> None:
        queued = await pipeline.queue_recheck(creg.SOURCE)
        logger.info("recheck queued", extra={"source": creg.SOURCE, "norms": queued})


async def main() -> None:
    setup_logging()
    init_sentry()
    connection = await asyncpg.connect(asyncpg_dsn())
    driver = AsyncpgDriver(connection)
    pgq = PgQueuer(driver)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, pgq.shutdown.set)
    async with corpus_pipeline(Queries(driver)) as pipeline:
        register(pgq, pipeline)
        logger.info("corpus worker started")
        try:
            await pgq.run()
        finally:
            await connection.close()
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
