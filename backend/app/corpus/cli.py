"""Operational commands for the regulatory corpus.

python -m app.corpus.cli install-queue        # deploy step, after alembic upgrade head
python -m app.corpus.cli backfill [--source creg] [--from YEAR]
python -m app.corpus.cli discover [--source creg]   # incremental crawl now
python -m app.corpus.cli process URL          # fetch and index one page, synchronously
python -m app.corpus.cli reparse [--source creg]    # re-read stored pages after parser fixes

Without --source, commands apply to every source.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime

import asyncpg
from pgqueuer import AsyncpgDriver
from pgqueuer.errors import DuplicateJobError
from pgqueuer.queries import Queries
from sqlalchemy import select

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.corpus.models import Norm
from app.corpus.sources import creg
from app.corpus.sources.registry import ADAPTERS
from app.corpus.worker import DISCOVER_ENTRYPOINT, asyncpg_dsn, corpus_pipeline
from app.db import SessionFactory, engine


async def install_queue() -> None:
    connection = await asyncpg.connect(asyncpg_dsn())
    try:
        queries = Queries(AsyncpgDriver(connection))
        if await queries.schema_is_installed():
            await queries.upgrade()
            print("PgQueuer schema upgraded.")
        else:
            await queries.install(create_schema=False)
            print("PgQueuer schema installed.")
    finally:
        await connection.close()


async def enqueue_discover(sources: list[str], years: list[int], mode: str) -> None:
    connection = await asyncpg.connect(asyncpg_dsn())
    try:
        queries = Queries(AsyncpgDriver(connection))
        for source in sources:
            try:
                await queries.enqueue(
                    DISCOVER_ENTRYPOINT,
                    json.dumps({"source": source, "years": years, "mode": mode}).encode(),
                    dedupe_key=f"{DISCOVER_ENTRYPOINT}:{source}:{mode}",
                )
            except DuplicateJobError:
                print(f"A {mode} discovery of {source} is already queued or running.")
                continue
            print(f"Queued {mode} discovery of {source} for {years[0]}–{years[-1]}.")
    finally:
        await connection.close()


async def process_url(url: str) -> None:
    """Index one page without the queue: useful to debug parsing against the live site.

    CREG pages can be new; other sources need the norm to be discovered first, because their
    listings carry facts the page lacks.
    """
    connection = await asyncpg.connect(asyncpg_dsn())
    try:
        async with corpus_pipeline(Queries(AsyncpgDriver(connection))) as pipeline:
            match = creg.DOC_HREF.search(url)
            if match:
                number, year = match["number"], int(match["year"])
                draft = year >= creg.SECTOR_PREFIX_YEAR and creg.normalize_number(
                    number
                ).startswith("7")
                await pipeline._upsert_from_listing(
                    creg.listing_item(url, number, year, draft=draft)
                )
            async with SessionFactory() as db:
                norm_id = await db.scalar(select(Norm.id).where(Norm.url == url))
            if norm_id is None:
                print("Unknown URL: run a discovery of its source first.")
                return
            print(await pipeline.process_norm(norm_id))
    finally:
        await connection.close()
        await engine.dispose()


async def enqueue_reparse(sources: list[str]) -> None:
    connection = await asyncpg.connect(asyncpg_dsn())
    try:
        async with corpus_pipeline(Queries(AsyncpgDriver(connection))) as pipeline:
            for source in sources:
                queued = await pipeline.queue_recheck(source, reparse=True)
                print(f"Queued {queued} {source} norms for re-parsing from stored pages.")
    finally:
        await connection.close()
        await engine.dispose()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="python -m app.corpus.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install-queue")
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--from", dest="start", type=int, default=settings.CORPUS_START_YEAR)
    discover = commands.add_parser("discover")
    process = commands.add_parser("process")
    process.add_argument("url")
    reparse = commands.add_parser("reparse")
    for command in (backfill, discover, reparse):
        command.add_argument("--source", choices=sorted(ADAPTERS))
    args = parser.parse_args()

    year = datetime.now(UTC).year
    sources = [args.source] if getattr(args, "source", None) else sorted(ADAPTERS)
    if args.command == "install-queue":
        asyncio.run(install_queue())
    elif args.command == "backfill":
        asyncio.run(enqueue_discover(sources, list(range(args.start, year + 1)), "backfill"))
    elif args.command == "discover":
        asyncio.run(enqueue_discover(sources, [year - 1, year], "incremental"))
    elif args.command == "process":
        asyncio.run(process_url(args.url))
    elif args.command == "reparse":
        asyncio.run(enqueue_reparse(sources))


if __name__ == "__main__":
    main()
