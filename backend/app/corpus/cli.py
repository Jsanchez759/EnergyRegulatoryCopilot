"""Operational commands for the regulatory corpus.

python -m app.corpus.cli install-queue        # deploy step, after alembic upgrade head
python -m app.corpus.cli backfill [--from YEAR]
python -m app.corpus.cli discover             # incremental crawl now
python -m app.corpus.cli process URL          # fetch and index one CREG page, synchronously
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


async def enqueue_discover(years: list[int], mode: str) -> None:
    connection = await asyncpg.connect(asyncpg_dsn())
    try:
        await Queries(AsyncpgDriver(connection)).enqueue(
            DISCOVER_ENTRYPOINT,
            json.dumps({"years": years, "mode": mode}).encode(),
            dedupe_key=f"{DISCOVER_ENTRYPOINT}:{mode}",
        )
    except DuplicateJobError:
        print(f"A {mode} discovery is already queued or running.")
        return
    finally:
        await connection.close()
    print(f"Queued {mode} discovery for {years[0]}–{years[-1]}; the worker will process it.")


async def process_url(url: str) -> None:
    """Index one page without the queue: useful to debug parsing against the live site."""
    connection = await asyncpg.connect(asyncpg_dsn())
    try:
        async with corpus_pipeline(Queries(AsyncpgDriver(connection))) as pipeline:
            listing = creg.ListingItem(
                url=url,
                number=creg.DOC_HREF.search(url)["number"],
                year=int(creg.DOC_HREF.search(url)["year"]),
                label="",
                title="",
                derogated=False,
                is_draft=creg.normalize_number(creg.DOC_HREF.search(url)["number"]).startswith("7"),
            )
            await pipeline._upsert_from_listing(listing)
            async with SessionFactory() as db:
                norm_id = await db.scalar(select(Norm.id).where(Norm.url == url))
            print(await pipeline.process_norm(norm_id))
    finally:
        await connection.close()
        await engine.dispose()


async def enqueue_reparse(source: str) -> None:
    connection = await asyncpg.connect(asyncpg_dsn())
    try:
        async with corpus_pipeline(Queries(AsyncpgDriver(connection))) as pipeline:
            queued = await pipeline.queue_recheck(source, reparse=True)
    finally:
        await connection.close()
        await engine.dispose()
    print(f"Queued {queued} {source} norms for re-parsing from stored pages.")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="python -m app.corpus.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install-queue")
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--from", dest="start", type=int, default=settings.CORPUS_START_YEAR)
    commands.add_parser("discover")
    process = commands.add_parser("process")
    process.add_argument("url")
    reparse = commands.add_parser("reparse")
    reparse.add_argument("--source", default=creg.SOURCE)
    args = parser.parse_args()

    year = datetime.now(UTC).year
    if args.command == "install-queue":
        asyncio.run(install_queue())
    elif args.command == "backfill":
        asyncio.run(enqueue_discover(list(range(args.start, year + 1)), "backfill"))
    elif args.command == "discover":
        asyncio.run(enqueue_discover([year - 1, year], "incremental"))
    elif args.command == "process":
        asyncio.run(process_url(args.url))
    elif args.command == "reparse":
        asyncio.run(enqueue_reparse(args.source))


if __name__ == "__main__":
    main()
