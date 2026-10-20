import hashlib
import logging
import mimetypes
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from asyncer import asyncify
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.corpus.fetch import FetchError, PoliteFetcher
from app.corpus.models import (
    CrawlRun,
    FetchCache,
    Norm,
    NormArticle,
    NormEvent,
    NormFile,
    NormRelation,
    NormVersion,
)
from app.corpus.sources.base import ListingItem, NormRef, ParsedFile, is_draft
from app.corpus.sources.registry import adapter_for
from app.corpus.storage import BlobStorage
from app.corpus.text import document_passages, document_text
from app.models import new_id
from app.services.embeddings import EmbeddingClient
from app.services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)

PROCESS_ENTRYPOINT = "corpus.process"
# Norms first seen within this window of their publication are news; older ones are backfill.
NEWS_WINDOW = timedelta(days=60)
# Below this, the HTML is a stub (e.g. only a header) and the original file holds the text.
MIN_HTML_TEXT_CHARS = 1500

# (size, storage key, sha256) of a stored original.
StoredFile = tuple[int, str, str]

Enqueue = Callable[[str, dict[str, Any], str], Awaitable[None]]


def utc_now() -> datetime:
    return datetime.now(UTC)


class CorpusPipeline:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        fetcher: PoliteFetcher,
        storage: BlobStorage,
        embeddings: EmbeddingClient,
        enqueue: Enqueue,
    ) -> None:
        self._sessions = sessions
        self._fetcher = fetcher
        self._storage = storage
        self._embeddings = embeddings
        self._enqueue = enqueue

    async def discover(self, source: str, years: list[int], mode: str) -> CrawlRun:
        """Read a source's listings and queue every norm that is new or needs processing.

        mode="incremental" reads recent listings, conditionally where the source allows it;
        mode="backfill" reads every listing for `years` and queues norms never processed.
        """
        adapter = adapter_for(source)
        async with self._sessions() as db:
            # Discovery runs one at a time per source, so any run still open was interrupted.
            await db.execute(
                update(CrawlRun)
                .where(CrawlRun.source == source, CrawlRun.status == "running")
                .values(status="abandoned", finished_at=utc_now())
            )
            run = CrawlRun(source=source, mode=mode)
            db.add(run)
            await db.commit()

        queued: set[str] = set()

        async def read(url: str, conditional: bool) -> bytes | None:
            try:
                content = await self._read_listing(url, conditional=conditional, run=run)
            except FetchError as exc:
                content = None
                if "HTTP 404" not in str(exc):
                    run.errors += 1
                    run.error_sample = str(exc)[:1000]
                    logger.warning("listing fetch failed", extra={"url": url, "source": source})
            run.items_queued = len(queued)
            await self._save_run(run)
            return content

        try:
            async for item in adapter.discover(read, years, mode):
                run.items_seen += 1
                norm_id = await self._upsert_from_listing(item)
                if norm_id and norm_id not in queued:
                    queued.add(norm_id)
                    await self._enqueue(PROCESS_ENTRYPOINT, {"norm_id": norm_id}, norm_id)
        except Exception as exc:
            run.status, run.error_sample = "failed", repr(exc)[:1000]
            run.finished_at = utc_now()
            await self._save_run(run)
            raise

        run.items_queued = len(queued)
        run.status = "failed" if run.errors and not run.items_seen else "succeeded"
        run.finished_at = utc_now()
        await self._save_run(run)
        logger.info(
            "crawl finished",
            extra={
                "source": source,
                "mode": mode,
                "pages_fetched": run.pages_fetched,
                "pages_unchanged": run.pages_unchanged,
                "items_seen": run.items_seen,
                "items_queued": run.items_queued,
                "errors": run.errors,
            },
        )
        return run

    async def _save_run(self, run: CrawlRun) -> None:
        async with self._sessions() as db:
            await db.merge(run)
            await db.commit()

    async def queue_recheck(self, source: str, reparse: bool = False) -> int:
        conditions = [Norm.source == source]
        if reparse:
            conditions.append(Norm.content_sha256.is_not(None))
        async with self._sessions() as db:
            ids = list(await db.scalars(select(Norm.id).where(*conditions)))
        for norm_id in ids:
            payload = {"norm_id": norm_id, "reparse": reparse}
            await self._enqueue(
                PROCESS_ENTRYPOINT, payload, f"reparse:{norm_id}" if reparse else norm_id
            )
        return len(ids)

    async def _read_listing(self, url: str, conditional: bool, run: CrawlRun) -> bytes | None:
        async with self._sessions() as db:
            cache = await db.get(FetchCache, url)
        result = await self._fetcher.get(
            url,
            etag=cache.etag if cache and conditional else None,
            last_modified=cache.last_modified if cache and conditional else None,
        )
        if result.not_modified:
            run.pages_unchanged += 1
            return None
        run.pages_fetched += 1
        if not (result.etag or result.last_modified):
            return result.content
        digest = hashlib.sha256(result.content).hexdigest()
        unchanged = cache is not None and cache.content_sha256 == digest
        async with self._sessions() as db:
            await db.merge(
                FetchCache(
                    url=url,
                    etag=result.etag,
                    last_modified=result.last_modified,
                    content_sha256=digest,
                    fetched_at=utc_now(),
                )
            )
            await db.commit()
        if unchanged and conditional:
            run.pages_unchanged += 1
            return None
        return result.content

    async def _upsert_from_listing(self, item: ListingItem) -> str | None:
        listing_facts = {"listing_derogated": item.derogated, **item.extra}
        async with self._sessions() as db:
            # The URL identifies the document even when its classification changes.
            norm = await db.scalar(select(Norm).where(Norm.url == item.url)) or await db.scalar(
                select(Norm).where(Norm.key == item.key)
            )
            if norm is None:
                norm = Norm(
                    key=item.key,
                    source=item.source,
                    doc_type=item.doc_type,
                    number=item.number,
                    year=item.year,
                    title=item.title,
                    issued_on=item.issued_on,
                    status="derogada" if item.derogated else "desconocido",
                    url=item.url,
                    extra=listing_facts,
                )
                db.add(norm)
                await db.commit()
                return norm.id
            changed = any(norm.extra.get(name) != value for name, value in listing_facts.items())
            if changed:
                norm.extra = {**norm.extra, **listing_facts}
            norm.title = norm.title or item.title
            norm.issued_on = norm.issued_on or item.issued_on
            if norm.doc_type != item.doc_type and item.is_draft:
                taken = await db.scalar(
                    select(Norm.id).where(Norm.key == item.key, Norm.id != norm.id)
                )
                if taken is None:
                    norm.key, norm.doc_type, changed = item.key, item.doc_type, True
            await db.commit()
            return norm.id if norm.content_sha256 is None or changed else None

    async def process_norm(self, norm_id: str, reparse: bool = False) -> str:
        """Fetch, parse and index one norm. Returns what happened, for logs and tests.

        reparse=True re-reads the stored page instead of the source, so parser fixes reach
        already indexed norms without new requests or news events.
        """
        async with self._sessions() as db:
            norm = await db.get(Norm, norm_id)
            if norm is None:
                return "missing"
            ref = NormRef(
                source=norm.source,
                url=norm.url,
                key=norm.key,
                doc_type=norm.doc_type,
                number=norm.number,
                year=norm.year,
                title=norm.title,
                issued_on=norm.issued_on,
                extra=dict(norm.extra or {}),
            )
            url, etag, last_modified, previous_hash = (
                norm.url,
                norm.etag,
                norm.last_modified,
                norm.content_sha256,
            )
            stored_page = await db.scalar(
                select(NormVersion.raw_storage_key)
                .where(NormVersion.norm_id == norm_id)
                .order_by(NormVersion.fetched_at.desc())
                .limit(1)
            )

        if reparse and stored_page:
            content = await self._storage.get(stored_page)
        else:
            reparse = False
            result = await self._fetcher.get(
                url,
                etag=etag if previous_hash else None,
                last_modified=last_modified if previous_hash else None,
            )
            if result.not_modified:
                await self._touch(norm_id)
                await self._reconcile_status(norm_id, derogated_in_text=None)
                await self._fill_missing_embeddings(norm_id)
                return "not_modified"
            content, etag, last_modified = result.content, result.etag, result.last_modified

        parsed = adapter_for(ref.source).parse(content, ref, settings.CORPUS_CHUNK_CHARS)
        primary = parsed.primary
        original = await self._original_file(norm_id, primary) if primary else None

        text, passages, text_source = parsed.text, parsed.passages, "html"
        if original is not None and (
            parsed.text_in_primary or len(parsed.text) < MIN_HTML_TEXT_CHARS
        ):
            stored = await self._storage.get(original[1])
            # OCR takes seconds per page; off the event loop, the worker keeps its heartbeat.
            extracted, method = await asyncify(document_text)(stored, primary.kind)
            if extracted:
                found = document_passages(extracted, settings.CORPUS_CHUNK_CHARS)
                if parsed.text_in_primary:
                    text = "\n\n".join(part for part in (parsed.text, extracted) if part)
                    passages = [*parsed.passages, *found]
                else:
                    text, passages = extracted, found
                text_source = method
        if not passages and parsed.title:
            passages = [("Título", parsed.title)]

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        text_changed = content_hash != previous_hash
        if not text_changed and not reparse:
            await self._touch(norm_id, etag=etag, last_modified=last_modified)
            await self._reconcile_status(norm_id, derogated_in_text=parsed.derogated)
            await self._fill_missing_embeddings(norm_id)
            return "unchanged"

        # Parser fixes can re-split the same text, so a reparse compares the passages too.
        passages_changed = text_changed or await self._passages_differ(norm_id, passages)
        vectors = (
            await self._embed([f"{parsed.label}. {label}\n{body}" for label, body in passages])
            if passages_changed
            else None
        )
        raw_key = await self._storage.put(content, ".htm")
        async with self._sessions() as db:
            norm = await db.get(Norm, norm_id)
            previous_status = norm.status
            status = _status(parsed.doc_type, parsed.derogated, norm.extra)
            if parsed.key != norm.key:
                taken = await db.scalar(select(Norm.id).where(Norm.key == parsed.key))
                if taken is None:
                    norm.key, norm.doc_type = parsed.key, parsed.doc_type
                    norm.number, norm.year = parsed.number, parsed.year
                else:
                    logger.warning("parsed key belongs to another norm", extra={"key": parsed.key})
            norm.title = parsed.title or norm.title
            norm.issued_on = parsed.issued_on or norm.issued_on
            norm.published_on = parsed.published_on or norm.published_on
            norm.diario_oficial = parsed.diario_oficial
            norm.status = status
            norm.etag, norm.last_modified = etag, last_modified
            norm.content_sha256 = content_hash
            norm.last_checked_at = utc_now()

            if text_changed:
                norm.last_changed_at = utc_now()
                db.add(
                    NormVersion(
                        norm_id=norm_id,
                        content_sha256=content_hash,
                        text=text,
                        text_source=text_source,
                        raw_storage_key=raw_key,
                    )
                )
            if passages_changed:
                await db.execute(delete(NormArticle).where(NormArticle.norm_id == norm_id))
                db.add_all(
                    NormArticle(
                        norm_id=norm_id,
                        position=position,
                        label=label[:200],
                        norm_label=parsed.label,
                        content=body,
                        embedding=vectors[position] if vectors else None,
                    )
                    for position, (label, body) in enumerate(passages)
                )
            else:
                await db.execute(
                    update(NormArticle)
                    .where(NormArticle.norm_id == norm_id)
                    .values(norm_label=parsed.label)
                )
            await db.execute(delete(NormRelation).where(NormRelation.reported_by_id == norm_id))
            if parsed.relations:
                # Two pages can report the same amendment; the first report is kept.
                await db.execute(
                    pg_insert(NormRelation)
                    .values(
                        [
                            {
                                "id": new_id(),
                                "from_key": relation.from_key,
                                "to_key": relation.to_key or norm.key,
                                "relation": relation.relation,
                                "from_article": relation.from_article,
                                "to_article": relation.to_article,
                                "evidence": relation.evidence,
                                "reported_by_id": norm_id,
                                "created_at": utc_now(),
                            }
                            for relation in parsed.relations
                        ]
                    )
                    .on_conflict_do_nothing(constraint="uq_norm_relation")
                )
            await self._sync_files(db, norm_id, parsed.files, primary, original)
            # Corrections from re-parsing stored pages are not regulatory news.
            if not reparse:
                kind = (
                    "norma_actualizada"
                    if previous_hash is not None
                    else "norma_nueva"
                    if _recently_published(parsed.published_on or parsed.issued_on)
                    else "norma_incorporada"
                )
                db.add(
                    NormEvent(
                        norm_id=norm_id,
                        kind=kind,
                        details={"relations": len(parsed.relations), "passages": len(passages)},
                    )
                )
                if previous_hash is not None and _is_news(previous_status, status):
                    db.add(
                        NormEvent(
                            norm_id=norm_id,
                            kind="estado_cambiado",
                            details={"from": previous_status, "to": status},
                        )
                    )
            await db.commit()
        return "indexed" if text_changed else "reparsed"

    async def _sync_files(
        self,
        db: AsyncSession,
        norm_id: str,
        files: list[ParsedFile],
        primary: ParsedFile | None,
        original: StoredFile | None,
    ) -> None:
        existing = {
            item.url: item
            for item in await db.scalars(select(NormFile).where(NormFile.norm_id == norm_id))
        }
        for parsed_file in files:
            row = existing.get(parsed_file.url) or NormFile(norm_id=norm_id, url=parsed_file.url)
            row.label = parsed_file.label[:200]
            row.kind = "original" if parsed_file is primary else parsed_file.kind
            if parsed_file is primary and original is not None and row.storage_key != original[1]:
                row.size_bytes, row.storage_key, row.sha256 = original
                row.content_type = mimetypes.guess_type(f"original.{parsed_file.kind}")[0]
                row.fetched_at = utc_now()
            db.add(row)

    async def _original_file(self, norm_id: str, file: ParsedFile) -> StoredFile | None:
        """The act's own file, downloaded only the first time."""
        async with self._sessions() as db:
            stored = await db.scalar(
                select(NormFile).where(
                    NormFile.norm_id == norm_id,
                    NormFile.url == file.url,
                    NormFile.storage_key.is_not(None),
                )
            )
        if stored is not None:
            return stored.size_bytes, stored.storage_key, stored.sha256
        downloaded = await self._download(file.url, f".{file.kind}")
        if downloaded is None:
            return None
        content, storage_key = downloaded
        return len(content), storage_key, hashlib.sha256(content).hexdigest()

    async def _download(self, url: str, suffix: str) -> tuple[bytes, str] | None:
        try:
            result = await self._fetcher.get(url)
        except FetchError:
            logger.warning("original file download failed", extra={"url": url})
            return None
        return result.content, await self._storage.put(result.content, suffix)

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self._embeddings.available or not texts:
            return None
        try:
            return await self._embeddings.embed(texts)
        except ProviderUnavailableError:
            # Lexical search still works; a later recheck fills the vectors.
            logger.warning("embeddings unavailable; indexing text only")
            return None

    async def _reconcile_status(self, norm_id: str, derogated_in_text: bool | None) -> None:
        """Apply status changes that come from the listing when the page text did not change."""
        async with self._sessions() as db:
            norm = await db.get(Norm, norm_id)
            derogated = (
                norm.status == "derogada" if derogated_in_text is None else derogated_in_text
            )
            status = _status(norm.doc_type, derogated, norm.extra)
            if status == norm.status:
                return
            if _is_news(norm.status, status):
                db.add(
                    NormEvent(
                        norm_id=norm_id,
                        kind="estado_cambiado",
                        details={"from": norm.status, "to": status},
                    )
                )
            norm.status = status
            await db.commit()

    async def _passages_differ(self, norm_id: str, passages: list[tuple[str, str]]) -> bool:
        async with self._sessions() as db:
            stored = await db.execute(
                select(NormArticle.label, NormArticle.content)
                .where(NormArticle.norm_id == norm_id)
                .order_by(NormArticle.position)
            )
            return [tuple(row) for row in stored.all()] != [
                (label[:200], body) for label, body in passages
            ]

    async def _fill_missing_embeddings(self, norm_id: str) -> None:
        async with self._sessions() as db:
            missing = list(
                await db.scalars(
                    select(NormArticle).where(
                        NormArticle.norm_id == norm_id, NormArticle.embedding.is_(None)
                    )
                )
            )
            if not missing:
                return
            vectors = await self._embed(
                [f"{item.norm_label}. {item.label}\n{item.content}" for item in missing]
            )
            if vectors is None:
                return
            for item, vector in zip(missing, vectors, strict=True):
                item.embedding = vector
            await db.commit()

    async def _touch(
        self, norm_id: str, etag: str | None = None, last_modified: str | None = None
    ) -> None:
        async with self._sessions() as db:
            norm = await db.get(Norm, norm_id)
            norm.last_checked_at = utc_now()
            if etag or last_modified:
                norm.etag, norm.last_modified = etag, last_modified
            await db.commit()


def _status(doc_type: str, derogated_in_text: bool, extra: dict[str, Any]) -> str:
    if is_draft(doc_type):
        return "proyecto"
    return "derogada" if derogated_in_text or extra.get("listing_derogated") else "vigente"


def _recently_published(published: date | None) -> bool:
    return published is not None and published >= utc_now().date() - NEWS_WINDOW


def _is_news(previous_status: str, status: str) -> bool:
    """Derogations are news; reclassifying a draft or resolving an unknown status is a fix."""
    return previous_status != status and {previous_status, status} <= {"vigente", "derogada"}
