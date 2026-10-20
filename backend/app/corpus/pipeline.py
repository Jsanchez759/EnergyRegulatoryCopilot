import hashlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update
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
from app.corpus.sources import creg
from app.corpus.storage import BlobStorage
from app.corpus.text import pdf_text, split_passages
from app.services.embeddings import EmbeddingClient
from app.services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)

PROCESS_ENTRYPOINT = "corpus.process"
# Below this, the HTML is a stub (e.g. only a header) and the PDF holds the real text.
MIN_HTML_TEXT_CHARS = 1500

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

    async def discover_creg(self, years: list[int], mode: str) -> CrawlRun:
        """Read CREG listings and queue every norm that is new or needs processing.

        mode="incremental" sends conditional requests and skips unchanged listings;
        mode="backfill" reads every listing and queues norms that were never processed.
        """
        async with self._sessions() as db:
            # Discovery runs one at a time per source, so any run still open was interrupted.
            await db.execute(
                update(CrawlRun)
                .where(CrawlRun.source == creg.SOURCE, CrawlRun.status == "running")
                .values(status="abandoned", finished_at=utc_now())
            )
            run = CrawlRun(source=creg.SOURCE, mode=mode)
            db.add(run)
            await db.commit()

        queued: set[str] = set()
        for year in years:
            for url in creg.listing_urls(year):
                try:
                    items = await self._read_listing(
                        url, conditional=mode == "incremental", run=run
                    )
                except FetchError as exc:
                    if "HTTP 404" in str(exc):
                        continue
                    run.errors += 1
                    run.error_sample = str(exc)[:1000]
                    logger.warning("listing fetch failed", extra={"url": url})
                    continue
                if items is not None:
                    run.items_seen += len(items)
                    for item in items:
                        norm_id = await self._upsert_from_listing(item)
                        if norm_id and norm_id not in queued:
                            queued.add(norm_id)
                            await self._enqueue(PROCESS_ENTRYPOINT, {"norm_id": norm_id}, norm_id)
                run.items_queued = len(queued)
                await self._save_run(run)

        run.items_queued = len(queued)
        run.status = "failed" if run.errors and not run.items_seen else "succeeded"
        run.finished_at = utc_now()
        await self._save_run(run)
        logger.info(
            "crawl finished",
            extra={
                "source": creg.SOURCE,
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

    async def queue_recheck(self, source: str) -> int:
        async with self._sessions() as db:
            ids = list(await db.scalars(select(Norm.id).where(Norm.source == source)))
        for norm_id in ids:
            await self._enqueue(PROCESS_ENTRYPOINT, {"norm_id": norm_id}, norm_id)
        return len(ids)

    async def _read_listing(
        self, url: str, conditional: bool, run: CrawlRun
    ) -> list[creg.ListingItem] | None:
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
        return creg.parse_listing(result.content, url)

    async def _upsert_from_listing(self, item: creg.ListingItem) -> str | None:
        doc_type = (
            "proyecto_resolucion"
            if item.draft_listing or item.number.startswith("7")
            else "resolucion"
        )
        key = creg.norm_key(doc_type, item.number, item.year)
        async with self._sessions() as db:
            norm = await db.scalar(select(Norm).where(Norm.key == key))
            if norm is None:
                norm = Norm(
                    key=key,
                    source=creg.SOURCE,
                    doc_type=doc_type,
                    number=item.number,
                    year=item.year,
                    title=item.title,
                    status="derogada" if item.derogated else "desconocido",
                    url=item.url,
                    extra={"listing_derogated": item.derogated},
                )
                db.add(norm)
                await db.commit()
                return norm.id
            listing_changed = norm.extra.get("listing_derogated") != item.derogated
            if listing_changed:
                norm.extra = {**norm.extra, "listing_derogated": item.derogated}
                await db.commit()
            return norm.id if norm.content_sha256 is None or listing_changed else None

    async def process_norm(self, norm_id: str) -> str:
        """Fetch, parse and index one norm. Returns what happened, for logs and tests."""
        async with self._sessions() as db:
            norm = await db.get(Norm, norm_id)
            if norm is None:
                return "missing"
            url, etag, last_modified, previous_hash = (
                norm.url,
                norm.etag,
                norm.last_modified,
                norm.content_sha256,
            )
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

        parsed = creg.parse_document(result.content, url, settings.CORPUS_CHUNK_CHARS)
        primary = creg.primary_pdf(parsed.files, parsed.number)
        pdf = await self._download(primary.url) if primary else None

        text, passages, text_source = parsed.text, parsed.passages, "html"
        if len(parsed.text) < MIN_HTML_TEXT_CHARS and pdf is not None:
            pdf_content, _ = pdf
            text, text_source = pdf_text(pdf_content)
            passages = [
                ("Texto completo" if index == 0 else f"Texto completo (parte {index + 1})", part)
                for index, part in enumerate(split_passages(text, settings.CORPUS_CHUNK_CHARS))
            ]

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if content_hash == previous_hash:
            await self._touch(norm_id, etag=result.etag, last_modified=result.last_modified)
            await self._reconcile_status(norm_id, derogated_in_text=parsed.derogated)
            await self._fill_missing_embeddings(norm_id)
            return "unchanged"

        vectors = await self._embed(
            [f"{parsed.label}. {label}\n{body}" for label, body in passages]
        )
        raw_key = await self._storage.put(result.content, ".htm")
        async with self._sessions() as db:
            norm = await db.get(Norm, norm_id)
            previous_status = norm.status
            status = _status(parsed.doc_type, parsed.derogated, norm.extra)
            if parsed.key != norm.key:
                taken = await db.scalar(select(Norm.id).where(Norm.key == parsed.key))
                if taken is None:
                    norm.key, norm.doc_type = parsed.key, parsed.doc_type
                else:
                    logger.warning("parsed key belongs to another norm", extra={"key": parsed.key})
            norm.title = parsed.title or norm.title
            norm.issued_on = parsed.issued_on
            norm.published_on = parsed.published_on
            norm.diario_oficial = parsed.diario_oficial
            norm.status = status
            norm.etag, norm.last_modified = result.etag, result.last_modified
            norm.content_sha256 = content_hash
            norm.last_checked_at = norm.last_changed_at = utc_now()

            db.add(
                NormVersion(
                    norm_id=norm_id,
                    content_sha256=content_hash,
                    text=text,
                    text_source=text_source,
                    raw_storage_key=raw_key,
                )
            )
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
            await db.execute(delete(NormRelation).where(NormRelation.reported_by_id == norm_id))
            db.add_all(
                NormRelation(
                    from_key=relation.from_key,
                    to_key=parsed.key,
                    relation=relation.relation,
                    from_article=relation.from_article,
                    to_article=relation.to_article,
                    evidence=relation.evidence,
                    reported_by_id=norm_id,
                )
                for relation in parsed.relations
            )
            await self._sync_files(db, norm_id, parsed.files, primary, pdf)
            kind = "norma_nueva" if previous_hash is None else "norma_actualizada"
            db.add(
                NormEvent(
                    norm_id=norm_id,
                    kind=kind,
                    details={"relations": len(parsed.relations), "passages": len(passages)},
                )
            )
            if previous_hash is not None and previous_status != status:
                db.add(
                    NormEvent(
                        norm_id=norm_id,
                        kind="estado_cambiado",
                        details={"from": previous_status, "to": status},
                    )
                )
            await db.commit()
        return "indexed"

    async def _sync_files(
        self,
        db: AsyncSession,
        norm_id: str,
        files: list[creg.ParsedFile],
        primary: creg.ParsedFile | None,
        pdf: tuple[bytes, str] | None,
    ) -> None:
        existing = {
            item.url: item
            for item in await db.scalars(select(NormFile).where(NormFile.norm_id == norm_id))
        }
        for parsed_file in files:
            row = existing.get(parsed_file.url) or NormFile(norm_id=norm_id, url=parsed_file.url)
            row.label = parsed_file.label[:200]
            row.kind = "original" if parsed_file is primary else parsed_file.kind
            if parsed_file is primary and pdf is not None:
                content, storage_key = pdf
                row.storage_key = storage_key
                row.sha256 = hashlib.sha256(content).hexdigest()
                row.size_bytes = len(content)
                row.content_type = "application/pdf"
                row.fetched_at = utc_now()
            db.add(row)

    async def _download(self, url: str) -> tuple[bytes, str] | None:
        try:
            result = await self._fetcher.get(url)
        except FetchError:
            logger.warning("original file download failed", extra={"url": url})
            return None
        return result.content, await self._storage.put(result.content, ".pdf")

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
            db.add(
                NormEvent(
                    norm_id=norm_id,
                    kind="estado_cambiado",
                    details={"from": norm.status, "to": status},
                )
            )
            norm.status = status
            await db.commit()

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
    if doc_type == "proyecto_resolucion":
        return "proyecto"
    return "derogada" if derogated_in_text or extra.get("listing_derogated") else "vigente"
