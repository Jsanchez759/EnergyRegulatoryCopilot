import asyncio
import hashlib
import io
import logging
import time

from asyncer import asyncify
from openai import AsyncOpenAI, OpenAIError
from pypdf import PdfReader
from sqlalchemy import delete, exists, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.models import Document, DocumentChunk
from app.schemas.agent import EvidenceSource, IndexedDocument
from app.services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)


class DocumentProcessingError(ValueError):
    """Raised when an uploaded PDF cannot be safely indexed."""


class DuplicateDocumentError(ValueError):
    """Raised when the user already indexed a file with the same content."""


def indexed_document(document: Document) -> IndexedDocument:
    return IndexedDocument(
        document_id=document.id,
        filename=document.filename,
        chunks_indexed=document.chunk_count,
        page_count=document.page_count,
        file_sha256=document.file_sha256,
        created_at=document.created_at.isoformat(),
    )


class DocumentService:
    """Manage PDF ingestion and similarity search over pgvector."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory
        self._llm_client = (
            AsyncOpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
                # Free-tier providers stall a share of requests; a short timeout plus retries
                # recovers.
                timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
                max_retries=settings.EMBEDDING_MAX_RETRIES,
            )
            if settings.OPENROUTER_API_KEY
            else None
        )

    def _require_client(self) -> AsyncOpenAI:
        if self._llm_client is None:
            raise ProviderUnavailableError("OPENROUTER_API_KEY is not configured on the server.")
        return self._llm_client

    async def ingest_pdf(self, file_bytes: bytes, filename: str, user_id: str) -> IndexedDocument:
        self._require_client()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        async with self._sessions() as db:
            duplicate = await db.scalar(
                select(Document.filename).where(
                    Document.user_id == user_id, Document.file_sha256 == file_hash
                )
            )
        if duplicate:
            raise DuplicateDocumentError(f"Este PDF ya está en tu biblioteca como «{duplicate}».")

        pages, page_count = await asyncify(self._extract_pages)(file_bytes)
        chunks = self._chunk_pages(pages)
        if not chunks:
            raise DocumentProcessingError("The PDF contains no selectable text to index.")
        embeddings = await self._embed_texts([str(chunk["text"]) for chunk in chunks])

        document = Document(
            user_id=user_id,
            filename=filename[:255],
            file_sha256=file_hash,
            size_bytes=len(file_bytes),
            page_count=page_count,
            chunk_count=len(chunks),
            embedding_model=settings.OPENROUTER_EMBEDDING_MODEL,
        )
        document.chunks = [
            DocumentChunk(
                user_id=user_id,
                chunk_index=index,
                page=int(chunk["page"]),
                content=str(chunk["text"]),
                embedding=embedding,
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        async with self._sessions() as db:
            db.add(document)
            await db.commit()
        return indexed_document(document)

    async def has_documents(self, user_id: str) -> bool:
        async with self._sessions() as db:
            return bool(await db.scalar(select(exists().where(Document.user_id == user_id))))

    async def search(self, query: str, user_id: str, top_k: int = 6) -> list[EvidenceSource]:
        if not await self.has_documents(user_id):
            return []
        query_embedding = (await self._embed_texts([query]))[0]
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        async with self._sessions() as db:
            # Without iterative scans, HNSW filters by owner after picking candidates and can
            # return too few rows for users with small libraries.
            await db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
            rows = await db.execute(
                select(DocumentChunk, Document, distance.label("distance"))
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.user_id == user_id)
                .order_by(distance)
                .limit(top_k)
            )
            results = rows.all()
        return [
            EvidenceSource(
                id=f"D{index}",
                kind="document",
                title=document.filename,
                page=chunk.page,
                snippet=chunk.content,
                score=round(1 - float(chunk_distance), 4),
                file_sha256=document.file_sha256,
                indexed_at=document.created_at.isoformat(),
            )
            for index, (chunk, document, chunk_distance) in enumerate(results, start=1)
        ]

    async def list_documents(self, user_id: str) -> list[IndexedDocument]:
        async with self._sessions() as db:
            documents = await db.scalars(
                select(Document).where(Document.user_id == user_id).order_by(Document.filename)
            )
            return [indexed_document(document) for document in documents]

    async def delete_document(self, user_id: str, document_id: str) -> bool:
        async with self._sessions() as db:
            result = await db.execute(
                delete(Document).where(Document.id == document_id, Document.user_id == user_id)
            )
            await db.commit()
        return result.rowcount > 0

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._require_client()
        batch_size = settings.EMBEDDING_BATCH_SIZE
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        # Provider stalls are per request, so concurrent batches keep one stall from blocking all.
        semaphore = asyncio.Semaphore(settings.EMBEDDING_CONCURRENCY)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                response = await client.embeddings.create(
                    model=settings.OPENROUTER_EMBEDDING_MODEL, input=batch
                )
            return [item.embedding for item in response.data]

        started = time.perf_counter()
        try:
            results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        except OpenAIError as exc:
            logger.warning("embedding provider request failed: %s", type(exc).__name__)
            raise ProviderUnavailableError(
                "The embedding provider is not available right now."
            ) from exc
        embeddings = [vector for batch in results for vector in batch]
        logger.info(
            "embedded texts",
            extra={
                "texts": len(texts),
                "batches": len(batches),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        if any(len(vector) != settings.EMBEDDING_DIMENSIONS for vector in embeddings):
            logger.error(
                "embedding model %s returned vectors that do not match EMBEDDING_DIMENSIONS=%s",
                settings.OPENROUTER_EMBEDDING_MODEL,
                settings.EMBEDDING_DIMENSIONS,
            )
            raise ProviderUnavailableError("The embedding model is misconfigured on the server.")
        return embeddings

    async def close(self) -> None:
        if self._llm_client is not None:
            await self._llm_client.close()

    @staticmethod
    def _extract_pages(file_bytes: bytes) -> tuple[list[dict[str, int | str]], int]:
        if not file_bytes.startswith(b"%PDF-"):
            raise DocumentProcessingError("The file does not have a valid PDF header.")
        try:
            reader = PdfReader(io.BytesIO(file_bytes), strict=False)
            if reader.is_encrypted:
                raise DocumentProcessingError("The PDF is password protected.")
            # Postgres text columns reject NUL bytes, which some PDF extractors emit.
            pages = [
                {"page": index + 1, "text": (page.extract_text() or "").replace("\x00", "").strip()}
                for index, page in enumerate(reader.pages)
            ]
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError("The PDF could not be read.") from exc
        return [page for page in pages if page["text"]], len(pages)

    @staticmethod
    def _chunk_pages(pages: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
        chunks: list[dict[str, int | str]] = []
        size = settings.RAG_CHUNK_SIZE
        step = size - settings.RAG_CHUNK_OVERLAP
        for page in pages:
            text_value = str(page["text"])
            for start in range(0, len(text_value), step):
                chunk = text_value[start : start + size].strip()
                if chunk:
                    chunks.append({"page": int(page["page"]), "text": chunk})
        return chunks
