import hashlib
import io
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from asyncer import asyncify
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import AsyncOpenAI, OpenAIError
from pypdf import PdfReader

from app.core.config import settings
from app.schemas.agent import EvidenceSource, IndexedDocument
from app.services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)
EMBEDDING_BATCH_SIZE = 1


class DocumentProcessingError(ValueError):
    """Raised when an uploaded PDF cannot be safely indexed."""


class DocumentService:
    """Manage PDF ingestion and LlamaIndex retrieval over the Chroma collection."""

    def __init__(self) -> None:
        path = Path(settings.VECTOR_DB_PATH)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.VECTOR_COLLECTION_NAME
        )
        self._llm_client = self._create_openai_client()
        self._index = self._create_index(self._collection) if self._llm_client is not None else None

    @staticmethod
    def _create_openai_client() -> AsyncOpenAI | None:
        if not settings.OPENROUTER_API_KEY:
            return None
        return AsyncOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            timeout=30.0,
            max_retries=0,
        )

    def _require_client(self) -> AsyncOpenAI:
        if self._llm_client is None:
            raise ProviderUnavailableError("OPENROUTER_API_KEY is not configured on the server.")
        return self._llm_client

    async def ingest_pdf(self, file_bytes: bytes, filename: str, user_id: str) -> tuple[str, int]:
        self._require_client()
        pages = await asyncify(self._extract_pages)(file_bytes)
        chunks = self._chunk_pages(pages)
        if not chunks:
            raise DocumentProcessingError("The PDF contains no selectable text to index.")

        embeddings = await self._embed_texts([str(chunk["text"]) for chunk in chunks])
        document_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        created_at = datetime.now(UTC).isoformat()
        await asyncify(self._collection.add)(
            ids=[f"{document_id}:{index}" for index in range(len(chunks))],
            embeddings=embeddings,
            documents=[str(chunk["text"]) for chunk in chunks],
            metadatas=[
                {
                    "document_id": document_id,
                    "user_id": user_id,
                    "filename": filename,
                    "page": int(chunk["page"]),
                    "file_sha256": file_hash,
                    "indexed_at": created_at,
                    "embedding_model": settings.OPENROUTER_EMBEDDING_MODEL,
                    "embedding_dimensions": len(embeddings[0]),
                }
                for chunk in chunks
            ],
        )
        return document_id, len(chunks)

    async def search(self, query: str, user_id: str, top_k: int = 6) -> list[EvidenceSource]:
        if await asyncify(self._collection.count)(where={"user_id": user_id}) == 0:
            return []
        if self._index is None:
            raise ProviderUnavailableError("OPENROUTER_API_KEY is not configured on the server.")
        query_embedding = (await self._embed_texts([query]))[0]
        filters = MetadataFilters(filters=[ExactMatchFilter(key="user_id", value=user_id)])
        retriever = self._index.as_retriever(similarity_top_k=top_k, filters=filters)
        query_bundle = QueryBundle(query_str=query, embedding=query_embedding)
        results: list[NodeWithScore] = await asyncify(retriever.retrieve)(query_bundle)
        return [
            EvidenceSource(
                id=f"D{index}",
                kind="document",
                title=str(node.node.metadata.get("filename", "Uploaded document")),
                page=node.node.metadata.get("page"),
                snippet=node.node.get_content(),
                score=node.score,
                file_sha256=node.node.metadata.get("file_sha256"),
                indexed_at=node.node.metadata.get("indexed_at"),
            )
            for index, node in enumerate(results, start=1)
        ]

    async def list_documents(self, user_id: str) -> list[IndexedDocument]:
        result = await asyncify(self._collection.get)(
            where={"user_id": user_id}, include=["metadatas"]
        )
        documents: dict[str, IndexedDocument] = {}
        for metadata in result.get("metadatas", []) or []:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id", "unknown"))
            item = documents.setdefault(
                document_id,
                IndexedDocument(
                    document_id=document_id,
                    filename=str(metadata.get("filename", "document.pdf")),
                    chunks_indexed=0,
                ),
            )
            item.chunks_indexed += 1
        return sorted(documents.values(), key=lambda document: document.filename.casefold())

    async def delete_user_documents(self, user_id: str) -> None:
        await asyncify(self._collection.delete)(where={"user_id": user_id})

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._require_client()
            embeddings = []
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                response = await client.embeddings.create(
                    model=settings.OPENROUTER_EMBEDDING_MODEL,
                    input=texts[start : start + EMBEDDING_BATCH_SIZE],
                    timeout=30.0,
                )
                embeddings.extend(item.embedding for item in response.data)
        except OpenAIError as exc:
            logger.warning("embedding provider request failed: %s", type(exc).__name__)
            raise ProviderUnavailableError(
                "The embedding provider is not available right now."
            ) from exc
        return embeddings

    def _create_index(self, collection: chromadb.Collection) -> VectorStoreIndex:
        embedding_model = OpenAIEmbedding(
            model_name=settings.OPENROUTER_EMBEDDING_MODEL,
            api_key=settings.OPENROUTER_API_KEY,
            api_base=settings.OPENROUTER_BASE_URL,
            timeout=90.0,
            max_retries=2,
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding_model)

    async def close(self) -> None:
        if self._llm_client is not None:
            await self._llm_client.close()

    @staticmethod
    def _extract_pages(file_bytes: bytes) -> list[dict[str, int | str]]:
        if not file_bytes.startswith(b"%PDF-"):
            raise DocumentProcessingError("The file does not have a valid PDF header.")
        try:
            reader = PdfReader(io.BytesIO(file_bytes), strict=False)
            if reader.is_encrypted:
                raise DocumentProcessingError("The PDF is password protected.")
            pages = [
                {"page": index + 1, "text": (page.extract_text() or "").strip()}
                for index, page in enumerate(reader.pages)
            ]
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError("The PDF could not be read.") from exc
        return [page for page in pages if page["text"]]

    @staticmethod
    def _chunk_pages(pages: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
        chunks: list[dict[str, int | str]] = []
        size = settings.RAG_CHUNK_SIZE
        step = size - settings.RAG_CHUNK_OVERLAP
        for page in pages:
            text = str(page["text"])
            for start in range(0, len(text), step):
                chunk = text[start : start + size].strip()
                if chunk:
                    chunks.append({"page": int(page["page"]), "text": chunk})
        return chunks
