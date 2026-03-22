import asyncio
import io
import math
import hashlib
import threading
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI
from pypdf import PdfReader

from app.core.config import settings
from app.schemas.rag import IndexedDocument, RetrievedChunk


class RagService:
    """Service responsible for document ingestion and vector retrieval."""

    def __init__(self) -> None:
        """Initialize vector database client and OpenRouter-backed clients."""
        persist_path = Path(settings.VECTOR_DB_PATH)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=settings.VECTOR_COLLECTION_NAME)

        self._llm_client = None
        if settings.OPENROUTER_API_KEY:
            self._llm_client = OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
            )

    def _resolve_client(self, api_key_override: str | None) -> OpenAI | None:
        """Resolve request-scoped or default OpenRouter client."""
        if api_key_override:
            return OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=api_key_override,
            )
        return self._llm_client

    async def ingest_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        api_key_override: str | None = None,
    ) -> tuple[str, int]:
        """Async wrapper for PDF ingestion."""
        return await asyncio.to_thread(self._ingest_pdf_sync, file_bytes, filename, api_key_override)

    def _ingest_pdf_sync(
        self,
        file_bytes: bytes,
        filename: str,
        api_key_override: str | None = None,
    ) -> tuple[str, int]:
        """Extract text from a PDF, split it into chunks, and store vectors."""
        document_id = str(uuid.uuid4())
        pages = self._extract_pages(file_bytes=file_bytes)
        chunks = self._chunk_pages(pages=pages)

        if not chunks:
            return document_id, 0

        chunk_ids = [f"{document_id}:{idx}" for idx in range(len(chunks))]
        texts = [item["text"] for item in chunks]
        embeddings = self._embed_texts(texts, api_key_override)

        metadatas = []
        for item in chunks:
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page": item["page"],
                }
            )

        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return document_id, len(chunks)

    async def retrieve(
        self,
        question: str,
        top_k: int,
        api_key_override: str | None = None,
    ) -> list[RetrievedChunk]:
        """Async wrapper for vector retrieval."""
        return await asyncio.to_thread(self._retrieve_sync, question, top_k, api_key_override)

    def _retrieve_sync(
        self,
        question: str,
        top_k: int,
        api_key_override: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks relevant to the input question."""
        query_embedding = self._embed_texts([question], api_key_override)[0]

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            chunk_id = ids[idx] if idx < len(ids) else "unknown"

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_document=str(metadata.get("filename", "unknown")),
                    page=metadata.get("page"),
                    distance=distance,
                )
            )

        return chunks

    async def list_indexed_documents(self) -> list[IndexedDocument]:
        """Async wrapper for listing indexed documents."""
        return await asyncio.to_thread(self._list_indexed_documents_sync)

    def _list_indexed_documents_sync(self) -> list[IndexedDocument]:
        """Return unique indexed documents with chunk counts."""
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", []) or []

        counters: dict[str, IndexedDocument] = {}
        for metadata in metadatas:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id", "unknown"))
            filename = str(metadata.get("filename", "unknown.pdf"))

            if document_id not in counters:
                counters[document_id] = IndexedDocument(
                    document_id=document_id,
                    filename=filename,
                    chunks_indexed=0,
                )
            counters[document_id].chunks_indexed += 1

        return sorted(counters.values(), key=lambda item: item.filename.lower())

    async def answer_with_context(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        api_key_override: str | None = None,
    ) -> str:
        """Async wrapper for grounded answer generation."""
        return await asyncio.to_thread(self._answer_with_context_sync, question, chunks, api_key_override)

    async def stream_answer_with_context(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        api_key_override: str | None = None,
    ):
        """Stream grounded answer chunks from OpenRouter when available."""
        if not chunks:
            for chunk in self._chunk_text_fallback(
                "No se encontro contexto relevante. Carga al menos un PDF antes de consultar."
            ):
                yield chunk
            return

        context_blocks = []
        for chunk in chunks:
            context_blocks.append(f"[source={chunk.source_document}, page={chunk.page}]\n{chunk.text}")
        combined_context = "\n\n".join(context_blocks)

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en regulacion energetica colombiana. "
                    "Responde siempre en espanol. "
                    "Responde solo con base en el contexto entregado. "
                    "Si el contexto no alcanza, indicalo explicitamente."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Pregunta: {question}\n\n"
                    f"Contexto:\n{combined_context}\n\n"
                    "Entrega una respuesta clara en espanol y menciona archivo/pagina cuando aplique."
                ),
            },
        ]

        client = self._resolve_client(api_key_override)
        if client is None:
            fallback = (
                "Respuesta en modo prueba (sin API key) para RAG. "
                f"Pregunta: '{question}'.\n\nContexto principal:\n{combined_context[:1200]}"
            )
            for chunk in self._chunk_text_fallback(fallback):
                yield chunk
            return

        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                stream = client.chat.completions.create(
                    model=settings.OPENROUTER_CHAT_MODEL,
                    messages=messages,
                    extra_body={"reasoning": {"enabled": True}},
                    stream=True,
                )
                for event in stream:
                    delta = event.choices[0].delta.content
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                fallback = (
                    "Respuesta de contingencia RAG. "
                    f"Pregunta: '{question}'. "
                    "La llamada a OpenRouter fallo, pero el endpoint sigue operativo."
                )
                for chunk in self._chunk_text_fallback(fallback):
                    yield chunk
                break
            yield item

    def _answer_with_context_sync(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        api_key_override: str | None = None,
    ) -> str:
        """Generate a grounded answer from retrieved chunks using OpenRouter chat completion."""
        if not chunks:
            return "No relevant context was found. Upload at least one PDF first."

        context_blocks = []
        for chunk in chunks:
            context_blocks.append(
                f"[source={chunk.source_document}, page={chunk.page}]\n{chunk.text}"
            )
        combined_context = "\n\n".join(context_blocks)

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en regulacion energetica colombiana. "
                    "Responde siempre en espanol. "
                    "Responde solo con base en el contexto entregado. "
                    "Si el contexto no alcanza, indicalo explicitamente."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Pregunta: {question}\n\n"
                    f"Contexto:\n{combined_context}\n\n"
                    "Entrega una respuesta clara en espanol y menciona archivo/pagina cuando aplique."
                ),
            },
        ]

        client = self._resolve_client(api_key_override)
        if client is None:
            return (
                f"Test mode answer (no OPENROUTER_API_KEY). "
                f"Question: {question}\n\nTop context:\n{combined_context[:1200]}"
            )

        try:
            completion = client.chat.completions.create(
                model=settings.OPENROUTER_CHAT_MODEL,
                messages=messages,
                extra_body={"reasoning": {"enabled": True}},
            )
            content = completion.choices[0].message.content
            return content or "The model returned an empty response."
        except Exception:
            return (
                "Test mode fallback answer (chat provider failed). "
                f"Question: {question}\n\nTop context:\n{combined_context[:1200]}"
            )

    def _embed_texts(self, texts: list[str], api_key_override: str | None = None) -> list[list[float]]:
        """Create vector embeddings for a list of texts using OpenRouter embeddings."""
        client = self._resolve_client(api_key_override)
        if client is None:
            return [self._local_embedding(text) for text in texts]

        try:
            response = client.embeddings.create(
                model=settings.OPENROUTER_EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception:
            return [self._local_embedding(text) for text in texts]

    def _local_embedding(self, text: str, dims: int = 256) -> list[float]:
        """Build a deterministic local embedding for offline test mode."""
        vector = [0.0] * dims
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def _chunk_text_fallback(self, text: str, size: int = 80) -> list[str]:
        """Split fallback text into streamable chunks."""
        return [text[i : i + size] for i in range(0, len(text), size)]

    def _extract_pages(self, file_bytes: bytes) -> list[dict[str, int | str]]:
        """Read PDF pages and return extracted text with page numbers."""
        buffer = io.BytesIO(file_bytes)
        reader = PdfReader(buffer)

        pages: list[dict[str, int | str]] = []
        for page_index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": page_index + 1, "text": text})

        return pages

    def _chunk_pages(self, pages: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
        """Split page text into fixed-size chunks with overlap."""
        chunk_size = settings.RAG_CHUNK_SIZE
        chunk_overlap = settings.RAG_CHUNK_OVERLAP

        chunks: list[dict[str, int | str]] = []
        for page in pages:
            text = str(page["text"])
            page_number = int(page["page"])

            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({"page": page_number, "text": chunk_text})

                if end >= text_len:
                    break
                start = max(0, end - chunk_overlap)

        return chunks
