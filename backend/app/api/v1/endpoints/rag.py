import json

from fastapi import APIRouter, File, HTTPException, Header, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.rag import (
    DocumentIngestionResponse,
    IndexedDocumentsResponse,
    RagQuestionRequest,
    RagQuestionResponse,
)
from app.services.rag_service import RagService

router = APIRouter()
rag_service = RagService()


@router.post("/documents/upload", response_model=DocumentIngestionResponse, summary="Upload and index a PDF")
async def upload_document(
    file: UploadFile = File(...),
    x_openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-Api-Key"),
) -> DocumentIngestionResponse:
    """Upload a PDF file and index its content in the vector database."""
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        document_id, chunks_indexed = await rag_service.ingest_pdf(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded.pdf",
            api_key_override=x_openrouter_api_key,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc

    return DocumentIngestionResponse(
        document_id=document_id,
        filename=file.filename or "uploaded.pdf",
        chunks_indexed=chunks_indexed,
    )


@router.get("/documents", response_model=IndexedDocumentsResponse, summary="List indexed documents")
async def list_documents() -> IndexedDocumentsResponse:
    """Return indexed documents currently available in the vector database."""
    documents = await rag_service.list_indexed_documents()
    return IndexedDocumentsResponse(documents=documents)


@router.post("/ask", response_model=RagQuestionResponse, summary="Ask with retrieved PDF context")
async def ask_with_rag(
    payload: RagQuestionRequest,
    x_openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-Api-Key"),
) -> RagQuestionResponse:
    """Retrieve relevant chunks from indexed documents and return contextual answer."""
    chunks = await rag_service.retrieve(
        question=payload.question,
        top_k=payload.top_k,
        api_key_override=x_openrouter_api_key,
    )

    if not chunks:
        return RagQuestionResponse(
            answer="No relevant context was found. Upload at least one PDF first.",
            retrieved_context=[],
        )

    answer = await rag_service.answer_with_context(
        question=payload.question,
        chunks=chunks,
        api_key_override=x_openrouter_api_key,
    )
    return RagQuestionResponse(answer=answer, retrieved_context=chunks)


@router.post("/ask/stream", summary="Stream an answer with retrieved PDF context")
async def ask_with_rag_stream(
    payload: RagQuestionRequest,
    x_openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-Api-Key"),
) -> StreamingResponse:
    """Stream RAG answer as server-sent events and send references at the end."""

    async def event_generator():
        try:
            chunks = await rag_service.retrieve(
                question=payload.question,
                top_k=payload.top_k,
                api_key_override=x_openrouter_api_key,
            )
            references = [
                {
                    "id": f"{chunk.chunk_id}-{idx}",
                    "label": f"{chunk.source_document} (p.{chunk.page if chunk.page else '?'})",
                    "chunk": chunk.text,
                    "distance": chunk.distance,
                }
                for idx, chunk in enumerate(chunks)
            ]
            yield f"data: {json.dumps({'references': references}, ensure_ascii=False)}\n\n"

            async for delta in rag_service.stream_answer_with_context(
                payload.question,
                chunks,
                api_key_override=x_openrouter_api_key,
            ):
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
