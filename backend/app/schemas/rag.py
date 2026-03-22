from pydantic import BaseModel, Field


class DocumentIngestionResponse(BaseModel):
    """Response model returned after a PDF is processed and indexed."""

    document_id: str
    filename: str
    chunks_indexed: int


class IndexedDocument(BaseModel):
    """Metadata summary for an indexed source document."""

    document_id: str
    filename: str
    chunks_indexed: int


class IndexedDocumentsResponse(BaseModel):
    """Response model for listing indexed documents."""

    documents: list[IndexedDocument]


class RagQuestionRequest(BaseModel):
    """Request model for retrieval-augmented questions."""

    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


class RetrievedChunk(BaseModel):
    """Single retrieved chunk with metadata and score."""

    chunk_id: str
    text: str
    source_document: str
    page: int | None = None
    distance: float | None = None


class RagQuestionResponse(BaseModel):
    """Response model for RAG retrieval-only answers."""

    answer: str
    retrieved_context: list[RetrievedChunk]
