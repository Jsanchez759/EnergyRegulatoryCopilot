from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AgentRequest(BaseModel):
    conversation_id: str
    question: str = Field(min_length=3, max_length=2000)


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="Nueva consulta", min_length=1, max_length=200)


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class EvidenceSource(BaseModel):
    id: str
    kind: Literal["document", "web"]
    title: str
    snippet: str
    url: str | None = None
    page: int | None = None
    score: float | None = None
    file_sha256: str | None = None
    indexed_at: str | None = None


class AuditStep(BaseModel):
    tool: str
    query: str
    summary: str
    source_ids: list[str] = Field(default_factory=list)


class AgentAudit(BaseModel):
    method: str
    steps: list[AuditStep] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)


class ConversationMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    audit: AgentAudit | None = None
    created_at: str


class ConversationMessagesResponse(BaseModel):
    messages: list[ConversationMessageResponse]


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class IndexedDocument(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class IndexedDocumentsResponse(BaseModel):
    documents: list[IndexedDocument]
