import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.agent_service import AgentService
from app.services.document_service import DocumentProcessingError, DocumentService


def test_search_without_documents_returns_no_results(client: TestClient) -> None:
    service: DocumentService = client.app.state.document_service
    assert asyncio.run(service.has_documents("nobody")) is False
    assert asyncio.run(service.search("autogeneración", user_id="nobody")) == []


def test_chunks_keep_page_numbers_and_overlap() -> None:
    text = "a" * (settings.RAG_CHUNK_SIZE * 2)
    chunks = DocumentService._chunk_pages([{"page": 3, "text": text}])
    step = settings.RAG_CHUNK_SIZE - settings.RAG_CHUNK_OVERLAP
    assert len(chunks) == len(range(0, len(text), step))
    assert {chunk["page"] for chunk in chunks} == {3}
    assert all(len(str(chunk["text"])) <= settings.RAG_CHUNK_SIZE for chunk in chunks)


def test_rejects_non_pdf_bytes() -> None:
    with pytest.raises(DocumentProcessingError):
        DocumentService._extract_pages(b"not a pdf")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"choices": [{"finish_reason": "length"}]}, "length"),
        ({"choices": [{"finish_reason": "stop"}]}, "stop"),
        ({"choices": []}, None),
        (None, None),
    ],
)
def test_finish_reason(raw, expected) -> None:
    assert AgentService._finish_reason(raw) == expected


def test_sources_are_deduplicated_and_numbered_per_kind() -> None:
    from app.schemas.agent import EvidenceSource

    sources: dict[str, EvidenceSource] = {}

    def web(url: str) -> EvidenceSource:
        return EvidenceSource(id="", kind="web", title=url, url=url, snippet="texto")

    first = AgentService._add_source(sources, web("https://creg.gov.co/a"))
    duplicate = AgentService._add_source(sources, web("https://creg.gov.co/a"))
    second = AgentService._add_source(sources, web("https://creg.gov.co/b"))
    document = AgentService._add_source(
        sources, EvidenceSource(id="", kind="document", title="r.pdf", page=1, snippet="x")
    )

    assert (first.id, duplicate.id, second.id, document.id) == ("W1", "W1", "W2", "D1")
