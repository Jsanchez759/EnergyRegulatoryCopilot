import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import SessionFactory
from app.models import DocumentChunk
from app.services.document_service import DocumentService
from app.services.errors import ProviderUnavailableError
from tests.conftest import PASSWORD
from tests.helpers import bag_of_words_embedding, build_text_pdf

TARIFF_PDF = build_text_pdf(
    [
        "Resolucion de prueba sobre tarifas.",
        "El cargo por confiabilidad se liquida mensualmente en pesos por kilovatio hora.",
    ]
)
CONNECTION_PDF = build_text_pdf(
    [
        "Resolucion de prueba sobre conexiones.",
        "El operador de red responde solicitudes de conexion en nueve dias habiles.",
    ]
)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [bag_of_words_embedding(text) for text in texts]

    monkeypatch.setattr(DocumentService, "_embed_texts", embed)
    monkeypatch.setattr(DocumentService, "_require_client", lambda self: None)


def upload(user: TestClient, name: str, content: bytes):
    return user.post("/api/v1/sources", files={"file": (name, content, "application/pdf")})


def search(client: TestClient, user: TestClient, query: str):
    user_id = user.get("/api/v1/auth/me").json()["id"]
    service: DocumentService = client.app.state.document_service
    return asyncio.run(service.search(query, user_id=user_id, top_k=3))


def test_upload_lists_document_with_provenance(make_user) -> None:
    user = make_user()
    response = upload(user, "tarifas.pdf", TARIFF_PDF)
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["filename"] == "tarifas.pdf"
    assert created["page_count"] == 1
    assert created["chunks_indexed"] >= 1
    assert len(created["file_sha256"]) == 64

    documents = user.get("/api/v1/sources").json()["documents"]
    assert [item["document_id"] for item in documents] == [created["document_id"]]


def test_duplicate_upload_is_rejected(make_user) -> None:
    user = make_user()
    assert upload(user, "tarifas.pdf", TARIFF_PDF).status_code == 201
    response = upload(user, "copia.pdf", TARIFF_PDF)
    assert response.status_code == 409
    assert "tarifas.pdf" in response.json()["detail"]


def test_search_ranks_relevant_chunk_first_with_page_and_hash(client, make_user) -> None:
    user = make_user()
    upload(user, "tarifas.pdf", TARIFF_PDF)
    connection = upload(user, "conexiones.pdf", CONNECTION_PDF).json()

    results = search(client, user, "operador de red dias habiles conexion")
    assert results[0].title == "conexiones.pdf"
    assert results[0].page == 1
    assert results[0].file_sha256 == connection["file_sha256"]
    assert results[0].score > results[-1].score


def test_search_never_returns_other_users_documents(client, make_user) -> None:
    owner, other = make_user(), make_user()
    upload(owner, "conexiones.pdf", CONNECTION_PDF)
    upload(other, "tarifas.pdf", TARIFF_PDF)

    results = search(client, other, "operador de red dias habiles conexion")
    assert {item.title for item in results} == {"tarifas.pdf"}


def test_delete_document_is_scoped_to_owner(make_user) -> None:
    owner, intruder = make_user(), make_user()
    document_id = upload(owner, "tarifas.pdf", TARIFF_PDF).json()["document_id"]

    assert intruder.delete(f"/api/v1/sources/{document_id}").status_code == 404
    assert owner.delete(f"/api/v1/sources/{document_id}").status_code == 204
    assert owner.get("/api/v1/sources").json()["documents"] == []
    assert owner.delete(f"/api/v1/sources/{document_id}").status_code == 404


def test_deleting_account_removes_documents_and_chunks(client, make_user) -> None:
    user = make_user()
    user_id = user.get("/api/v1/auth/me").json()["id"]
    upload(user, "tarifas.pdf", TARIFF_PDF)
    response = user.request("DELETE", "/api/v1/auth/me", json={"password": PASSWORD})
    assert response.status_code == 204

    async def remaining_chunks() -> int:
        async with SessionFactory() as db:
            return await db.scalar(select(func.count()).where(DocumentChunk.user_id == user_id))

    assert asyncio.run(remaining_chunks()) == 0


def test_upload_without_embedding_provider_returns_503(make_user, monkeypatch) -> None:
    def unavailable(self):
        raise ProviderUnavailableError("OPENROUTER_API_KEY is not configured on the server.")

    monkeypatch.setattr(DocumentService, "_require_client", unavailable)
    assert upload(make_user(), "tarifas.pdf", TARIFF_PDF).status_code == 503
