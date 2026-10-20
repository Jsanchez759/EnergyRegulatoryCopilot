import pytest
from fastapi.testclient import TestClient

from tests.corpus_helpers import doc_url, fixture
from tests.test_corpus_pipeline import _listing_item, harness  # noqa: F401

RES_174 = "resolucion_creg_0174_2021.htm"
RES_24 = "resolucion_creg_0024_2015.htm"
DRAFT = "resolucion_creg_701-116_2026.htm"


@pytest.fixture
def indexed(harness):  # noqa: F811
    pages = {doc_url(name): fixture(name) for name in (RES_174, RES_24, DRAFT)}
    h = harness(pages)
    ids = {}
    for name in (RES_174, RES_24, DRAFT):
        item = _listing_item(doc_url(name))
        ids[name] = h.run(h.pipeline._upsert_from_listing(item))
        assert h.run(h.pipeline.process_norm(ids[name])) == "indexed"
    return ids


def test_corpus_requires_authentication(client: TestClient) -> None:
    anonymous = TestClient(client.app)
    assert anonymous.get("/api/v1/corpus/norms").status_code == 401
    assert anonymous.get("/api/v1/corpus/search", params={"q": "autogeneración"}).status_code == 401


def test_list_filters_by_type_status_and_text(make_user, indexed) -> None:
    user = make_user()
    everything = user.get("/api/v1/corpus/norms").json()
    assert everything["total"] == 3

    drafts = user.get("/api/v1/corpus/norms", params={"doc_type": "proyecto_resolucion"}).json()
    assert [item["key"] for item in drafts["items"]] == ["creg:proyecto_resolucion:701-116:2026"]
    assert drafts["items"][0]["status"] == "proyecto"

    by_number = user.get("/api/v1/corpus/norms", params={"q": "174"}).json()
    assert [item["key"] for item in by_number["items"]] == ["creg:resolucion:174:2021"]


def test_search_finds_articles_by_topic_and_by_citation(make_user, indexed) -> None:
    user = make_user()
    topic = user.get(
        "/api/v1/corpus/search", params={"q": "autogeneración a pequeña escala"}
    ).json()["hits"]
    assert topic and topic[0]["norm_label"] == "Resolución CREG 174 de 2021"

    citation = user.get("/api/v1/corpus/search", params={"q": "Resolución 24 de 2015"}).json()
    assert citation["hits"][0]["norm_label"] == "Resolución CREG 24 de 2015"


def test_detail_includes_articles_amendments_and_original_file(make_user, indexed) -> None:
    user = make_user()
    detail = user.get(f"/api/v1/corpus/norms/{indexed[RES_174]}").json()

    assert detail["diario_oficial"] == "51.867 de 23 de noviembre de 2021"
    assert any(article["label"] == "Artículo 1" for article in detail["articles"])
    amendment = next(item for item in detail["amended_by"] if item["to_article"] == "Artículo 1")
    assert amendment["from_key"] == "creg:resolucion:101-99:2026"
    assert amendment["relation"] == "modifica"
    assert amendment["other_norm_id"] is None

    original = detail["files"][0]
    assert (original["kind"], original["stored"]) == ("original", True)
    download = user.get(f"/api/v1/corpus/files/{original['id']}")
    assert download.status_code == 200
    assert download.content.startswith(b"%PDF")


def test_feed_and_status_report_the_indexed_norms(make_user, indexed) -> None:
    user = make_user()
    events = user.get("/api/v1/corpus/events").json()["events"]
    assert {event["kind"] for event in events} == {"norma_incorporada"}
    assert len(events) == 3
    news = user.get(
        "/api/v1/corpus/events", params={"kind": "norma_nueva,norma_actualizada"}
    ).json()["events"]
    assert news == []

    status = user.get("/api/v1/corpus/status").json()["sources"]
    assert status == [
        {
            **status[0],
            "source": "creg",
            "norms": 3,
            "indexed": 3,
            "by_status": {"vigente": 2, "proyecto": 1},
        }
    ]


def test_labels_cover_crawled_and_external_norms() -> None:
    from app.corpus.catalog import key_label

    assert key_label("creg:resolucion:101-99:2026") == "Resolución CREG 101 99 de 2026"
    assert key_label("creg:proyecto_resolucion:701-116:2026") == (
        "Proyecto de Resolución CREG 701 116 de 2026"
    )
    assert key_label("co:decreto:1073:2015") == "Decreto 1073 de 2015"
    assert key_label("minenergia:resolucion:40072:2018") == "Resolución MinEnergía 40072 de 2018"


def test_catalog_and_source_filters(make_user, indexed) -> None:
    user = make_user()
    catalog = user.get("/api/v1/corpus/sources").json()["sources"]
    assert catalog[0]["id"] == "creg"
    assert catalog[0]["doc_types"]["proyecto_resolucion"] == "Proyecto de Resolución"

    creg_norms = user.get("/api/v1/corpus/norms", params={"source": "creg"}).json()
    assert creg_norms["total"] == 3
    assert creg_norms["items"][0]["label"].startswith(("Resolución CREG", "Proyecto"))
    assert user.get("/api/v1/corpus/norms", params={"source": "upme"}).json()["total"] == 0
    assert user.get("/api/v1/corpus/events", params={"source": "upme"}).json()["events"] == []


def test_amendments_by_norms_outside_the_corpus_still_have_labels(make_user, indexed) -> None:
    user = make_user()
    detail = user.get(f"/api/v1/corpus/norms/{indexed[RES_174]}").json()
    amendment = next(item for item in detail["amended_by"] if item["to_article"] == "Artículo 1")
    assert amendment["other_norm_label"] == "Resolución CREG 101 99 de 2026"


def test_missing_stored_file_is_a_404_not_a_server_error(make_user, indexed) -> None:
    import os

    from app.core.config import settings

    user = make_user()
    detail = user.get(f"/api/v1/corpus/norms/{indexed[RES_24]}").json()
    original = next(item for item in detail["files"] if item["stored"])
    for root, _, files in os.walk(settings.CORPUS_STORAGE_DIR):
        for name in files:
            if name.endswith(".pdf"):
                os.remove(os.path.join(root, name))

    response = user.get(f"/api/v1/corpus/files/{original['id']}")
    assert response.status_code == 404
