import asyncio

import pytest
from sqlalchemy import delete, func, select

from app.core.config import settings
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
from app.corpus.pipeline import PROCESS_ENTRYPOINT, CorpusPipeline
from app.corpus.storage import LocalBlobStorage
from app.db import SessionFactory
from tests.corpus_helpers import FakeFetcher, doc_url, fixture, listing_url
from tests.helpers import bag_of_words_embedding

GENERAL_2015 = "resoluciones_caracter_general_servicio_energia_electrica_2015.html"
RES_174 = "resolucion_creg_0174_2021.htm"


class FakeEmbeddings:
    available = True

    async def embed(self, texts):
        return [bag_of_words_embedding(text) for text in texts]

    async def close(self):
        pass


class Harness:
    def __init__(self, tmp_path, pages):
        self.fetcher = FakeFetcher(pages)
        self.queued: list[tuple[str, dict, str]] = []

        async def enqueue(entrypoint, payload, dedupe_key):
            self.queued.append((entrypoint, payload, dedupe_key))

        self.pipeline = CorpusPipeline(
            sessions=SessionFactory,
            fetcher=self.fetcher,
            storage=LocalBlobStorage(settings.CORPUS_STORAGE_DIR),
            embeddings=FakeEmbeddings(),
            enqueue=enqueue,
        )

    def run(self, coroutine):
        return asyncio.run(coroutine)

    def scalar(self, statement):
        async def query():
            async with SessionFactory() as db:
                return await db.scalar(statement)

        return asyncio.run(query())


@pytest.fixture
def harness(client, tmp_path):
    async def clear():
        async with SessionFactory() as db:
            for model in (NormEvent, NormRelation, NormArticle, NormFile, NormVersion, Norm):
                await db.execute(delete(model))
            await db.execute(delete(FetchCache))
            await db.execute(delete(CrawlRun))
            await db.commit()

    asyncio.run(clear())

    def build(pages):
        return Harness(tmp_path, pages)

    return build


def test_backfill_queues_each_listed_norm_once(harness) -> None:
    h = harness({listing_url(GENERAL_2015): fixture(GENERAL_2015)})
    run = h.run(h.pipeline.discover_creg(years=[2015], mode="backfill"))

    assert run.status == "succeeded"
    assert run.items_seen == 37
    assert run.items_queued == 37
    assert {entrypoint for entrypoint, _, _ in h.queued} == {PROCESS_ENTRYPOINT}
    assert h.scalar(select(func.count()).select_from(Norm)) == 37
    assert h.scalar(select(func.count()).where(Norm.status == "derogada")) == 9

    h.queued.clear()
    h.run(h.pipeline.discover_creg(years=[2015], mode="backfill"))
    assert len(h.queued) == 37, "unprocessed norms are queued again on the next backfill"


def test_incremental_crawl_skips_unchanged_listings(harness) -> None:
    h = harness({listing_url(GENERAL_2015): fixture(GENERAL_2015)})
    h.run(h.pipeline.discover_creg(years=[2015], mode="incremental"))
    h.queued.clear()

    run = h.run(h.pipeline.discover_creg(years=[2015], mode="incremental"))
    assert run.pages_unchanged == 1
    assert h.queued == []


def test_processing_indexes_text_relations_files_and_events(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))

    assert h.run(h.pipeline.process_norm(norm_id)) == "indexed"

    norm = h.scalar(select(Norm).where(Norm.id == norm_id))
    assert (norm.key, norm.status, norm.year) == ("creg:resolucion:174:2021", "vigente", 2021)
    assert h.scalar(select(func.count()).where(NormArticle.norm_id == norm_id)) > 31
    assert h.scalar(
        select(func.count()).where(
            NormArticle.norm_id == norm_id, NormArticle.embedding.is_not(None)
        )
    ) == h.scalar(select(func.count()).where(NormArticle.norm_id == norm_id))
    assert (
        h.scalar(
            select(func.count()).where(
                NormRelation.to_key == "creg:resolucion:174:2021",
                NormRelation.from_key == "creg:resolucion:101-99:2026",
            )
        )
        >= 2
    )
    original = h.scalar(
        select(NormFile).where(NormFile.norm_id == norm_id, NormFile.kind == "original")
    )
    assert original.label == "Creg174-2021.pdf"
    assert original.storage_key is not None
    assert h.scalar(select(NormEvent.kind).where(NormEvent.norm_id == norm_id)) == (
        "norma_incorporada"
    ), "a 2021 resolution found today is backfill, not news"


def test_reprocessing_uses_conditional_requests_and_versions_changes(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))

    assert h.run(h.pipeline.process_norm(norm_id)) == "not_modified"
    assert h.scalar(select(func.count()).where(NormVersion.norm_id == norm_id)) == 1

    amended = fixture(RES_174).replace(
        b"Regular aspectos t&eacute;cnicos", b"Regular nuevos aspectos t&eacute;cnicos"
    )
    h.fetcher.replace(url, amended)
    assert h.run(h.pipeline.process_norm(norm_id)) == "indexed"
    assert h.scalar(select(func.count()).where(NormVersion.norm_id == norm_id)) == 2
    kinds = h.scalar(select(func.array_agg(NormEvent.kind)).where(NormEvent.norm_id == norm_id))
    assert sorted(kinds) == ["norma_actualizada", "norma_incorporada"]


def test_listing_derogation_requeues_and_changes_status(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))

    assert h.run(h.pipeline._upsert_from_listing(_listing_item(url, derogated=True))) == norm_id
    h.fetcher.replace(url, fixture(RES_174) + b"<!-- regenerated -->")
    h.run(h.pipeline.process_norm(norm_id))
    assert h.scalar(select(Norm.status).where(Norm.id == norm_id)) == "derogada"
    assert h.scalar(
        select(NormEvent.details).where(
            NormEvent.norm_id == norm_id, NormEvent.kind == "estado_cambiado"
        )
    ) == {"from": "vigente", "to": "derogada"}


def _listing_item(url, derogated=False):
    from app.corpus.sources import creg

    match = creg.DOC_HREF.search(url)
    return creg.ListingItem(
        url=url,
        number=creg.normalize_number(match["number"]),
        year=int(match["year"]),
        label="",
        title="",
        derogated=derogated,
        is_draft=False,
    )


def test_new_discovery_closes_interrupted_runs(harness) -> None:
    h = harness({listing_url(GENERAL_2015): fixture(GENERAL_2015)})

    async def interrupted():
        async with SessionFactory() as db:
            db.add(CrawlRun(source="creg", mode="backfill"))
            await db.commit()

    h.run(interrupted())
    h.run(h.pipeline.discover_creg(years=[2015], mode="backfill"))
    statuses = h.scalar(select(func.array_agg(CrawlRun.status)).where(CrawlRun.source == "creg"))
    assert sorted(statuses) == ["abandoned", "succeeded"]


def test_recently_published_norms_are_news(harness, monkeypatch) -> None:
    from app.corpus import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "_recently_published", lambda published: True)
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))
    assert h.scalar(select(NormEvent.kind).where(NormEvent.norm_id == norm_id)) == "norma_nueva"


def test_draft_listing_reclassifies_an_existing_norm_instead_of_duplicating(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))

    draft = _listing_item(url)
    draft = draft.__class__(**{**draft.__dict__, "is_draft": True})
    assert h.run(h.pipeline._upsert_from_listing(draft)) == norm_id
    assert h.scalar(select(func.count()).select_from(Norm)) == 1

    h.fetcher.replace(url, fixture(RES_174) + b"<!-- reclassified -->")
    h.run(h.pipeline.process_norm(norm_id))
    norm = h.scalar(select(Norm).where(Norm.id == norm_id))
    assert (norm.key, norm.status) == ("creg:proyecto_resolucion:174:2021", "proyecto")


def test_the_same_amendment_reported_by_two_pages_is_stored_once(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))

    async def report_same_edge_from_another_page():
        async with SessionFactory() as db:
            other = Norm(
                key="creg:resolucion:101-99:2026",
                source="creg",
                doc_type="resolucion",
                number="101-99",
                year=2026,
                url=doc_url("resolucion_creg_101-99_2026.htm"),
                extra={},
            )
            db.add(other)
            await db.flush()
            edge = await db.scalar(
                select(NormRelation).where(
                    NormRelation.reported_by_id == norm_id, NormRelation.to_article == "Artículo 1"
                )
            )
            edge.reported_by_id = other.id
            await db.commit()

    h.run(report_same_edge_from_another_page())
    amended = fixture(RES_174).replace(b"Regular aspectos", b"Regular nuevos aspectos")
    h.fetcher.replace(url, amended)

    assert h.run(h.pipeline.process_norm(norm_id)) == "indexed"
    assert (
        h.scalar(
            select(func.count()).where(
                NormRelation.to_key == "creg:resolucion:174:2021",
                NormRelation.to_article == "Artículo 1",
                NormRelation.from_key == "creg:resolucion:101-99:2026",
            )
        )
        == 1
    )


def test_reparse_fixes_metadata_from_stored_pages_without_requests(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))

    async def lose_metadata():
        async with SessionFactory() as db:
            norm = await db.get(Norm, norm_id)
            norm.issued_on = None
            await db.commit()

    h.run(lose_metadata())
    requests_before = len(h.fetcher.requests)

    assert h.run(h.pipeline.process_norm(norm_id, reparse=True)) == "reparsed"
    assert len(h.fetcher.requests) == requests_before
    norm = h.scalar(select(Norm).where(Norm.id == norm_id))
    assert str(norm.issued_on) == "2021-10-07"
    assert h.scalar(select(func.count()).where(NormVersion.norm_id == norm_id)) == 1
    assert h.scalar(select(func.count()).where(NormEvent.norm_id == norm_id)) == 1


def test_original_pdf_is_downloaded_once(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))
    h.fetcher.replace(url, fixture(RES_174).replace(b"Regular aspectos", b"Regular otros aspectos"))
    h.run(h.pipeline.process_norm(norm_id))

    assert sum(request.endswith(".pdf") for request in h.fetcher.requests) == 1


def test_reclassifying_a_resolution_as_draft_is_not_a_status_event(harness) -> None:
    url = doc_url(RES_174)
    h = harness({url: fixture(RES_174)})
    norm_id = h.run(h.pipeline._upsert_from_listing(_listing_item(url)))
    h.run(h.pipeline.process_norm(norm_id))

    draft = _listing_item(url)
    h.run(h.pipeline._upsert_from_listing(draft.__class__(**{**draft.__dict__, "is_draft": True})))
    assert h.run(h.pipeline.process_norm(norm_id)) == "not_modified"

    assert h.scalar(select(Norm.status).where(Norm.id == norm_id)) == "proyecto"
    assert (
        h.scalar(
            select(func.count()).where(
                NormEvent.norm_id == norm_id, NormEvent.kind == "estado_cambiado"
            )
        )
        == 0
    )
