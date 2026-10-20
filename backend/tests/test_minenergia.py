import io
import json
import zipfile
from dataclasses import replace
from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import aggregate_order_by

from app.corpus.catalog import norm_label
from app.corpus.models import Norm, NormArticle, NormEvent, NormFile, NormRelation, NormVersion
from app.corpus.sources import minenergia as me
from app.corpus.sources.base import NormRef
from app.corpus.text import document_passages, docx_text, pdf_text
from tests.corpus_helpers import minenergia_fixture as fixture
from tests.test_corpus_pipeline import harness  # noqa: F401

PDF = "resolucion_01480_2021.pdf"
DETAIL_URL = "https://normativame.minenergia.gov.co/normatividad/7445/norma/"
FORUM_URL = (
    "https://www.minenergia.gov.co/es/servicio-al-ciudadano/foros/"
    "adoptar-el-plan-de-expansion-de-transmision-2025-2039-05052026/"
)
FORUM_PDF = (
    "https://www.minenergia.gov.co/documents/15648/"
    "1-Resoluci%C3%B3n_Plan_de_Expansi%C3%B3n_Transmisi%C3%B3n_2025-2039.pdf"
)
TITLE_7445 = (
    "Por la cual se modifica la Resolución 40279 de 2021 en relación con el Plan de Expansión"
)


def ref(url, key="", doc_type="resolucion", title="", extra=None, issued_on=None) -> NormRef:
    return NormRef(
        source="minenergia",
        url=url,
        key=key,
        doc_type=doc_type,
        number="",
        year=2026,
        title=title,
        issued_on=issued_on,
        extra=extra or {},
    )


def forum_page() -> dict:
    return json.loads(fixture("foro_5312.json"))


def test_listing_rows_carry_number_type_date_and_title() -> None:
    rows, has_next = me.parse_norm_listing(fixture("list_s2_2025_p1.html"))

    assert has_next
    assert len(rows) == 10
    first = me.norm_item(rows[0])
    assert (first.key, first.number, first.issued_on) == (
        "minenergia:resolucion:2357:2025",
        "2357",
        date(2025, 12, 31),
    )
    assert first.url == "https://normativame.minenergia.gov.co/normatividad/7506/norma/"
    assert first.title.startswith("Por la cual se ordena el giro de recursos")


def test_last_listing_page_has_no_next_and_decrees_use_national_keys() -> None:
    rows, has_next = me.parse_norm_listing(fixture("list_s2_2025_p11.html"))

    assert not has_next
    decrees = [me.norm_item(row) for row in rows if row.doc_type == "decreto"]
    assert [item.key for item in decrees] == ["co:decreto:220:2025", "co:decreto:1538:2025"]
    assert norm_label("minenergia", "decreto", "220", 2025) == "Decreto 220 de 2025"


def test_electricity_filter_by_sector() -> None:
    energy, _ = me.parse_norm_listing(fixture("list_s2_2025_p11.html"))
    kept = {row.number for row in energy if me.is_electricity_norm(me.SECTOR_ENERGY, row.title)}
    assert "40035" not in kept, "palm oil freight is tagged as energy but is not electricity"
    assert {"00238", "0220", "1538"} <= kept

    general, _ = me.parse_norm_listing(fixture("list_s4_2025_p1.html"))
    kept_general = {row.number for row in general if me.is_electricity_norm(4, row.title)}
    assert kept_general == {"02334"}, "only the energy-communities act among budget and mining"


def test_norm_page_finds_status_viewer_pdf_and_annexes() -> None:
    parsed = me.parse_norm_page(fixture("detail_7445.html"), ref(DETAIL_URL, title=TITLE_7445))

    assert (parsed.key, parsed.number, parsed.issued_on) == (
        "minenergia:resolucion:40632:2025",
        "40632",
        date(2025, 12, 29),
    )
    assert not parsed.derogated
    assert parsed.text_in_primary
    assert parsed.primary.url.endswith("lTipo=viewpdf&id=7798")
    assert parsed.primary.label == "Resolucion-40632-29122025.pdf"
    annex = [item for item in parsed.files if item is not parsed.primary]
    assert [(item.label, item.kind) for item in annex] == [
        ("Matriz de comentarios (Resolución 40632).pdf", "pdf")
    ]
    assert [(item.to_key, item.relation) for item in parsed.relations] == [
        ("minenergia:resolucion:40279:2021", "modifica")
    ]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "Por la cual se modifica el Articulo 1 de la Resolución 02203 de 2025, que modificó el "
            "artículo 9 de la Resolución 01756 de 2025",
            [("minenergia:resolucion:2203:2025", "modifica", "Artículo 1")],
        ),
        (
            'Por el cual se corrige un yerro en el decreto 1540 de 2024 el cual "reglamenta"',
            [("co:decreto:1540:2024", "corrige", "")],
        ),
        (
            "Por medio de la cual se corrigen unos yerros contenidos en la Resolución 01949 del "
            "31 diciembre de 2024",
            [("minenergia:resolucion:1949:2024", "corrige", "")],
        ),
        (
            "Por la cual se adiciona la Resolución CREG 101 072 de 2025",
            [("creg:resolucion:101-72:2025", "adiciona", "")],
        ),
        ("Por la cual se ordena el giro de subsidios", []),
    ],
)
def test_amendments_announced_in_titles(title, expected) -> None:
    relations = me.title_relations(title, "minenergia:resolucion:1:2026")
    assert [(item.to_key, item.relation, item.to_article) for item in relations] == expected
    assert all(item.from_key == "minenergia:resolucion:1:2026" for item in relations)


def test_forum_listing_keeps_electricity_drafts_with_their_comment_window() -> None:
    pages = json.loads(fixture("foros_api.json"))["items"]
    kept = {page["id"] for page in pages if me.is_electricity_forum(page)}
    assert {5516, 5514, 5503, 5490} <= kept
    assert not kept & {5587, 5582, 5571, 5493}, "hydrocarbons, mining and royalties are out"

    item = me.forum_item(forum_page())
    assert item.url == FORUM_URL
    assert item.key == "minenergia:proyecto_resolucion:foro-5312:2026"
    assert (
        item.title
        == "Por medio de la cual se adopta el Plan de Expansión de Transmisión 2025 – 2039"
    )
    assert item.extra == {
        "forum_id": 5312,
        "topic": "Adoptar el Plan de Expansión de Transmisión 2025 - 2039",
        "comment_start": "2026-05-05",
        "comment_end": "2026-05-15",
    }


def test_forum_page_picks_the_draft_over_supporting_documents() -> None:
    item = me.forum_item(forum_page())
    norm = ref(item.url, item.key, item.doc_type, item.title, item.extra, item.issued_on)
    parsed = me.parse_forum_page(fixture("foro_5312.html"), norm)

    assert parsed.primary.url == FORUM_PDF
    assert parsed.primary.label == "Proyecto de Resolución"
    assert {item.kind for item in parsed.files} == {"pdf", "xlsx"}
    assert "15 de mayo de 2026" in parsed.text
    assert parsed.label == (
        "Proyecto de Resolución MinEnergía: Adoptar el Plan de Expansión de Transmisión 2025 - 2039"
    )

    retie = me.parse_forum_page(fixture("foro_2898.html"), ref("https://www.minenergia.gov.co/x/"))
    assert (retie.primary.label, retie.primary.kind) == ("Anexo General", "docx")


def test_extracted_documents_are_split_into_articles() -> None:
    text, method = pdf_text(fixture(PDF))
    labels = [label for label, _ in document_passages(text, 2400)]

    assert method == "pdf"
    assert labels[0].startswith("Encabezado y considerandos")
    assert labels[-3:] == ["Artículo 1", "Artículo 2", "Artículo 3"]


def test_docx_text_and_unreadable_pdfs() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "word/document.xml",
            "<w:document><w:body><w:p><w:r><w:t>ARTÍCULO 1. Objeto</w:t></w:r></w:p>"
            '<w:p><w:r><w:t xml:space="preserve">Se modifica el </w:t></w:r>'
            "<w:r><w:t>RETIE &amp; RETILAP</w:t></w:r></w:p></w:body></w:document>",
        )
    assert docx_text(buffer.getvalue()) == "ARTÍCULO 1. Objeto\nSe modifica el RETIE & RETILAP"
    assert pdf_text(b"%PDF-1.4 truncated") == ("", "pdf")


def listing(sector: int, page: int) -> str:
    return me.norm_listing_url(sector, date(2025, 1, 1), date(2025, 12, 31), page)


def forums(offset: int = 0) -> str:
    return f"{me.FORUMS_API}&limit={me.FORUMS_PAGE_SIZE}&offset={offset}"


def test_backfill_reads_both_systems_and_stops_when_pages_wrap(harness) -> None:  # noqa: F811
    first_page = fixture("list_s2_2025_p1.html")
    h = harness(
        {
            listing(me.SECTOR_ENERGY, 1): first_page,
            # Past the last page the site serves page 1 again.
            listing(me.SECTOR_ENERGY, 2): first_page,
            listing(4, 1): fixture("list_s4_2025_p1.html"),
            forums(): fixture("foros_api.json"),
        },
    )
    h.fetcher.validators = False

    run = h.run(h.pipeline.discover("minenergia", years=[2025], mode="backfill"))

    assert run.status == "succeeded"
    assert run.errors == 0
    assert listing(me.SECTOR_ENERGY, 3) not in h.fetcher.requests
    forum_count = sum(
        me.is_electricity_forum(page) for page in json.loads(fixture("foros_api.json"))["items"]
    )
    assert run.items_seen == 10 + 1 + forum_count
    assert h.scalar(select(func.count()).where(Norm.source == "minenergia")) == run.items_seen
    assert h.scalar(select(func.count()).where(Norm.status == "desconocido")) == run.items_seen, (
        "nothing is indexed before the worker processes it"
    )


def test_processing_a_norm_indexes_its_pdf_articles(harness) -> None:  # noqa: F811
    viewer = me.VIEW_PDF.format(id=7798)
    h = harness({DETAIL_URL: fixture("detail_7445.html"), viewer: fixture(PDF)})
    h.fetcher.validators = False
    item = me.norm_item(
        me.NormRow(DETAIL_URL, "40632", "resolucion", date(2025, 12, 29), TITLE_7445)
    )
    norm_id = h.run(h.pipeline._upsert_from_listing(item))

    assert h.run(h.pipeline.process_norm(norm_id)) == "indexed"
    norm = h.scalar(select(Norm).where(Norm.id == norm_id))
    assert (norm.key, norm.status, norm.title) == (
        "minenergia:resolucion:40632:2025",
        "vigente",
        TITLE_7445,
    )
    labels = h.scalar(
        select(func.array_agg(NormArticle.label)).where(NormArticle.norm_id == norm_id)
    )
    assert {"Artículo 1", "Artículo 2", "Artículo 3"} <= set(labels)
    assert h.scalar(select(NormVersion.text_source).where(NormVersion.norm_id == norm_id)) == "pdf"
    original = h.scalar(
        select(NormFile).where(NormFile.norm_id == norm_id, NormFile.kind == "original")
    )
    assert (original.content_type, original.storage_key is not None) == ("application/pdf", True)
    assert h.scalar(select(NormRelation.to_key).where(NormRelation.from_key == norm.key)) == (
        "minenergia:resolucion:40279:2021"
    )

    assert h.run(h.pipeline.process_norm(norm_id)) == "unchanged"
    assert h.fetcher.requests.count(viewer) == 1, "the PDF is downloaded once"

    h.fetcher.replace(
        DETAIL_URL,
        fixture("detail_7445.html").replace(b"<span>Vigente</span>", b"<span>Derogada</span>"),
    )
    h.run(h.pipeline.process_norm(norm_id))
    assert h.scalar(select(Norm.status).where(Norm.id == norm_id)) == "derogada"
    assert h.scalar(
        select(NormEvent.details).where(
            NormEvent.norm_id == norm_id, NormEvent.kind == "estado_cambiado"
        )
    ) == {"from": "vigente", "to": "derogada"}


def test_processing_a_forum_keeps_the_call_and_the_draft_text(harness) -> None:  # noqa: F811
    h = harness({FORUM_URL: fixture("foro_5312.html"), FORUM_PDF: fixture(PDF)})
    h.fetcher.validators = False
    norm_id = h.run(h.pipeline._upsert_from_listing(me.forum_item(forum_page())))

    assert h.run(h.pipeline.process_norm(norm_id)) == "indexed"
    norm = h.scalar(select(Norm).where(Norm.id == norm_id))
    assert (norm.status, norm.issued_on) == ("proyecto", date(2026, 5, 5))
    passages = h.scalar(
        select(func.array_agg(aggregate_order_by(NormArticle.label, NormArticle.position))).where(
            NormArticle.norm_id == norm_id
        )
    )
    assert passages[0] == "Convocatoria a comentarios"
    assert "Artículo 3" in passages
    norm_labels = h.scalar(
        select(func.array_agg(func.distinct(NormArticle.norm_label))).where(
            NormArticle.norm_id == norm_id
        )
    )
    assert norm_labels == [
        "Proyecto de Resolución MinEnergía: Adoptar el Plan de Expansión de Transmisión 2025 - 2039"
    ]

    extended = replace(
        me.forum_item(forum_page()),
        extra={**me.forum_item(forum_page()).extra, "comment_end": "2026-05-29"},
    )
    assert h.run(h.pipeline._upsert_from_listing(extended)) == norm_id, (
        "a new deadline re-queues the draft"
    )


def test_api_exposes_comment_window_and_open_consultations(harness, make_user) -> None:  # noqa: F811
    h = harness({})
    item = me.forum_item(forum_page())
    open_item = replace(
        item,
        url=f"{FORUM_URL}abierto/",
        key="minenergia:proyecto_resolucion:foro-9999:2026",
        extra={**item.extra, "forum_id": 9999, "comment_end": "2099-01-31"},
    )
    h.run(h.pipeline._upsert_from_listing(item))
    h.run(h.pipeline._upsert_from_listing(open_item))
    user = make_user()

    drafts = user.get("/api/v1/corpus/norms", params={"source": "minenergia"}).json()
    assert drafts["total"] == 2
    closed = next(norm for norm in drafts["items"] if norm["comment_end"] == "2026-05-15")
    assert closed["comment_start"] == "2026-05-05"
    assert closed["label"].startswith("Proyecto de Resolución MinEnergía: Adoptar el Plan")

    open_now = user.get("/api/v1/corpus/norms", params={"open_consultation": True}).json()
    assert [norm["key"] for norm in open_now["items"]] == [open_item.key]
