from datetime import date

import pytest

from app.corpus.sources import creg
from app.corpus.text import split_passages
from tests.corpus_helpers import doc_url, fixture, listing_url

GENERAL_2015 = "resoluciones_caracter_general_servicio_energia_electrica_2015.html"
CHRONOLOGICAL_2026 = "resoluciones_por_orden_cronologico_2026.html"


def parse(name: str) -> creg.ParsedNorm:
    return creg.parse_document(fixture(name), doc_url(name), chunk_chars=2400)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0174", "174"),
        ("101-072", "101-72"),
        ("101 99", "101-99"),
        ("101_127", "101-127"),
        ("0004a", "4a"),
    ],
)
def test_numbers_are_normalized_across_old_and_new_formats(raw, expected) -> None:
    assert creg.normalize_number(raw) == expected


def test_electricity_listing_marks_derogated_items() -> None:
    items = creg.parse_listing(fixture(GENERAL_2015), listing_url(GENERAL_2015))
    assert len(items) == 37
    assert sum(item.derogated for item in items) == 9
    first = items[0]
    assert first.url == doc_url("resolucion_creg_0212_2015.htm")
    assert (first.number, first.year) == ("212", 2015)
    assert first.title.startswith("Por la cual se realizan algunos ajustes")
    assert not any(item.title.lower().startswith("derogad") for item in items)


def test_chronological_listing_keeps_only_electricity_prefixes() -> None:
    items = creg.parse_listing(fixture(CHRONOLOGICAL_2026), listing_url(CHRONOLOGICAL_2026))
    prefixes = {item.number.split("-")[0] for item in items}
    assert items
    assert prefixes <= creg.ELECTRICITY_PREFIXES
    assert {"101", "501"} <= prefixes


def test_resolution_metadata_articles_and_amendments() -> None:
    norm = parse("resolucion_creg_0174_2021.htm")

    assert norm.key == "creg:resolucion:174:2021"
    assert norm.label == "Resolución CREG 174 de 2021"
    assert norm.issued_on == date(2021, 10, 7)
    assert norm.diario_oficial == "51.867 de 23 de noviembre de 2021"
    assert norm.title.startswith("Por la cual se regulan las actividades de autogeneración")
    assert not norm.derogated

    labels = {label.split(" (parte")[0] for label, _ in norm.passages}
    assert {f"Artículo {number}" for number in range(1, 32)} <= labels
    assert all(len(body) <= 2400 for _, body in norm.passages)

    first = next(relation for relation in norm.relations if relation.to_article == "Artículo 1")
    assert (first.from_key, first.relation, first.from_article) == (
        "creg:resolucion:101-99:2026",
        "modifica",
        "Artículo 50",
    )
    assert {relation.relation for relation in norm.relations} >= {"modifica", "adiciona"}


def test_older_resolutions_normalize_ordinal_article_labels() -> None:
    norm = parse("resolucion_creg_0024_2015.htm")
    labels = [label for label, _ in norm.passages]
    assert "Artículo 1" in labels
    assert not any("1o" in label for label in labels)


def test_draft_resolution_is_typed_and_dated() -> None:
    norm = parse("resolucion_creg_701-116_2026.htm")
    assert norm.key == "creg:proyecto_resolucion:701-116:2026"
    assert norm.label == "Proyecto de Resolución CREG 701 116 de 2026"
    assert norm.published_on == date(2026, 1, 8)
    assert norm.diario_oficial is None


def test_primary_pdf_skips_supporting_documents() -> None:
    norm = parse("resolucion_creg_101-127_2026.htm")
    assert creg.primary_pdf(norm.files, norm.number).label == "Resolución_CREG_101_127_2026.pdf"
    old = parse("resolucion_creg_0174_2021.htm")
    assert creg.primary_pdf(old.files, old.number).label == "Creg174-2021.pdf"


def test_split_passages_respects_limit_and_keeps_text() -> None:
    text = "\n".join(f"Párrafo {index} " + "palabra " * 60 for index in range(20))
    passages = split_passages(text, 1000)
    assert all(len(passage) <= 1000 for passage in passages)
    assert "".join(passages).replace("\n", "") == text.replace("\n", "")


def test_pre_2022_drafts_are_tagged_in_the_listing() -> None:
    page = "proyectos_resolucion_servicio_energia_electrica_2019.html"
    items = creg.parse_listing(fixture(page), listing_url(page))
    assert items
    assert all(item.is_draft and item.doc_type == "proyecto_resolucion" for item in items)
    assert items[0].number == "192"
    assert items[0].title.startswith("Por la cual se ordena hacer público")


def test_listings_since_2022_classify_drafts_by_prefix_only() -> None:
    items = creg.parse_listing(fixture(CHRONOLOGICAL_2026), listing_url(CHRONOLOGICAL_2026))
    assert not any(item.is_draft for item in items if not item.number.startswith("7"))


def test_draft_hint_overrides_resolution_header() -> None:
    norm = creg.parse_document(
        fixture("resolucion_creg_0024_2015.htm"),
        doc_url("resolucion_creg_0024_2015.htm"),
        chunk_chars=2400,
        draft=True,
    )
    assert norm.key == "creg:proyecto_resolucion:24:2015"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("RESOLUCIÓN 47 DE 2016\n(abril 1o)", date(2016, 4, 1)),
        ("RESOLUCIÓN 174 DE 2021\n(octubre 7)", date(2016, 10, 7)),
        ("RESOLUCIÓN 257 DE 2016\n()", None),
    ],
)
def test_issue_dates_accept_ordinals_and_missing_values(header, expected) -> None:
    assert creg._issued_on(header, 2016) == expected
