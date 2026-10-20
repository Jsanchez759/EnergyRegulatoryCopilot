"""CREG Gestor Normativo: electricity resolutions and draft resolutions.

See docs/spikes/fuentes/creg.md for the site structure this adapter relies on.
"""

import re
from collections.abc import AsyncIterator
from datetime import date
from urllib.parse import unquote, urljoin

from bs4 import BeautifulSoup

from app.corpus.sources.base import (
    MONTHS,
    ListingItem,
    NormRef,
    ParsedFile,
    ParsedNorm,
    ParsedRelation,
    Reader,
    clean,
    digits_number,
    fold,
    is_draft,
    spanish_date,
)
from app.corpus.text import (
    PREAMBLE,
    decode_html,
    normalize_whitespace,
    section_passages,
    split_sections,
)

SOURCE = "creg"
BASE_URL = "https://gestornormativo.creg.gov.co/gestor/entorno/"
HOST = "gestornormativo.creg.gov.co"
MIN_INTERVAL_SECONDS = 1.0

ELECTRICITY_PAGES = (
    "resoluciones_caracter_general_servicio_energia_electrica",
    "resoluciones_caracter_particular_servicio_energia_electrica",
    "proyectos_resolucion_servicio_energia_electrica",
)
# The per-service pages lag behind; since 2022 the number prefix identifies the sector, so the
# chronological pages fill the gaps.
CHRONOLOGICAL_PAGES = (
    "resoluciones_por_orden_cronologico",
    "proyectos_resolucion_por_orden_cronologico",
)
SECTOR_PREFIX_YEAR = 2022
# 101/501/701: electricity general, particular and draft; 105/705: cross-sector.
ELECTRICITY_PREFIXES = {"101", "501", "701", "105", "505", "705"}


DOC_HREF = re.compile(r"docs/(?P<slug>[a-z_]+?)_creg_(?P<number>[0-9a-z\-]+)_(?P<year>\d{4})\.htm$")
NOTE = re.compile(r"<([^<>]{5,800})>")
AMENDMENT = re.compile(
    r"(?P<subject>Art[ií]culo|Par[áa]grafo|Numeral|Literal|Inciso|Resoluci[óo]n|Cap[ií]tulo|"
    r"Anexo|T[íi]tulo|Secci[óo]n|Definici[óo]n)s?\b[^<>]*?\b"
    r"(?P<verb>modificad|adicionad|derogad|sustituid|subrogad|suspendid|corregid|aclarad|"
    r"reglamentad|prorrogad)[oa]s?\b[^<>]*?\bpor\s+"
    r"(?:(?:el|los)\s+(?:art[ií]culos?|numeral(?:es)?|par[áa]grafos?)\s+"
    r"(?P<from_article>[\w\.]+)[^<>]*?\s+de(?:l| la| el)?\s+)?"
    r"(?:la\s+|el\s+)?(?P<doc>Resoluci[óo]n|Decreto|Ley|Circular)\s+(?:CREG\s+)?"
    r"(?P<number>\d[\d\s\.\-_]*?)\s+de\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
RELATION_BY_VERB = {
    "modificad": "modifica",
    "adicionad": "adiciona",
    "derogad": "deroga",
    "sustituid": "sustituye",
    "subrogad": "sustituye",
    "suspendid": "suspende",
    "corregid": "corrige",
    "aclarad": "corrige",
    "reglamentad": "reglamenta",
    "prorrogad": "prorroga",
}


def listing_urls(year: int) -> list[str]:
    pages = list(ELECTRICITY_PAGES)
    if year >= SECTOR_PREFIX_YEAR:
        pages += CHRONOLOGICAL_PAGES
    return [urljoin(BASE_URL, f"{page}_{year}.html") for page in pages]


def normalize_number(raw: str) -> str:
    """'0174' -> '174', '101-072' or '101 72' -> '101-72', '0004a' -> '4a'."""
    parts = [part for part in re.split(r"[\s\-_\.]+", raw.strip().lower()) if part]
    cleaned = [part.lstrip("0") or "0" for part in parts]
    return "-".join(cleaned)


def norm_key(doc_type: str, number: str, year: int) -> str:
    return f"{SOURCE}:{doc_type}:{normalize_number(number)}:{year}"


def is_electricity(number: str, year: int, from_electricity_page: bool) -> bool:
    if from_electricity_page:
        return True
    prefix = normalize_number(number).split("-")[0]
    return year >= SECTOR_PREFIX_YEAR and prefix in ELECTRICITY_PREFIXES


def parse_listing(content: bytes, page_url: str) -> list[ListingItem]:
    soup = BeautifulSoup(decode_html(content), "lxml")
    from_electricity_page = "energia_electrica" in page_url
    draft_listing = "proyectos_resolucion" in page_url
    items: list[ListingItem] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        match = DOC_HREF.search(anchor["href"])
        if not match:
            continue
        number, year = match["number"], int(match["year"])
        if not is_electricity(number, year, from_electricity_page):
            continue
        url = urljoin(page_url, anchor["href"])
        if url in seen:
            continue
        seen.add(url)
        title_node = anchor.select_one(".descripcion-documento")
        title = clean(title_node.get_text(" ")) if title_node else ""
        derogated = title.lower().startswith("derogad")
        # Before 2022 drafts were numbered as resolutions and only a "Proyecto" tag tells them
        # apart; since then the 7xx prefix does, and draft listings also carry final acts.
        if year >= SECTOR_PREFIX_YEAR:
            draft = normalize_number(number).startswith("7")
        else:
            draft = draft_listing or bool(title_node and title_node.select_one(".proyecto"))
        title = re.sub(r"^proyecto\s+", "", title, flags=re.I)
        items.append(
            listing_item(
                url,
                number,
                year,
                title=re.sub(r"^derogad[oa]\.?\s*[-:]?\s*", "", title, flags=re.I),
                derogated=derogated,
                draft=draft,
            )
        )
    return items


def listing_item(
    url: str, number: str, year: int, title: str = "", derogated: bool = False, draft: bool = False
) -> ListingItem:
    doc_type = "proyecto_resolucion" if draft else "resolucion"
    return ListingItem(
        source=SOURCE,
        url=url,
        key=norm_key(doc_type, number, year),
        doc_type=doc_type,
        number=normalize_number(number),
        year=year,
        title=title,
        derogated=derogated,
    )


def parse_document(content: bytes, url: str, chunk_chars: int, draft: bool = False) -> ParsedNorm:
    """Parse a document page; `draft` carries the listing's classification for pre-2022 drafts,
    whose pages are headed as ordinary resolutions."""
    soup = BeautifulSoup(decode_html(content), "lxml")
    panel = soup.select_one("div.panel-documento") or soup.body or soup
    # Hidden boxes hold superseded wording and editorial notes, not the current text.
    for hidden in panel.select('[class^="caja_vja"], [class*=" caja_vja"]'):
        hidden.decompose()
    lines = [
        clean(node.get_text(" "))
        for node in panel.find_all(["p", "li", "h1", "h2", "h3", "h4", "td"])
        if not node.find_parent(["p", "li", "td"])
    ]
    lines = [line for line in lines if line]
    text = normalize_whitespace("\n".join(lines))

    match = DOC_HREF.search(url)
    if not match:
        raise ValueError(f"Unrecognized CREG document URL: {url}")
    number, year = normalize_number(match["number"]), int(match["year"])
    header = "\n".join(lines[:12])
    draft = draft or bool(re.match(r"PROYECTO DE RESOLUCI", header))
    doc_type = "proyecto_resolucion" if draft else "resolucion"
    key = norm_key(doc_type, number, year)

    passages: list[tuple[str, str]] = []
    relations: list[ParsedRelation] = []
    for label, body in split_sections(lines):
        passages.extend(section_passages(label, body, chunk_chars))
        to_article = "" if label == PREAMBLE else label
        relations.extend(_relations(body, to_article))

    derogated = any(
        relation.relation == "deroga" and relation.to_article == "" for relation in relations
    )
    files = _files(soup, url)
    return ParsedNorm(
        source=SOURCE,
        key=key,
        doc_type=doc_type,
        number=number,
        year=year,
        title=_title(lines),
        issued_on=_issued_on(header, year),
        published_on=_published_on(header),
        diario_oficial=_diario_oficial(header),
        derogated=derogated,
        text=text,
        passages=passages,
        relations=_dedupe(relations),
        files=files,
        primary=primary_pdf(files, number),
    )


def _relations(body: str, to_article: str) -> list[ParsedRelation]:
    relations = []
    for note in NOTE.findall(body):
        match = AMENDMENT.search(note)
        if not match:
            continue
        subject = fold(match["subject"])
        relation = RELATION_BY_VERB[fold(match["verb"])]
        target_article = "" if subject == "resolucion" else to_article
        relations.append(
            ParsedRelation(
                from_key=_referenced_key(match["doc"], match["number"], int(match["year"]), note),
                relation=relation,
                from_article=f"Artículo {match['from_article'].rstrip('.')}"
                if match["from_article"]
                else "",
                to_article=target_article,
                evidence=note.strip()[:500],
            )
        )
    return relations


def _referenced_key(doc: str, number: str, year: int, context: str) -> str:
    kind = fold(doc)
    if kind == "resolucion":
        lowered = fold(context)
        if "ministerio" in lowered or "mme" in lowered.split() or "upme" in lowered:
            entity = "upme" if "upme" in lowered else "minenergia"
            return f"{entity}:resolucion:{digits_number(number)}:{year}"
        return norm_key("resolucion", number, year)
    if kind == "circular":
        return f"{SOURCE}:circular:{normalize_number(number)}:{year}"
    return f"co:{kind}:{digits_number(number)}:{year}"


def _dedupe(relations: list[ParsedRelation]) -> list[ParsedRelation]:
    unique: dict[tuple[str, str, str, str], ParsedRelation] = {}
    for relation in relations:
        key = (relation.from_key, relation.relation, relation.from_article, relation.to_article)
        unique.setdefault(key, relation)
    return list(unique.values())


def _files(soup: BeautifulSoup, page_url: str) -> list[ParsedFile]:
    files: list[ParsedFile] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "$FILE" not in href and "/originales/" not in href:
            continue
        url = urljoin(page_url, href)
        name = unquote(url.rsplit("/", 1)[-1])
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if extension not in {"pdf", "doc", "docx", "xls", "xlsx", "xlsm", "zip"} or url in seen:
            continue
        seen.add(url)
        files.append(ParsedFile(url=url, label=name, kind=extension))
    return files


def primary_pdf(files: list[ParsedFile], number: str) -> ParsedFile | None:
    """The PDF of the act itself, not its supporting documents or comment analyses."""
    pdfs = [item for item in files if item.kind == "pdf"]
    compact = re.sub(r"\D", "", number)
    for item in pdfs:
        name = fold(item.label)
        if name.startswith(("documento_creg", "d-", "anexo")):
            continue
        if compact and compact.lstrip("0") in re.sub(r"\D", "", name):
            return item
    return next(
        (item for item in pdfs if not fold(item.label).startswith(("documento_creg", "d-"))),
        None,
    )


def _title(lines: list[str]) -> str:
    for line in lines[:15]:
        if re.match(r"^Por (la|el|medio de la) cual", line, re.IGNORECASE):
            return line
    return ""


def _issued_on(header: str, year: int) -> date | None:
    match = re.search(r"\((\w+)\s+(\d{1,2})\s*(?:o|º|°)?\)", header)
    if not match or fold(match[1]) not in MONTHS:
        return None
    try:
        return date(year, MONTHS[fold(match[1])], int(match[2]))
    except ValueError:
        return None


def _published_on(header: str) -> date | None:
    match = re.search(r"Publicad[oa] en la p[áa]gina web de la CREG:\s*([^>\n]+)", header)
    return spanish_date(match[1]) if match else None


def _diario_oficial(header: str) -> str | None:
    match = re.search(r"Diario Oficial No\.?\s*([\d\.]+)\s+de\s+([^\n]+?\d{4})", header)
    return f"{match[1]} de {match[2]}" if match else None


class CregSource:
    id = SOURCE
    min_interval_by_host = {HOST: MIN_INTERVAL_SECONDS}
    incremental_cron = "0 */3 * * *"
    # Amendments rewrite existing pages; conditional requests make this weekly pass cheap.
    recheck_cron = "30 4 * * 0"

    async def discover(
        self, read: Reader, years: list[int], mode: str
    ) -> AsyncIterator[ListingItem]:
        for year in years:
            for url in listing_urls(year):
                content = await read(url, mode == "incremental")
                if content is not None:
                    for item in parse_listing(content, url):
                        yield item

    def parse(self, content: bytes, norm: NormRef, chunk_chars: int) -> ParsedNorm:
        return parse_document(content, norm.url, chunk_chars, draft=is_draft(norm.doc_type))


ADAPTER = CregSource()
