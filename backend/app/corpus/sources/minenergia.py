"""Ministerio de Minas y Energía: issued norms (Nexura normograma) and drafts for comments
(Wagtail "Foros").

See docs/spikes/fuentes/minenergia.md for the structure of both systems.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

from app.corpus.sources import creg
from app.corpus.sources.base import (
    ListingItem,
    NormRef,
    ParsedFile,
    ParsedNorm,
    ParsedRelation,
    Reader,
    clean,
    digits_number,
    fold,
)
from app.corpus.text import decode_html, normalize_whitespace

logger = logging.getLogger(__name__)

SOURCE = "minenergia"
NORMOGRAMA = "https://normativame.minenergia.gov.co/"
SITE = "https://www.minenergia.gov.co/"
# Both hosts drop connections under bursts instead of answering 429.
MIN_INTERVAL_SECONDS = 4.0

NORM_LISTING = f"{NORMOGRAMA}loader.php?lServicio=Normatividad&lTipo=User&lFuncion=avanzado"
VIEW_PDF = f"{NORMOGRAMA}loader.php?lServicio=Tools2&lTipo=viewpdf&id={{id}}"
SECTOR_ENERGY = 2
# "General" and "Regulatorio" mix electricity acts with budget, mining and gas ones.
SECTORS = (SECTOR_ENERGY, 4, 19)
MAX_LISTING_PAGES = 80
# Acts are often uploaded weeks after their date, so incremental crawls look back this far.
INCREMENTAL_WINDOW = timedelta(days=120)

FORUMS_API = (
    f"{SITE}api/v2/pages/?type=servicio_ciudadano.ForosPage"
    "&fields=start_date,end_date,sectors_list,tags,introduction&order=-id"
)
FORUMS_PAGE_SIZE = 400
INCREMENTAL_FORUMS = 50
ELECTRICITY_FORUM_SECTORS = {"Energía"}
MIXED_FORUM_SECTORS = {"Regulatorio", "General", "Ambiental y Social"}

ELECTRIC = re.compile(
    r"el[eé]ctric|energ[ií]a\s+(?:solar|e[oó]lica|renovable|social)|"
    r"fuentes\s+no\s+convencionales|\bFNCER?\b|\bFAZNI\b|\bFOES\b|\bFSSRI\b|\bPRONE\b|\bFAER\b|"
    r"\bZNI\b|zonas?\s+no\s+interconectad|\bRETIE\b|\bRETILAP\b|transmisi[oó]n|autogenera|"
    r"comunidades\s+energ[eé]ticas|menores\s+tarifas|hidroel[eé]ctric|e[oó]lic|fotovoltaic|"
    r"geot[eé]rmi|estrechez\s+energ|\bCREG\b|\bUPME\b",
    re.IGNORECASE,
)
# Hydrocarbon acts that end up tagged as energy; mining has its own sector.
NON_ELECTRIC = re.compile(
    r"crudo|hidrocarbur|petr[óo]le|gas\s+natural|\bGLP\b|combustible|carburante|gasolina|"
    r"\bACPM\b|refinaci",
    re.IGNORECASE,
)
TITLE_RELATION = re.compile(
    r"\b(?P<verb>modifican?|adicionan?|derogan?|sustituyen?|subrogan?|corrigen?|aclaran?|"
    r"reglamentan?|prorrogan?|suspenden?)\b"
    r"(?P<span>[^;]{0,160}?)"
    r"\b(?P<doc>Resoluci[óo]n|Decreto|Ley)\s+"
    r"(?:(?P<issuer>CREG|MME|UPME|MinEnerg[íi]a|MinMinas)\s+)?(?:No\.?\s*)?"
    r"(?P<number>\d[\d\s\.]*?\d|\d)\s+(?:de|del)\s+"
    r"(?:\d{1,2}\s+(?:de\s+)?\w+\s+(?:de\s+)?)?(?P<year>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
RELATION_BY_VERB = {
    "modific": "modifica",
    "adicion": "adiciona",
    "derog": "deroga",
    "sustitu": "sustituye",
    "subrog": "sustituye",
    "corrig": "corrige",
    "aclar": "corrige",
    "reglament": "reglamenta",
    "prorrog": "prorroga",
    "suspend": "suspende",
}
NATIONAL_TYPES = {"decreto", "ley"}
SUPPORTING_DOCUMENT = re.compile(
    r"memoria|justificativ|matriz|formul|comentari|respuesta|informe|presentaci", re.IGNORECASE
)
DRAFT_DOCUMENT = re.compile(r"proyecto|resoluci|decreto|documento\s+propuesto", re.IGNORECASE)


@dataclass(frozen=True)
class NormRow:
    url: str
    number: str
    doc_type: str
    issued_on: date | None
    title: str


def norm_listing_url(sector: int, start: date, end: date, page: int) -> str:
    return (
        f"{NORM_LISTING}&id_sector={sector}&vigencia_inicio={start:%d/%m/%Y}"
        f"&vigencia_fin={end:%d/%m/%Y}&genPag={page}"
    )


def doc_type_of(label: str) -> str:
    return re.sub(r"\W+", "_", fold(clean(label))).strip("_") or "norma"


def norm_key(doc_type: str, number: str, year: int) -> str:
    # Decrees and laws are national acts; sharing their key links them with other sources.
    issuer = "co" if doc_type in NATIONAL_TYPES else SOURCE
    return f"{issuer}:{doc_type}:{digits_number(number)}:{year}"


def is_electricity_norm(sector: int, title: str) -> bool:
    if ELECTRIC.search(title):
        return True
    return sector == SECTOR_ENERGY and not NON_ELECTRIC.search(title)


def parse_norm_listing(content: bytes) -> tuple[list[NormRow], bool]:
    """Rows of one results page, and whether a next page exists."""
    soup = BeautifulSoup(decode_html(content), "lxml")
    rows: list[NormRow] = []
    for row in soup.select("#date_table tbody tr"):
        cells = row.find_all("td")
        anchor = row.find("a", href=True)
        if len(cells) < 4 or anchor is None:
            continue
        rows.append(
            NormRow(
                url=urljoin(NORMOGRAMA, anchor["href"]),
                number=clean(cells[0].get_text(" ")),
                doc_type=doc_type_of(cells[1].get_text(" ")),
                issued_on=_day_month_year(cells[2].get_text(" ")),
                title=clean(cells[3].get_text(" ")),
            )
        )
    has_next = any("siguiente" in anchor.get_text().lower() for anchor in soup.select("a[href]"))
    return rows, has_next


def norm_item(row: NormRow) -> ListingItem | None:
    if row.issued_on is None or not re.search(r"\d", row.number):
        return None
    year = row.issued_on.year
    return ListingItem(
        source=SOURCE,
        url=row.url,
        key=norm_key(row.doc_type, row.number, year),
        doc_type=row.doc_type,
        number=digits_number(row.number),
        year=year,
        title=row.title,
        issued_on=row.issued_on,
    )


def parse_norm_page(content: bytes, norm: NormRef) -> ParsedNorm:
    raw = decode_html(content)
    soup = BeautifulSoup(raw, "lxml")
    fields = _labeled_fields(soup)
    doc_type = doc_type_of(fields["Tipo"]) if fields.get("Tipo") else norm.doc_type
    issued_on = _day_month_year(fields.get("Vigencia", "")) or norm.issued_on
    raw_number = fields.get("Nombre estandar", "")
    number = digits_number(raw_number) if re.search(r"\d", raw_number) else norm.number
    year = issued_on.year if issued_on else norm.year
    key = norm_key(doc_type, number, year)

    files: list[ParsedFile] = []
    primary = None
    # The usable PDF link sits in a commented-out iframe; the visible one points to a temporary
    # copy under a path robots.txt disallows.
    viewer = re.search(r"lTipo=viewpdf&(?:amp;)?id=(\d+)", raw)
    if viewer:
        frame = soup.find("iframe", title=True)
        name = frame["title"] if frame and frame["title"].lower().endswith(".pdf") else ""
        primary = ParsedFile(
            url=VIEW_PDF.format(id=viewer[1]), label=name or f"{number}.pdf", kind="pdf"
        )
        files.append(primary)
    for anchor in soup.select("#tabla-anexos a[href]"):
        label = clean(anchor.get("title") or anchor.get_text(" "))
        files.append(
            ParsedFile(url=urljoin(NORMOGRAMA, anchor["href"]), label=label, kind=_kind(label))
        )

    title = norm.title
    return ParsedNorm(
        source=SOURCE,
        key=key,
        doc_type=doc_type,
        number=number,
        year=year,
        title=title,
        issued_on=issued_on,
        published_on=None,
        diario_oficial=None,
        derogated=fold(fields.get("Estado", "")).startswith("derog"),
        text="",
        passages=[],
        relations=title_relations(title, key),
        files=files,
        primary=primary,
        text_in_primary=True,
    )


def forum_item(page: dict[str, Any]) -> ListingItem | None:
    try:
        start = date.fromisoformat(page["start_date"])
        end = date.fromisoformat(page["end_date"]) if page.get("end_date") else None
    except (KeyError, TypeError, ValueError):
        return None
    intro = page.get("introduction") or ""
    kind = re.search(r"proyecto\s+de\s+(resoluci[oó]n|decreto)", fold(intro))
    doc_type = "proyecto_decreto" if kind and kind[1] == "decreto" else "proyecto_resolucion"
    formal_title = re.search(r"[\"“](Por\s+(?:la|el|medio\s+de\s+la)\s+cual[^\"”]+)[\"”]", intro)
    topic = clean(page.get("title", ""))
    return ListingItem(
        source=SOURCE,
        url=SITE.rstrip("/") + urlsplit(page["meta"]["html_url"]).path,
        key=f"{SOURCE}:{doc_type}:foro-{page['id']}:{start.year}",
        doc_type=doc_type,
        number="",
        year=start.year,
        title=clean(formal_title[1]) if formal_title else topic,
        issued_on=start,
        extra={
            "forum_id": page["id"],
            "topic": topic,
            "comment_start": start.isoformat(),
            "comment_end": end.isoformat() if end else None,
        },
    )


def is_electricity_forum(page: dict[str, Any]) -> bool:
    sectors = set(page.get("sectors_list") or [])
    text = " ".join(
        [page.get("title", ""), page.get("introduction") or "", *(page.get("tags") or [])]
    )
    if sectors & ELECTRICITY_FORUM_SECTORS:
        return bool(ELECTRIC.search(text)) or not NON_ELECTRIC.search(page.get("title", ""))
    return bool(sectors & MIXED_FORUM_SECTORS) and bool(ELECTRIC.search(text))


def parse_forum_page(content: bytes, norm: NormRef) -> ParsedNorm:
    soup = BeautifulSoup(decode_html(content), "lxml")
    section = soup.select_one(".link-section-border-text")
    body = section.find_parent("div", class_="mt-3") if section else None
    files: list[ParsedFile] = []
    seen: set[str] = set()
    for anchor in (body or soup).select('a[href*="/documents/"]'):
        url = urljoin(norm.url, anchor["href"])
        if url in seen:
            continue
        seen.add(url)
        name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        files.append(
            ParsedFile(url=url, label=clean(anchor.get_text(" ")) or name, kind=_kind(name))
        )

    text = normalize_whitespace(body.get_text("\n")) if body else ""
    title = norm.title
    return ParsedNorm(
        source=SOURCE,
        key=norm.key,
        doc_type=norm.doc_type,
        number="",
        year=norm.year,
        title=title,
        issued_on=norm.issued_on,
        published_on=None,
        diario_oficial=None,
        derogated=False,
        text=text,
        passages=[("Convocatoria a comentarios", text)] if text else [],
        relations=title_relations(title, norm.key),
        files=files,
        primary=_draft_document(files),
        text_in_primary=True,
        topic=norm.extra.get("topic", ""),
    )


def title_relations(title: str, own_key: str) -> list[ParsedRelation]:
    """Amendments announced in an act's title, e.g. "Por la cual se modifica la Resolución…"."""
    relations: dict[tuple[str, str, str], ParsedRelation] = {}
    for match in TITLE_RELATION.finditer(title):
        target = _referenced_key(match["doc"], match["issuer"], match["number"], int(match["year"]))
        if target == own_key:
            continue
        verb = fold(match["verb"])
        relation = next(
            value for prefix, value in RELATION_BY_VERB.items() if verb.startswith(prefix)
        )
        article = re.search(r"art[íi]culos?\s+(\d+[\w\.]*)", match["span"], re.IGNORECASE)
        to_article = f"Artículo {article[1].rstrip('.')}" if article else ""
        relations.setdefault(
            (target, relation, to_article),
            ParsedRelation(
                from_key=own_key,
                to_key=target,
                relation=relation,
                from_article="",
                to_article=to_article,
                evidence=title[:500],
            ),
        )
    return list(relations.values())


def _referenced_key(doc: str, issuer: str | None, number: str, year: int) -> str:
    kind = fold(doc)
    if kind in NATIONAL_TYPES:
        return f"co:{kind}:{digits_number(number)}:{year}"
    entity = fold(issuer or "")
    if entity == "creg":
        return creg.norm_key("resolucion", number, year)
    if entity == "upme":
        return f"upme:resolucion:{digits_number(number)}:{year}"
    return norm_key("resolucion", number, year)


def _draft_document(files: list[ParsedFile]) -> ParsedFile | None:
    candidates = [
        item
        for item in files
        if item.kind in {"pdf", "docx"}
        and not SUPPORTING_DOCUMENT.search(f"{item.label} {item.url}")
    ]
    named = [item for item in candidates if DRAFT_DOCUMENT.search(item.label)]
    return (named or candidates or [None])[0]


def _labeled_fields(soup: BeautifulSoup) -> dict[str, str]:
    fields = {}
    for strong in soup.find_all("strong"):
        name = clean(strong.get_text()).rstrip(":")
        holder = strong.find_parent("span") or strong
        value = holder.find_next_sibling("span")
        if name and value is not None:
            fields.setdefault(name, clean(value.get_text(" ")))
    return fields


def _day_month_year(value: str) -> date | None:
    try:
        return datetime.strptime(value.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def _kind(name: str) -> str:
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return extension if extension.isalnum() and len(extension) <= 5 else "archivo"


class MinEnergiaSource:
    id = SOURCE
    min_interval_by_host = {
        urlsplit(NORMOGRAMA).hostname: MIN_INTERVAL_SECONDS,
        urlsplit(SITE).hostname: MIN_INTERVAL_SECONDS,
    }
    incremental_cron = "15 */6 * * *"
    # The normograma sends no validators, so status changes need a full weekly pass.
    recheck_cron = "0 3 * * 6"

    async def discover(
        self, read: Reader, years: list[int], mode: str
    ) -> AsyncIterator[ListingItem]:
        if mode == "incremental":
            today = datetime.now(UTC).date()
            windows = [(today - INCREMENTAL_WINDOW, today)]
        else:
            windows = [(date(year, 1, 1), date(year, 12, 31)) for year in years]
        seen: set[str] = set()
        for start, end in windows:
            for sector in SECTORS:
                async for item in self._norms(read, sector, start, end):
                    if item.url not in seen:
                        seen.add(item.url)
                        yield item
        async for item in self._forums(read, years, mode):
            yield item

    async def _norms(
        self, read: Reader, sector: int, start: date, end: date
    ) -> AsyncIterator[ListingItem]:
        first_url = None
        for page in range(1, MAX_LISTING_PAGES + 1):
            content = await read(norm_listing_url(sector, start, end, page), False)
            if content is None:
                return
            rows, has_next = parse_norm_listing(content)
            # Past the last page the site serves page 1 again.
            if not rows or (page > 1 and rows[0].url == first_url):
                return
            first_url = first_url or rows[0].url
            for row in rows:
                item = norm_item(row)
                if item is not None and is_electricity_norm(sector, row.title):
                    yield item
            if not has_next:
                return

    async def _forums(
        self, read: Reader, years: list[int], mode: str
    ) -> AsyncIterator[ListingItem]:
        first_year = min(years)
        offset = 0
        page_size = INCREMENTAL_FORUMS if mode == "incremental" else FORUMS_PAGE_SIZE
        while True:
            content = await read(f"{FORUMS_API}&limit={page_size}&offset={offset}", False)
            if content is None:
                return
            try:
                payload = json.loads(content)
                pages = payload["items"]
            except (ValueError, KeyError):
                logger.warning("unexpected forums API response", extra={"offset": offset})
                return
            for page in pages:
                item = forum_item(page)
                if item is not None and item.year >= first_year and is_electricity_forum(page):
                    yield item
            offset += len(pages)
            if (
                mode == "incremental"
                or not pages
                or offset >= payload.get("meta", {}).get("total_count", 0)
            ):
                return

    def parse(self, content: bytes, norm: NormRef, chunk_chars: int) -> ParsedNorm:
        if urlsplit(norm.url).hostname == urlsplit(NORMOGRAMA).hostname:
            return parse_norm_page(content, norm)
        return parse_forum_page(content, norm)


ADAPTER = MinEnergiaSource()
