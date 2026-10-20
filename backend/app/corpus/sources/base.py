"""What every source adapter produces, so the pipeline never depends on a specific site."""

import re
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from app.corpus.catalog import norm_label

# Returns the page body, or None when it is unchanged (conditional reads) or unavailable.
Reader = Callable[[str, bool], Awaitable[bytes | None]]

MONTHS = {
    month: index
    for index, month in enumerate(
        (
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ),
        start=1,
    )
}
MONTHS["setiembre"] = 9


@dataclass(frozen=True)
class ListingItem:
    source: str
    url: str
    key: str
    doc_type: str
    number: str
    year: int
    title: str = ""
    derogated: bool = False
    issued_on: date | None = None
    # Listing facts kept in Norm.extra; a change in any of them re-queues the norm.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_draft(self) -> bool:
        return is_draft(self.doc_type)


@dataclass(frozen=True)
class NormRef:
    """What the pipeline already knows about a norm when it parses its page."""

    source: str
    url: str
    key: str
    doc_type: str
    number: str
    year: int
    title: str
    issued_on: date | None
    extra: dict[str, Any]


@dataclass
class ParsedRelation:
    from_key: str
    relation: str
    from_article: str
    to_article: str
    evidence: str
    # None means the norm being parsed is the target, as in CREG's vigencia notes.
    to_key: str | None = None


@dataclass
class ParsedFile:
    url: str
    label: str
    kind: str


@dataclass
class ParsedNorm:
    source: str
    key: str
    doc_type: str
    number: str
    year: int
    title: str
    issued_on: date | None
    published_on: date | None
    diario_oficial: str | None
    derogated: bool
    text: str
    passages: list[tuple[str, str]]
    relations: list[ParsedRelation] = field(default_factory=list)
    files: list[ParsedFile] = field(default_factory=list)
    primary: ParsedFile | None = None
    # The page only introduces the act; its text is in the primary file.
    text_in_primary: bool = False
    topic: str = ""

    @property
    def label(self) -> str:
        return norm_label(self.source, self.doc_type, self.number, self.year, self.topic)


class Source(Protocol):
    id: str
    min_interval_by_host: dict[str, float]
    incremental_cron: str
    recheck_cron: str

    def discover(self, read: Reader, years: list[int], mode: str) -> AsyncIterator[ListingItem]:
        """Yield every listed norm; mode is "backfill" or "incremental"."""
        ...

    def parse(self, content: bytes, norm: NormRef, chunk_chars: int) -> ParsedNorm: ...


def is_draft(doc_type: str) -> bool:
    return doc_type.startswith("proyecto")


def digits_number(raw: str) -> str:
    """'4 0310' -> '40310', '02203' -> '2203': numbering without sector prefixes."""
    return re.sub(r"\D", "", raw).lstrip("0") or "0"


def spanish_date(text: str) -> date | None:
    found = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+(?:de|del)\s+(\d{4})", text)
    if not found or fold(found[2]) not in MONTHS:
        return None
    try:
        return date(int(found[3]), MONTHS[fold(found[2])], int(found[1]))
    except ValueError:
        return None


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fold(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in folded if not unicodedata.combining(char))
