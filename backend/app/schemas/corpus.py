from datetime import date, datetime

from pydantic import BaseModel


class NormSummary(BaseModel):
    id: str
    key: str
    label: str
    source: str
    doc_type: str
    number: str
    year: int
    title: str
    status: str
    issued_on: date | None
    published_on: date | None
    url: str
    last_changed_at: datetime | None


class NormListResponse(BaseModel):
    total: int
    items: list[NormSummary]


class NormArticleOut(BaseModel):
    id: str
    position: int
    label: str
    content: str


class NormFileOut(BaseModel):
    id: str
    label: str
    kind: str
    url: str
    stored: bool


class NormRelationOut(BaseModel):
    relation: str
    from_key: str
    to_key: str
    from_article: str
    to_article: str
    evidence: str
    other_norm_id: str | None
    other_norm_label: str


class NormEventOut(BaseModel):
    id: str
    kind: str
    occurred_at: datetime
    details: dict


class NormDetail(NormSummary):
    diario_oficial: str | None
    articles: list[NormArticleOut]
    files: list[NormFileOut]
    amended_by: list[NormRelationOut]
    amends: list[NormRelationOut]
    events: list[NormEventOut]


class CorpusSearchHit(BaseModel):
    norm_id: str
    source: str
    norm_label: str
    title: str
    doc_type: str
    status: str
    year: int
    issued_on: date | None
    url: str
    article_label: str
    content: str
    score: float


class CorpusSearchResponse(BaseModel):
    hits: list[CorpusSearchHit]


class FeedEvent(BaseModel):
    id: str
    kind: str
    occurred_at: datetime
    details: dict
    norm: NormSummary


class FeedResponse(BaseModel):
    events: list[FeedEvent]


class SourceStatus(BaseModel):
    source: str
    norms: int
    indexed: int
    by_status: dict[str, int]
    last_run_at: datetime | None
    last_run_status: str | None
    last_run_errors: int | None
    last_success_at: datetime | None


class CorpusStatusResponse(BaseModel):
    sources: list[SourceStatus]


class SourceCatalogEntry(BaseModel):
    id: str
    name: str
    full_name: str
    url: str
    doc_types: dict[str, str]


class SourceCatalogResponse(BaseModel):
    sources: list[SourceCatalogEntry]
