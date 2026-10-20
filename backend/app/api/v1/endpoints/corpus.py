from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select

from app.api.dependencies import (
    CorpusServiceDep,
    DbSessionDep,
    StorageDep,
    get_current_user,
    rate_limit,
)
from app.core.config import settings
from app.corpus.catalog import SOURCES, key_label, norm_label
from app.corpus.models import CrawlRun, Norm, NormEvent, NormFile, NormRelation
from app.schemas.corpus import (
    CorpusSearchHit,
    CorpusSearchResponse,
    CorpusStatusResponse,
    FeedEvent,
    FeedResponse,
    NormArticleOut,
    NormDetail,
    NormEventOut,
    NormFileOut,
    NormListResponse,
    NormRelationOut,
    NormSummary,
    SourceCatalogEntry,
    SourceCatalogResponse,
    SourceStatus,
)

router = APIRouter(prefix="/corpus", tags=["corpus"], dependencies=[Depends(get_current_user)])


def summary(norm: Norm) -> NormSummary:
    return NormSummary(
        id=norm.id,
        key=norm.key,
        label=norm_label(norm.source, norm.doc_type, norm.number, norm.year),
        source=norm.source,
        doc_type=norm.doc_type,
        number=norm.number,
        year=norm.year,
        title=norm.title,
        status=norm.status,
        issued_on=norm.issued_on,
        published_on=norm.published_on,
        url=norm.url,
        last_changed_at=norm.last_changed_at,
    )


@router.get("/norms", summary="List regulatory norms with filters")
async def list_norms(
    corpus: CorpusServiceDep,
    q: Annotated[str | None, Query(max_length=200)] = None,
    source: Annotated[str | None, Query(max_length=40)] = None,
    doc_type: Annotated[str | None, Query(max_length=40)] = None,
    status: Annotated[str | None, Query(max_length=20)] = None,
    year: Annotated[int | None, Query(ge=1990, le=2100)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> NormListResponse:
    total, norms = await corpus.find_norms(q, source, doc_type, status, year, offset, limit)
    return NormListResponse(total=total, items=[summary(norm) for norm in norms])


@router.get("/norms/{norm_id}", summary="Norm detail with articles, files and amendments")
async def get_norm(norm_id: str, db: DbSessionDep) -> NormDetail:
    norm = await db.get(Norm, norm_id)
    if norm is None:
        raise HTTPException(status_code=404, detail="Norm not found.")
    await db.refresh(norm, ["articles", "files"])
    incoming = list(await db.scalars(select(NormRelation).where(NormRelation.to_key == norm.key)))
    outgoing = list(await db.scalars(select(NormRelation).where(NormRelation.from_key == norm.key)))
    related_keys = {item.from_key for item in incoming} | {item.to_key for item in outgoing}
    related = {
        item.key: item for item in await db.scalars(select(Norm).where(Norm.key.in_(related_keys)))
    }
    events = await db.scalars(
        select(NormEvent).where(NormEvent.norm_id == norm.id).order_by(NormEvent.occurred_at.desc())
    )

    def relation_out(relation: NormRelation, other_key: str) -> NormRelationOut:
        other = related.get(other_key)
        return NormRelationOut(
            relation=relation.relation,
            from_key=relation.from_key,
            to_key=relation.to_key,
            from_article=relation.from_article,
            to_article=relation.to_article,
            evidence=relation.evidence,
            other_norm_id=other.id if other else None,
            other_norm_label=key_label(other_key),
        )

    return NormDetail(
        **summary(norm).model_dump(),
        diario_oficial=norm.diario_oficial,
        articles=[
            NormArticleOut(
                id=item.id, position=item.position, label=item.label, content=item.content
            )
            for item in norm.articles
        ],
        files=[
            NormFileOut(
                id=item.id,
                label=item.label,
                kind=item.kind,
                url=item.url,
                stored=item.storage_key is not None,
            )
            for item in sorted(norm.files, key=lambda item: item.kind != "original")
        ],
        amended_by=[relation_out(item, item.from_key) for item in incoming],
        amends=[relation_out(item, item.to_key) for item in outgoing],
        events=[
            NormEventOut(
                id=item.id, kind=item.kind, occurred_at=item.occurred_at, details=item.details
            )
            for item in events
        ],
    )


@router.get(
    "/search",
    summary="Hybrid lexical and semantic search over norm articles",
    dependencies=[rate_limit("corpus_search", lambda: settings.RATE_LIMIT_AGENT)],
)
async def search_corpus(
    corpus: CorpusServiceDep,
    q: Annotated[str, Query(min_length=2, max_length=300)],
    limit: Annotated[int, Query(ge=1, le=30)] = 10,
    include_derogated: bool = True,
) -> CorpusSearchResponse:
    hits = await corpus.search(q, limit=limit, include_derogated=include_derogated)
    return CorpusSearchResponse(
        hits=[
            CorpusSearchHit(
                norm_id=hit.norm_id,
                source=hit.source,
                norm_label=hit.norm_label,
                title=hit.title,
                doc_type=hit.doc_type,
                status=hit.status,
                year=hit.year,
                issued_on=hit.issued_on,
                url=hit.url,
                article_label=hit.article_label,
                content=hit.content,
                score=round(hit.score, 6),
            )
            for hit in hits
        ]
    )


@router.get("/sources", summary="Catalog of regulatory sources and their document types")
async def list_sources() -> SourceCatalogResponse:
    return SourceCatalogResponse(
        sources=[
            SourceCatalogEntry(
                id=info.id,
                name=info.name,
                full_name=info.full_name,
                url=info.url,
                doc_types=info.doc_types,
            )
            for info in SOURCES.values()
        ]
    )


@router.get("/events", summary="Latest changes in the corpus")
async def list_events(
    db: DbSessionDep,
    kind: Annotated[
        str | None, Query(max_length=200, description="One event kind or a comma-separated list")
    ] = None,
    source: Annotated[str | None, Query(max_length=40)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> FeedResponse:
    statement = (
        select(NormEvent, Norm)
        .join(Norm, Norm.id == NormEvent.norm_id)
        .order_by(NormEvent.occurred_at.desc())
        .limit(limit)
    )
    if kind:
        statement = statement.where(NormEvent.kind.in_([item.strip() for item in kind.split(",")]))
    if source:
        statement = statement.where(Norm.source == source)
    statement = statement.offset(offset)
    rows = await db.execute(statement)
    return FeedResponse(
        events=[
            FeedEvent(
                id=event.id,
                kind=event.kind,
                occurred_at=event.occurred_at,
                details=event.details,
                norm=summary(norm),
            )
            for event, norm in rows.all()
        ]
    )


@router.get("/status", summary="Coverage and crawl health per source")
async def corpus_status(db: DbSessionDep) -> CorpusStatusResponse:
    counts = await db.execute(
        select(
            Norm.source,
            Norm.status,
            func.count(),
            func.count(Norm.content_sha256),
        ).group_by(Norm.source, Norm.status)
    )
    by_source: dict[str, SourceStatus] = {}
    for source, status, total, indexed in counts.all():
        item = by_source.setdefault(
            source,
            SourceStatus(
                source=source,
                norms=0,
                indexed=0,
                by_status={},
                last_run_at=None,
                last_run_status=None,
                last_run_errors=None,
                last_success_at=None,
            ),
        )
        item.norms += total
        item.indexed += indexed
        item.by_status[status] = total
    for source, item in by_source.items():
        last = await db.scalar(
            select(CrawlRun)
            .where(CrawlRun.source == source)
            .order_by(CrawlRun.started_at.desc())
            .limit(1)
        )
        if last is not None:
            item.last_run_at, item.last_run_status = last.started_at, last.status
            item.last_run_errors = last.errors
        item.last_success_at = await db.scalar(
            select(func.max(CrawlRun.finished_at)).where(
                CrawlRun.source == source, CrawlRun.status == "succeeded"
            )
        )
    return CorpusStatusResponse(sources=list(by_source.values()))


@router.get("/files/{file_id}", summary="Download a stored original file")
async def download_file(file_id: str, db: DbSessionDep, storage: StorageDep) -> Response:
    item = await db.get(NormFile, file_id)
    if item is None or item.storage_key is None:
        raise HTTPException(status_code=404, detail="File not found.")
    content = await storage.get(item.storage_key)
    return Response(
        content=content,
        media_type=item.content_type or "application/octet-stream",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(item.label)}"},
    )
