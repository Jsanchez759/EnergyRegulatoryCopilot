import logging
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, literal_column, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.corpus.models import Norm, NormArticle
from app.services.embeddings import EmbeddingClient
from app.services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)

CANDIDATES_PER_METHOD = 40
RRF_K = 60
# Repealed rules stay findable (users ask about them) but should not outrank current ones.
DEROGATED_WEIGHT = 0.6


@dataclass(frozen=True)
class CorpusHit:
    norm_id: str
    norm_key: str
    norm_label: str
    title: str
    doc_type: str
    status: str
    year: int
    issued_on: date | None
    url: str
    article_id: str
    article_label: str
    content: str
    score: float


class CorpusService:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], embeddings: EmbeddingClient
    ) -> None:
        self._sessions = sessions
        self._embeddings = embeddings

    async def search(
        self,
        query: str,
        limit: int = 8,
        doc_types: list[str] | None = None,
        include_derogated: bool = True,
    ) -> list[CorpusHit]:
        vector = await self._query_vector(query)
        filters = []
        if doc_types:
            filters.append(Norm.doc_type.in_(doc_types))
        if not include_derogated:
            filters.append(Norm.status != "derogada")

        async with self._sessions() as db:
            tsquery = func.websearch_to_tsquery(literal_column("'spanish'"), query)
            lexical = await db.execute(
                select(NormArticle.id)
                .join(Norm, Norm.id == NormArticle.norm_id)
                .where(NormArticle.search.op("@@")(tsquery), *filters)
                .order_by(func.ts_rank_cd(NormArticle.search, tsquery).desc())
                .limit(CANDIDATES_PER_METHOD)
            )
            rankings = [[row for (row,) in lexical]]
            if vector is not None:
                await db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
                semantic = await db.execute(
                    select(NormArticle.id)
                    .join(Norm, Norm.id == NormArticle.norm_id)
                    .where(NormArticle.embedding.is_not(None), *filters)
                    .order_by(NormArticle.embedding.cosine_distance(vector))
                    .limit(CANDIDATES_PER_METHOD)
                )
                rankings.append([row for (row,) in semantic])

            fused: dict[str, float] = {}
            for ranking in rankings:
                for rank, article_id in enumerate(ranking, start=1):
                    fused[article_id] = fused.get(article_id, 0.0) + 1 / (RRF_K + rank)
            if not fused:
                return []

            rows = await db.execute(
                select(NormArticle, Norm)
                .join(Norm, Norm.id == NormArticle.norm_id)
                .where(NormArticle.id.in_(fused))
            )
            hits = [
                CorpusHit(
                    norm_id=norm.id,
                    norm_key=norm.key,
                    norm_label=article.norm_label,
                    title=norm.title,
                    doc_type=norm.doc_type,
                    status=norm.status,
                    year=norm.year,
                    issued_on=norm.issued_on,
                    url=norm.url,
                    article_id=article.id,
                    article_label=article.label,
                    content=article.content,
                    score=fused[article.id]
                    * (DEROGATED_WEIGHT if norm.status == "derogada" else 1.0),
                )
                for article, norm in rows.all()
            ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    async def find_norms(
        self,
        query: str | None,
        doc_type: str | None,
        status: str | None,
        year: int | None,
        offset: int,
        limit: int,
    ) -> tuple[int, list[Norm]]:
        conditions = []
        if query:
            pattern = f"%{query.strip()}%"
            compact = query.strip().replace(" ", "-")
            conditions.append(
                or_(
                    Norm.title.ilike(pattern),
                    Norm.number.ilike(f"%{compact}%"),
                    Norm.key.ilike(f"%{compact}%"),
                )
            )
        if doc_type:
            conditions.append(Norm.doc_type == doc_type)
        if status:
            conditions.append(Norm.status == status)
        if year:
            conditions.append(Norm.year == year)
        async with self._sessions() as db:
            total = await db.scalar(select(func.count()).select_from(Norm).where(*conditions))
            norms = await db.scalars(
                select(Norm)
                .where(*conditions)
                .order_by(Norm.issued_on.desc().nulls_last(), Norm.year.desc(), Norm.number.desc())
                .offset(offset)
                .limit(limit)
            )
            return total or 0, list(norms)

    async def _query_vector(self, query: str) -> list[float] | None:
        if not self._embeddings.available:
            return None
        try:
            return (await self._embeddings.embed([query]))[0]
        except ProviderUnavailableError:
            logger.warning("query embedding unavailable; using lexical corpus search only")
            return None
