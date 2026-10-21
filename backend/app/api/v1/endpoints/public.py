import logging
import time
from datetime import date, datetime

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.dependencies import DbSessionDep, MarketServiceDep, rate_limit
from app.core.config import settings
from app.corpus.catalog import SOURCES, norm_label
from app.corpus.models import Norm, NormArticle
from app.schemas.market import MarketIndicator
from app.schemas.public import PublicConsultation, PublicNorm, PublicOverview, PublicSource
from app.services.corpus_service import COLOMBIA
from app.services.errors import ProviderUnavailableError
from app.services.market_service import MarketDataError

router = APIRouter(
    prefix="/public",
    tags=["public"],
    dependencies=[rate_limit("public", lambda: settings.RATE_LIMIT_PUBLIC, by="ip")],
)
logger = logging.getLogger(__name__)

CACHE_SECONDS = 600
LANDING_INDICATORS = ("precio_bolsa", "embalses", "demanda")
_cache: dict[str, tuple[float, PublicOverview]] = {}


def _label(norm: Norm) -> str:
    return norm_label(
        norm.source, norm.doc_type, norm.number, norm.year, norm.extra.get("topic", "")
    )


async def _build(db: DbSessionDep, market: MarketServiceDep) -> PublicOverview:
    counts = await db.execute(
        select(Norm.source, func.count(Norm.id))
        .where(Norm.content_sha256.is_not(None))
        .group_by(Norm.source)
    )
    by_source = dict(counts.all())
    articles = await db.scalar(select(func.count(NormArticle.id))) or 0
    today = datetime.now(COLOMBIA).date().isoformat()
    # Dates are ISO strings, so text order is date order.
    comment_end = Norm.extra["comment_end"].astext
    open_filter = comment_end >= today
    open_count = await db.scalar(select(func.count(Norm.id)).where(open_filter)) or 0
    closing = await db.scalars(select(Norm).where(open_filter).order_by(comment_end).limit(3))
    latest = await db.scalars(
        select(Norm)
        .where(Norm.content_sha256.is_not(None), Norm.issued_on.is_not(None))
        .order_by(Norm.issued_on.desc())
        .limit(3)
    )
    try:
        indicators = [item for item in await market.overview() if item["key"] in LANDING_INDICATORS]
    except (MarketDataError, ProviderUnavailableError):
        logger.warning("market indicators unavailable for the landing page")
        indicators = []
    return PublicOverview(
        registration_mode=settings.REGISTRATION_MODE,
        sources=[
            PublicSource(id=source.id, name=source.name, norms=by_source.get(source.id, 0))
            for source in SOURCES.values()
        ],
        articles=articles,
        open_consultations=open_count,
        closing_soon=[
            PublicConsultation(
                label=_label(norm),
                title=norm.title,
                source=norm.source,
                comment_end=date.fromisoformat(norm.extra["comment_end"]),
                url=norm.url,
            )
            for norm in closing
        ],
        latest=[
            PublicNorm(
                label=_label(norm),
                title=norm.title,
                source=norm.source,
                issued_on=norm.issued_on,
                url=norm.url,
            )
            for norm in latest
        ],
        market=[MarketIndicator.model_validate(item) for item in indicators],
    )


@router.get("/overview", summary="Live figures for the public landing page")
async def public_overview(db: DbSessionDep, market: MarketServiceDep) -> PublicOverview:
    cached = _cache.get("overview")
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    overview = await _build(db, market)
    _cache["overview"] = (time.monotonic(), overview)
    return overview
