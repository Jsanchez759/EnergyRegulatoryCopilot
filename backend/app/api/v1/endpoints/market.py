import logging
from collections.abc import AsyncIterator
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.dependencies import (
    AgentServiceDep,
    CurrentUserDep,
    MarketServiceDep,
    get_current_user,
    rate_limit,
)
from app.core.config import settings
from app.schemas.market import (
    MarketAssistantRequest,
    MarketChart,
    MarketCodeOut,
    MarketCodesResponse,
    MarketIndicator,
    MarketMetricOut,
    MarketMetricsResponse,
    MarketOverviewResponse,
    MarketSeriesOut,
)
from app.services.errors import ProviderUnavailableError
from app.services.market_service import SINERGOX_URL, MarketDataError, MetricInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["market"], dependencies=[Depends(get_current_user)])
market_limit = rate_limit("market", lambda: settings.RATE_LIMIT_MARKET)


def metric_out(metric: MetricInfo) -> MarketMetricOut:
    return MarketMetricOut(
        metric_id=metric.metric_id,
        entity=metric.entity,
        name=metric.name,
        description=metric.description,
        units=metric.units,
        granularity=metric.granularity,
        max_days=metric.max_days,
        recommended=metric.note,
    )


def unavailable() -> HTTPException:
    return HTTPException(
        status_code=503, detail="XM no está respondiendo. Inténtalo de nuevo en unos minutos."
    )


@router.get("/metrics", summary="XM metric catalog, ranked by relevance when q is given")
async def list_metrics(
    market: MarketServiceDep,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
) -> MarketMetricsResponse:
    try:
        metrics = await market.find_metrics(q, limit=limit) if q else await market.catalog()
    except ProviderUnavailableError as exc:
        raise unavailable() from exc
    return MarketMetricsResponse(metrics=[metric_out(item) for item in metrics])


@router.get("/codes", summary="Codes of plants, agents, reservoirs or rivers by name")
async def list_codes(
    market: MarketServiceDep,
    entity: Annotated[str, Query(max_length=40)],
    q: Annotated[str, Query(min_length=2, max_length=100)],
) -> MarketCodesResponse:
    try:
        codes = await market.find_codes(entity, q, limit=12)
    except MarketDataError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc
    except ProviderUnavailableError as exc:
        raise unavailable() from exc
    return MarketCodesResponse(
        codes=[MarketCodeOut(code=item.code, name=item.name, detail=item.detail) for item in codes]
    )


@router.get(
    "/series", summary="An aggregated XM series ready to chart", dependencies=[market_limit]
)
async def get_series(
    market: MarketServiceDep,
    metric_id: Annotated[str, Query(max_length=60)],
    start: date,
    end: date,
    entity: Annotated[str, Query(max_length=40)] = "Sistema",
    codes: Annotated[str | None, Query(max_length=300, description="Comma-separated codes")] = None,
    group_by_technology: bool = False,
) -> MarketSeriesOut:
    code_list = [code.strip() for code in (codes or "").split(",") if code.strip()]
    try:
        series = await market.series(
            metric_id,
            start,
            end,
            entity=entity,
            codes=code_list or None,
            group_by_technology=group_by_technology,
        )
    except MarketDataError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc
    except ProviderUnavailableError as exc:
        raise unavailable() from exc
    return MarketSeriesOut(
        title=series.title,
        metric=metric_out(series.metric),
        start=series.start,
        end=series.end,
        period=series.period,
        unit=series.unit,
        stats=series.stats,
        chart=MarketChart.model_validate(series.chart()),
        request=series.request,
        resolved_from=series.resolved_from,
        source_url=SINERGOX_URL,
    )


@router.get("/overview", summary="Headline market indicators with a 30-day trend")
async def get_overview(market: MarketServiceDep) -> MarketOverviewResponse:
    indicators = await market.overview()
    return MarketOverviewResponse(
        indicators=[MarketIndicator.model_validate(item) for item in indicators]
    )


@router.post(
    "/assistant",
    summary="Ask the market assistant; the series it fetches come back as chart sources",
    response_class=EventSourceResponse,
    dependencies=[rate_limit("agent", lambda: settings.RATE_LIMIT_AGENT)],
)
async def market_assistant(
    payload: MarketAssistantRequest,
    agent_service: AgentServiceDep,
    current: CurrentUserDep,
) -> AsyncIterator[ServerSentEvent]:
    try:
        async for event in agent_service.stream_answer(
            payload.question,
            payload.history,
            current.user.id,
            mode="market",
            context=payload.canvas,
        ):
            yield ServerSentEvent(data=event)
    except ProviderUnavailableError as exc:
        yield ServerSentEvent(data={"error": str(exc)})
        yield ServerSentEvent(data={"done": True})
    except Exception:
        # The response is already streaming, so an unhandled error would just cut the connection.
        logger.exception("market assistant stream failed unexpectedly")
        yield ServerSentEvent(data={"error": "Ocurrió un error inesperado. Inténtalo de nuevo."})
        yield ServerSentEvent(data={"done": True})
