from datetime import date

from pydantic import BaseModel, Field

from app.schemas.agent import ChatMessage


class MarketMetricOut(BaseModel):
    metric_id: str
    entity: str
    name: str
    description: str
    units: str
    granularity: str
    max_days: int
    recommended: str


class MarketMetricsResponse(BaseModel):
    metrics: list[MarketMetricOut]


class MarketCodeOut(BaseModel):
    code: str
    name: str
    detail: str


class MarketCodesResponse(BaseModel):
    codes: list[MarketCodeOut]


class MarketChartSeries(BaseModel):
    name: str
    values: list[float | None]


class MarketChart(BaseModel):
    unit: str
    query: dict | None = None
    x: list[str]
    series: list[MarketChartSeries]


class MarketSeriesOut(BaseModel):
    title: str
    metric: MarketMetricOut
    start: date
    end: date
    period: str
    unit: str
    stats: dict[str, dict[str, str]]
    chart: MarketChart
    request: dict
    resolved_from: str | None
    source_url: str


class MarketIndicator(BaseModel):
    key: str
    label: str
    metric_id: str
    entity: str
    unit: str
    latest_date: str
    latest_value: float
    change: float | None
    change_unit: str
    spark: MarketChart


class MarketOverviewResponse(BaseModel):
    indicators: list[MarketIndicator]


class MarketAssistantRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    # Plain-text descriptions of the charts on the canvas, given to the assistant as context.
    canvas: list[str] = Field(default_factory=list, max_length=6)
