"""Live wholesale market data from XM's public API (Sinergox), shaped for the research agent.

The agent never sees raw API payloads: series are aggregated here into a few rows and summary
statistics, with the exact request kept for the citation.
"""

import asyncio
import json
import logging
import math
import re
import time
import unicodedata
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from app.core.config import settings
from app.services.embeddings import EmbeddingClient
from app.services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)

COLOMBIA = timezone(timedelta(hours=-5))
SINERGOX_URL = "https://sinergox.xm.com.co"
GRANULARITY_BY_TYPE = {
    "HourlyEntities": "hourly",
    "DailyEntities": "daily",
    "MonthlyEntities": "monthly",
}
ENDPOINT_BY_GRANULARITY = {"hourly": "hourly", "daily": "daily", "monthly": "monthly"}
# Levels and rates are averaged over a period; energy, money and volumes add up.
AVERAGED_UNITS = {"COP/kWh", "USD/kWh", "%", "m3/s", "W/m2", "°C", "gCO2e/kWh", "kW", "#", ""}
CODE_LISTS = {
    "Recurso": "ListadoRecursos",
    "Agente": "ListadoAgentes",
    "Embalse": "ListadoEmbalses",
    "Rio": "ListadoRios",
}
# The usual answer to common questions, among near-duplicates such as transaction prices or
# first-settlement (TX1) versions.
PREFERRED_METRICS = {
    (
        "PrecBolsNaci",
        "Sistema",
    ): "precio de bolsa nacional horario: la referencia para 'precio de bolsa'",
    ("PPPrecBolsNaci", "Sistema"): "precio de bolsa promedio ponderado diario",
    ("PrecEsca", "Sistema"): "precio de escasez",
    ("DemaReal", "Sistema"): "demanda real del SIN",
    (
        "Gene",
        "Sistema",
    ): "generación total; por tecnología use entity Recurso y group_by_technology",
    ("Gene", "Recurso"): "generación real por planta; filtre con los códigos de las plantas",
    ("PorcVoluUtilDiar", "Sistema"): "nivel agregado de los embalses en % del volumen útil",
    ("AporEner", "Sistema"): "aportes hídricos en energía",
    ("PorcApor", "Sistema"): "aportes hídricos como % de la media histórica",
    ("factorEmisionCO2e", "Sistema"): "factor de emisión de CO2 equivalente del SIN",
}
OVERVIEW = (
    ("precio_bolsa", "Precio de bolsa", "PPPrecBolsNaci", "Sistema"),
    ("precio_escasez", "Precio de escasez", "PrecEsca", "Sistema"),
    ("embalses", "Nivel de embalses", "PorcVoluUtilDiar", "Sistema"),
    ("aportes", "Aportes hídricos", "PorcApor", "Sistema"),
    ("demanda", "Demanda real", "DemaReal", "Sistema"),
)
OVERVIEW_DAYS = 30
CATALOG_TTL_SECONDS = 24 * 3600
CACHE_TTL_SECONDS = {"hourly": 3600, "daily": 3 * 3600, "monthly": 12 * 3600}
CACHE_ENTRIES = 256
WINDOW_DAYS = 30
MONTHLY_WINDOW_DAYS = 730
# A whole entity without a filter (every plant, every agent) is large; keep it to one window.
UNFILTERED_MAX_DAYS = 31
MAX_ROWS = 62
MAX_SERIES = 6
MAX_CHART_POINTS = 400
RRF_K = 60


# (day, entity code, [(hour, value)])
Observation = tuple[date, str, list[tuple[int, float]]]


class MarketDataError(RuntimeError):
    """A request the API rejected or that cannot be answered.

    The message guides the agent; user_message is shown in the interface.
    """

    def __init__(self, message: str, user_message: str) -> None:
        super().__init__(message)
        self.user_message = user_message


@dataclass(frozen=True)
class MetricInfo:
    metric_id: str
    entity: str
    name: str
    description: str
    units: str
    granularity: str
    max_days: int
    filter_hint: str

    @property
    def search_text(self) -> str:
        return f"{self.name}. {self.description} {self.note} ({self.metric_id}, {self.units})"

    @property
    def note(self) -> str:
        return PREFERRED_METRICS.get((self.metric_id, self.entity), "")


@dataclass(frozen=True)
class MarketCode:
    entity: str
    code: str
    name: str
    detail: str


@dataclass
class MarketSeries:
    metric: MetricInfo
    start: date
    end: date
    period: str
    unit: str
    rows: list[tuple[str, dict[str, float]]]
    stats: dict[str, dict[str, Any]]
    request: dict[str, Any]
    chart_rows: list[tuple[str, dict[str, float]]] = field(default_factory=list)
    resolved_from: str | None = None
    # The parameters that reproduce this series, so a client can refetch or extend it.
    query: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return f"XM · {self.metric.name} · {self.start:%Y-%m-%d} a {self.end:%Y-%m-%d}"

    def summary_text(self) -> str:
        lines = [
            f"{self.metric.name} ({self.metric.metric_id}, entidad {self.metric.entity})",
            *(
                [f"Nota: {self.resolved_from!r} no es un id de XM; se usó {self.metric.metric_id}."]
                if self.resolved_from
                else []
            ),
            f"Periodo: {self.start} a {self.end}. Unidad: {self.unit}. Filas por {self.period}.",
        ]
        for name, stats in self.stats.items():
            parts = [f"{key}={value}" for key, value in stats.items()]
            lines.append(f"{name}: " + ", ".join(parts))
        names = list(self.stats)
        lines.append(" | ".join([self.period, *names]))
        for label, values in self.rows:
            lines.append(" | ".join([label, *(_number(values.get(name)) for name in names)]))
        return "\n".join(lines)

    def chart(self) -> dict[str, Any]:
        names = list(self.stats)
        rows = self.chart_rows or self.rows
        return {
            "unit": self.unit,
            "query": self.query,
            "x": [label for label, _ in rows],
            "series": [
                {"name": name, "values": [values.get(name) for _, values in rows]} for name in names
            ],
        }


class MarketService:
    def __init__(
        self,
        embeddings: EmbeddingClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._client = httpx.AsyncClient(
            base_url=settings.XM_API_URL,
            timeout=settings.XM_TIMEOUT_SECONDS,
            headers={"User-Agent": settings.CRAWLER_USER_AGENT},
            transport=transport,
        )
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._catalog: list[MetricInfo] = []
        self._catalog_vectors: list[list[float]] | None = None
        self._catalog_loaded_at = 0.0
        self._codes: dict[str, tuple[float, list[MarketCode]]] = {}
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def catalog(self) -> list[MetricInfo]:
        async with self._lock:
            if self._catalog and time.monotonic() - self._catalog_loaded_at < CATALOG_TTL_SECONDS:
                return self._catalog
            payload = await self._post("lists", {"MetricId": "ListadoMetricas"}, ttl=0)
            self._catalog = parse_catalog(payload)
            self._catalog_loaded_at = time.monotonic()
            self._catalog_vectors = await self._embed([item.search_text for item in self._catalog])
            return self._catalog

    async def find_metrics(self, query: str, limit: int = 6) -> list[MetricInfo]:
        catalog = await self.catalog()
        lexical = sorted(
            range(len(catalog)), key=lambda index: -_lexical_score(query, catalog[index])
        )
        lexical = [index for index in lexical if _lexical_score(query, catalog[index]) > 0]
        rankings = [lexical[:20]]
        if self._catalog_vectors is not None:
            query_vector = await self._embed([query])
            if query_vector:
                similarity = [_cosine(query_vector[0], vector) for vector in self._catalog_vectors]
                rankings.append(sorted(range(len(catalog)), key=lambda i: -similarity[i])[:20])
        scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, index in enumerate(ranking):
                scores[index] += 1 / (RRF_K + rank + 1)
        best = sorted(scores, key=lambda index: -scores[index])[:limit]
        return [catalog[index] for index in best]

    async def find_codes(self, entity: str, query: str, limit: int = 8) -> list[MarketCode]:
        list_id = CODE_LISTS.get(entity)
        if list_id is None:
            raise MarketDataError(
                f"Entity {entity!r} has no code list. Use one of: {', '.join(CODE_LISTS)}.",
                "Esa entidad no tiene un listado de códigos en XM.",
            )
        cached = self._codes.get(entity)
        if cached is None or time.monotonic() - cached[0] > CATALOG_TTL_SECONDS:
            payload = await self._post("lists", {"MetricId": list_id, "Entity": "Sistema"}, ttl=0)
            cached = (time.monotonic(), parse_codes(entity, payload))
            self._codes[entity] = cached
        terms = _terms(query)
        folded_query = _fold(query)

        def score(item: MarketCode) -> int:
            label = _fold(f"{item.code} {item.name}")
            # Both directions: "Ituango" finds "ITUANGO" and "Hidroituango" finds it too.
            inside = sum(word in folded_query for word in re.findall(r"\w{4,}", label))
            return sum(term in label for term in terms) + inside

        scored = [(score(item), item) for item in cached[1]]
        return [item for score, item in sorted(scored, key=lambda pair: -pair[0]) if score][:limit]

    async def series(
        self,
        metric_id: str,
        start: date,
        end: date,
        entity: str = "Sistema",
        codes: list[str] | None = None,
        group_by_technology: bool = False,
    ) -> MarketSeries:
        catalog = await self.catalog()
        metric = next(
            (
                item
                for item in catalog
                if item.metric_id.lower() == metric_id.lower()
                and item.entity.lower() == entity.lower()
            ),
            None,
        )
        if metric is None:
            entities = [
                item.entity for item in catalog if item.metric_id.lower() == metric_id.lower()
            ]
            if entities:
                hint = f" Available entities: {', '.join(sorted(entities))}."
                raise MarketDataError(
                    f"Unknown metric {metric_id!r} for entity {entity!r}.{hint}",
                    f"La métrica {metric_id} no está disponible para «{entity}».",
                )
            # Models guess ids from the question ("generacion"); the closest metric for the same
            # entity answers it, and the series says which one was used.
            similar = await self.find_metrics(f"{metric_id.replace('_', ' ')} {entity}", limit=5)
            same_entity = [item for item in similar if item.entity.lower() == entity.lower()]
            if not same_entity:
                hint = " Similar metrics: " + "; ".join(
                    f"metric_id={item.metric_id} entity={item.entity} ({item.name})"
                    for item in similar
                )
                raise MarketDataError(
                    f"Unknown metric {metric_id!r} for entity {entity!r}.{hint}",
                    f"XM no tiene la métrica {metric_id}.",
                )
            resolved_from, metric = metric_id, same_entity[0]
        else:
            resolved_from = None
        today = datetime.now(COLOMBIA).date()
        end = min(end, today)
        if start > end:
            raise MarketDataError(
                "The start date is after the end date (or in the future).",
                "La fecha inicial es posterior a la final o está en el futuro.",
            )
        span = (end - start).days + 1
        limit = settings.MARKET_MAX_RANGE_DAYS * (3 if metric.granularity == "monthly" else 1)
        if metric.entity != "Sistema" and not codes and not group_by_technology:
            limit = UNFILTERED_MAX_DAYS
        if group_by_technology:
            limit = min(limit, UNFILTERED_MAX_DAYS)
        if span > limit:
            raise MarketDataError(
                f"The range covers {span} days; this query allows at most {limit}. "
                "Narrow it, or use a daily or monthly metric for long periods.",
                f"El rango tiene {span} días y esta consulta admite hasta {limit}. Acórtalo"
                + (" o filtra por códigos." if limit == UNFILTERED_MAX_DAYS else "."),
            )

        body_base: dict[str, Any] = {"MetricId": metric.metric_id, "Entity": metric.entity}
        if codes and metric.entity != "Sistema":
            body_base["Filter"] = [code.strip() for code in codes if code.strip()][:20]
        observations: list[Observation] = []
        window = MONTHLY_WINDOW_DAYS if metric.granularity == "monthly" else WINDOW_DAYS
        cursor = start
        while cursor <= end:
            stop = min(end, cursor + timedelta(days=window - 1))
            body = {
                **body_base,
                "StartDate": cursor.isoformat(),
                "EndDate": stop.isoformat(),
            }
            payload = await self._post(
                ENDPOINT_BY_GRANULARITY[metric.granularity],
                body,
                ttl=CACHE_TTL_SECONDS[metric.granularity],
            )
            observations.extend(parse_observations(payload))
            cursor = stop + timedelta(days=1)

        names: dict[str, str] = {}
        if group_by_technology and metric.entity == "Recurso":
            technology = {
                item.code: item.detail.split(" · ")[0].capitalize() or "Sin clasificar"
                for item in await self._all_codes("Recurso")
            }
            observations = [
                (day, technology.get(code, "Sin clasificar"), values)
                for day, code, values in observations
            ]
        elif metric.entity in CODE_LISTS:
            # Bare codes let a model attribute one plant's figures to another.
            known = {item.code: item.name for item in await self._all_codes(metric.entity)}
            names = {code: f"{code} · {known[code]}" for code in known}
        series = build_series(metric, start, end, observations, names, request=body_base)
        series.resolved_from = resolved_from
        series.query = {
            "metric_id": metric.metric_id,
            "entity": metric.entity,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "codes": body_base.get("Filter", []),
            "group_by_technology": group_by_technology,
        }
        return series

    async def overview(self) -> list[dict[str, Any]]:
        """Latest daily value, weekly change and a 30-day trend of the headline indicators."""
        today = datetime.now(COLOMBIA).date()
        start = today - timedelta(days=OVERVIEW_DAYS)

        async def indicator(key: str, label: str, metric_id: str, entity: str) -> dict | None:
            try:
                series = await self.series(metric_id, start, today, entity=entity)
            except (MarketDataError, ProviderUnavailableError):
                logger.warning("market indicator unavailable", extra={"metric": metric_id})
                return None
            name = next(iter(series.stats), None)
            points = [
                (label, values[name]) for label, values in series.chart_rows if name in values
            ]
            if not points:
                return None
            latest_label, latest = points[-1]
            week_ago = points[-8][1] if len(points) >= 8 else points[0][1]
            # Shares change in percentage points; everything else in relative terms.
            if series.unit == "%":
                change, change_unit = latest - week_ago, "pp"
            else:
                change = (latest - week_ago) / abs(week_ago) * 100 if week_ago else None
                change_unit = "%"
            return {
                "key": key,
                "label": label,
                "metric_id": series.metric.metric_id,
                "entity": series.metric.entity,
                "unit": series.unit,
                "latest_date": latest_label,
                "latest_value": latest,
                "change": change,
                "change_unit": change_unit,
                "spark": series.chart(),
            }

        results = await asyncio.gather(*(indicator(*item) for item in OVERVIEW))
        return [item for item in results if item is not None]

    async def _all_codes(self, entity: str) -> list[MarketCode]:
        await self.find_codes(entity, "")
        return self._codes[entity][1]

    async def _post(self, endpoint: str, body: dict[str, Any], ttl: int) -> Any:
        key = f"{endpoint}:{json.dumps(body, sort_keys=True)}"
        cached = self._cache.get(key)
        if cached and cached[0] > time.monotonic():
            self._cache.move_to_end(key)
            return cached[1]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self._client.post(f"/{endpoint}", json=body)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == 400:
                    raise MarketDataError(
                        f"XM rejected the request: {response.text[:300]}",
                        "XM rechazó la consulta.",
                    )
                if response.status_code < 500:
                    response.raise_for_status()
                    payload = response.json()
                    if ttl:
                        self._cache[key] = (time.monotonic() + ttl, payload)
                        while len(self._cache) > CACHE_ENTRIES:
                            self._cache.popitem(last=False)
                    return payload
                last_error = httpx.HTTPStatusError(
                    f"XM returned HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            await asyncio.sleep(1.5 * 2**attempt)
        logger.warning("XM API unavailable", extra={"endpoint": endpoint})
        raise ProviderUnavailableError("XM's market data API is not responding.") from last_error

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if self._embeddings is None or not self._embeddings.available:
            return None
        try:
            return await self._embeddings.embed(texts)
        except ProviderUnavailableError:
            return None


def parse_catalog(payload: dict[str, Any]) -> list[MetricInfo]:
    metrics: dict[tuple[str, str], MetricInfo] = {}
    for item in payload.get("Items", []):
        for entry in item.get("ListEntities", []):
            values = entry.get("Values") or {}
            granularity = GRANULARITY_BY_TYPE.get(values.get("Type", ""))
            if granularity is None or not values.get("MetricId"):
                continue
            info = MetricInfo(
                metric_id=values["MetricId"],
                entity=values.get("Entity") or "Sistema",
                name=_clean(values.get("MetricName", "")),
                description=_clean(values.get("MetricDescription", "")),
                units=(values.get("MetricUnits") or "").strip(),
                granularity=granularity,
                max_days=int(values.get("MaxDays") or WINDOW_DAYS),
                filter_hint=_clean(values.get("Filter", "")),
            )
            metrics.setdefault((info.metric_id, info.entity), info)
    return list(metrics.values())


def parse_codes(entity: str, payload: dict[str, Any]) -> list[MarketCode]:
    codes = []
    for item in payload.get("Items", []):
        for entry in item.get("ListEntities", []):
            values = entry.get("Values") or {}
            if not values.get("Code"):
                continue
            detail = [
                values.get(key)
                for key in ("Type", "Activity", "HydroRegion", "EnerSource", "State", "Status")
                if values.get(key)
            ]
            codes.append(
                MarketCode(
                    entity=entity,
                    code=values["Code"],
                    name=_clean(values.get("Name", "")),
                    detail=" · ".join(detail),
                )
            )
    return codes


def parse_observations(payload: dict[str, Any]) -> list[Observation]:
    """(day, entity code, [(hour, value)]): hours 1-24, or hour 0 for a daily or monthly value."""
    observations = []
    for item in payload.get("Items", []):
        try:
            day = date.fromisoformat(str(item.get("Date"))[:10])
        except ValueError:
            continue
        for key, entries in item.items():
            if key not in GRANULARITY_BY_TYPE:
                continue
            for entry in entries:
                hourly = entry.get("Values")
                if isinstance(hourly, dict):
                    code = str(hourly.get("code") or entry.get("Id") or "")
                    # Hours without data come empty; keeping positions keeps the rest in place.
                    values = [
                        (int(hour[4:]), _float(hourly[hour]))
                        for hour in sorted(hourly)
                        if hour.startswith("Hour") and _float(hourly[hour]) is not None
                    ]
                else:
                    code = str(entry.get("Name") or entry.get("Code") or entry.get("Id") or "")
                    value = _float(entry.get("Value"))
                    values = [] if value is None else [(0, value)]
                if values:
                    observations.append((day, code, values))
    return observations


def build_series(
    metric: MetricInfo,
    start: date,
    end: date,
    observations: list[Observation],
    names: dict[str, str],
    request: dict[str, Any],
) -> MarketSeries:
    averaged = metric.units in AVERAGED_UNITS
    if metric.granularity == "hourly" and not averaged:
        observations = _drop_partial_last_day(observations)

    def combine(values: list[float]) -> float:
        return sum(values) / len(values) if averaged else sum(values)

    all_values = [value for _, _, values in observations for _, value in values]
    # XM reports these shares as fractions although the unit says %.
    factor = 100.0 if metric.units == "%" and all_values and max(map(abs, all_values)) <= 1.5 else 1
    unit, scale = _display_unit(metric.units, all_values, averaged)
    factor /= scale

    totals: dict[str, float] = defaultdict(float)
    for _, code, values in observations:
        totals[code] += abs(sum(value for _, value in values))
    ranked = sorted(totals, key=lambda code: -totals[code])
    kept = set(ranked[:MAX_SERIES]) if len(ranked) > MAX_SERIES + 1 else set(ranked)

    def series_name(code: str) -> str:
        if code not in kept:
            return "Otros"
        return names.get(code, code)

    hourly = metric.granularity == "hourly"
    detailed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    daily: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for day, code, values in observations:
        name = series_name(code)
        scaled = [(hour, value * factor) for hour, value in values]
        if hourly:
            for hour, value in scaled:
                detailed[f"{day} {hour:02d}h"][name].append(value)
        label = day.strftime("%Y-%m") if metric.granularity == "monthly" else day.isoformat()
        daily[label][name].append(combine([value for _, value in scaled]))

    def collapse(groups: dict[str, dict[str, list[float]]]) -> list[tuple[str, dict[str, float]]]:
        # Several entities folded into one name (technology, "Otros") add up within a period.
        return [
            (label, {name: _add_entities(values, averaged) for name, values in per_name.items()})
            for label, per_name in sorted(groups.items())
        ]

    hours = collapse(detailed) if hourly else []
    days = collapse(daily)
    if hourly and (end - start).days < 2:
        rows, period = hours, "hora"
    elif len(days) > MAX_ROWS and metric.granularity != "monthly":
        months: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for label, values in days:
            for name, value in values.items():
                months[label[:7]][name].append(value)
        rows = [
            (label, {name: combine(values) for name, values in per_name.items()})
            for label, per_name in sorted(months.items())
        ]
        period = "mes"
    else:
        rows, period = days, "mes" if metric.granularity == "monthly" else "día"

    finest = hours or days
    stats: dict[str, dict[str, Any]] = {}
    names_in_order = sorted(
        {name for _, values in finest for name in values}, key=lambda name: name == "Otros"
    )
    unit_of_time = "mes" if metric.granularity == "monthly" else "dia"
    for name in names_in_order:
        day_points = [(label, values[name]) for label, values in days if name in values]
        hour_points = [(label, values[name]) for label, values in hours if name in values]
        stat: dict[str, Any] = {}
        if averaged:
            stat["promedio"] = _number(
                sum(value for _, value in hour_points or day_points)
                / len(hour_points or day_points)
            )
        else:
            stat["total"] = _number(sum(value for _, value in day_points))
            stat[f"promedio_por_{unit_of_time}"] = _number(
                sum(value for _, value in day_points) / len(day_points)
            )
        # Explicit names keep a model from reading one hour's value as a daily total.
        low = min(day_points, key=lambda point: point[1])
        high = max(day_points, key=lambda point: point[1])
        stat[f"{unit_of_time}_minimo"] = f"{_number(low[1])} ({low[0]})"
        stat[f"{unit_of_time}_maximo"] = f"{_number(high[1])} ({high[0]})"
        if hour_points:
            low = min(hour_points, key=lambda point: point[1])
            high = max(hour_points, key=lambda point: point[1])
            stat["hora_minima"] = f"{_number(low[1])} ({low[0]})"
            stat["hora_maxima"] = f"{_number(high[1])} ({high[0]})"
        if len(day_points) > 1 and day_points[0][1]:
            change = (day_points[-1][1] - day_points[0][1]) / abs(day_points[0][1]) * 100
            stat["variacion_inicio_fin"] = f"{change:+.1f}%"
        stats[name] = stat

    chart_rows = days if len(days) <= MAX_CHART_POINTS else rows
    if hourly and period == "hora":
        chart_rows = rows
    return MarketSeries(
        metric=metric,
        start=start,
        end=end,
        period=period,
        unit=unit,
        rows=rows,
        stats=stats,
        request={**request, "StartDate": start.isoformat(), "EndDate": end.isoformat()},
        chart_rows=chart_rows,
    )


def _drop_partial_last_day(observations: list[Observation]) -> list[Observation]:
    """XM fills the current day hour by hour; summing it would read as a collapse."""
    days = sorted({day for day, _, _ in observations})
    if len(days) < 2:
        return observations
    last_hours = max(
        (hour for day, _, values in observations if day == days[-1] for hour, _ in values),
        default=0,
    )
    if last_hours >= 24:
        return observations
    return [item for item in observations if item[0] != days[-1]]


def _add_entities(values: list[float], averaged: bool) -> float:
    if averaged:
        return sum(values) / len(values)
    return sum(values)


def _display_unit(units: str, values: list[float], averaged: bool) -> tuple[str, float]:
    peak = max((abs(value) for value in values), default=0.0)
    magnitude = peak if averaged else peak * 24
    if units == "kWh" and magnitude >= 1e6:
        return "GWh", 1e6
    if units == "kWh" and magnitude >= 1e3:
        return "MWh", 1e3
    if units == "kW" and peak >= 1e3:
        return "MW", 1e3
    if units in {"COP", "m3"} and magnitude >= 1e6:
        return f"millones de {units}", 1e6
    return units, 1.0


def _lexical_score(query: str, metric: MetricInfo) -> float:
    text = _fold(f"{metric.metric_id} {metric.name} {metric.description}")
    name = _fold(metric.name)
    score = 0.0
    for term in _terms(query):
        if term in name:
            score += 2
        elif term in text:
            score += 1
    if score and metric.entity == "Sistema":
        score += 0.5
    note_hits = sum(term in _fold(f"{metric.name} {metric.note}") for term in _terms(query))
    if metric.note and note_hits:
        score += 3 + note_hits
    return score


def _terms(query: str) -> list[str]:
    stop = {"de", "del", "la", "el", "los", "las", "por", "en", "y", "a", "para", "con", "que"}
    # A short stem lets "embalses" match "embalse" and "generación" match "generacion".
    return [word[:6] for word in re.findall(r"\w+", _fold(query)) if word not in stop]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}" if abs(value) < 1e6 else f"{value:,.0f}"


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _fold(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in folded if not unicodedata.combining(char))


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        raise MarketDataError(
            f"Invalid date {value!r}; use YYYY-MM-DD.", "La fecha no es válida."
        ) from None
