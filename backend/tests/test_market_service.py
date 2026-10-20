import asyncio
import gzip
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from app.services import market_service
from app.services.errors import ProviderUnavailableError
from app.services.market_service import MarketDataError, MarketService

FIXTURES = Path(__file__).parent / "fixtures" / "xm"


def fixture(name: str):
    return json.loads(gzip.decompress((FIXTURES / f"{name}.json.gz").read_bytes()))


RESPONSES = {
    "ListadoMetricas": fixture("lists"),
    "ListadoRecursos": fixture("recursos"),
    "ListadoEmbalses": fixture("embalses"),
}


class FakeXM:
    def __init__(self, data: dict[tuple[str, str], object] | None = None) -> None:
        self.data = data or {}
        self.requests: list[tuple[str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        endpoint = request.url.path.strip("/")
        self.requests.append((endpoint, body))
        if endpoint == "lists":
            return httpx.Response(200, json=RESPONSES[body["MetricId"]])
        payload = self.data.get((endpoint, body["MetricId"]))
        if payload is None:
            return httpx.Response(200, json={"Items": []})
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, int):
            return httpx.Response(payload, text="Error: Rango de consulta supera el maximo")
        return httpx.Response(200, json=payload)


def service(fake: FakeXM) -> MarketService:
    return MarketService(transport=httpx.MockTransport(fake.handler))


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    class Today(market_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return market_service.datetime(2026, 10, 20, 12, tzinfo=tz)

    monkeypatch.setattr(market_service, "datetime", Today)
    monkeypatch.setattr(market_service.asyncio, "sleep", _no_sleep)


async def _no_sleep(_seconds):
    return None


def run(coroutine):
    return asyncio.run(coroutine)


def test_catalog_lists_metrics_with_granularity_and_units() -> None:
    catalog = run(service(FakeXM()).catalog())
    by_key = {(item.metric_id, item.entity): item for item in catalog}

    price = by_key[("PrecBolsNaci", "Sistema")]
    assert (price.granularity, price.units) == ("hourly", "COP/kWh")
    assert by_key[("PorcVoluUtilDiar", "Embalse")].granularity == "daily"
    assert by_key[("MC", "Sistema")].granularity == "monthly"
    assert not any(item.metric_id.startswith("Listado") for item in catalog)


def test_find_metrics_matches_spanish_descriptions_without_embeddings() -> None:
    xm = service(FakeXM())
    found = run(xm.find_metrics("precio de bolsa nacional"))
    assert found[0].metric_id.startswith("PrecBolsNaci")
    reservoirs = run(xm.find_metrics("volumen útil embalses"))
    assert any(item.metric_id == "PorcVoluUtilDiar" for item in reservoirs)


def test_hourly_price_becomes_hourly_rows_for_short_ranges_and_daily_otherwise() -> None:
    fake = FakeXM({("hourly", "PrecBolsNaci"): fixture("prec")})
    xm = service(fake)

    two_days = run(xm.series("PrecBolsNaci", date(2026, 10, 1), date(2026, 10, 2)))
    assert two_days.period == "hora"
    assert len(two_days.rows) == 48
    assert two_days.unit == "COP/kWh"
    stats = two_days.stats["Sistema"]
    assert stats["hora_maxima"] == "2,001.67 (2026-10-02 19h)"
    assert stats["dia_maximo"].endswith("(2026-10-02)")
    assert "promedio" in stats

    fake.requests.clear()
    month = run(xm.series("PrecBolsNaci", date(2026, 9, 1), date(2026, 10, 15)))
    assert [body["StartDate"] for _, body in fake.requests] == ["2026-09-01", "2026-10-01"]
    assert month.period == "día"
    assert month.chart()["series"][0]["name"] == "Sistema"


def test_reservoir_fractions_are_reported_as_percentages() -> None:
    xm = service(FakeXM({("daily", "PorcVoluUtilDiar"): fixture("embalse_daily")}))
    series = run(xm.series("PorcVoluUtilDiar", date(2026, 9, 1), date(2026, 9, 30)))

    assert series.unit == "%"
    assert series.rows[0] == ("2026-09-01", {"Sistema": pytest.approx(79.51)})
    assert series.stats["Sistema"]["variacion_inicio_fin"] == "-4.1%"


def test_energy_by_resource_adds_hours_and_names_plants() -> None:
    xm = service(FakeXM({("hourly", "Gene"): fixture("gene_recurso")}))
    series = run(
        xm.series(
            "Gene", date(2026, 10, 1), date(2026, 10, 1), entity="Recurso", codes=["GVIO", "TBST"]
        )
    )

    assert series.unit == "GWh"
    assert set(series.stats) == {"GVIO · GUAVIO", "TBST · TEBSAB CC"}
    guavio = series.stats["GVIO · GUAVIO"]
    assert guavio["total"] == guavio["dia_maximo"].split(" ")[0], "one day: its total is the day"
    assert "hora_maxima" in guavio
    assert series.request["Filter"] == ["GVIO", "TBST"]
    assert series.chart()["query"] == {
        "metric_id": "Gene",
        "entity": "Recurso",
        "start": "2026-10-01",
        "end": "2026-10-01",
        "codes": ["GVIO", "TBST"],
        "group_by_technology": False,
    }


def test_empty_hours_keep_the_others_in_place() -> None:
    payload = {
        "Items": [
            {
                "Date": "2026-10-01",
                "HourlyEntities": [
                    {
                        "Id": "Recurso",
                        "Values": {
                            "code": "SOLAR1",
                            **{f"Hour{hour:02d}": "" for hour in range(1, 25)},
                            "Hour12": "900",
                            "Hour13": "1000",
                        },
                    }
                ],
            }
        ]
    }
    [(_, _, values)] = market_service.parse_observations(payload)
    assert values == [(12, 900.0), (13, 1000.0)]


def test_monthly_metrics_use_long_windows() -> None:
    fake = FakeXM({("monthly", "MC"): fixture("mc_monthly")})
    series = run(service(fake).series("MC", date(2025, 1, 1), date(2026, 9, 30)))

    assert len(fake.requests) == 2, "catalog plus one monthly request"
    assert series.period == "mes"
    assert series.rows[0][0] == "2025-01"


def test_invalid_requests_explain_what_to_do() -> None:
    xm = service(FakeXM({("hourly", "PrecBolsNaci"): 400}))
    with pytest.raises(MarketDataError, match="Similar metrics: metric_id="):
        run(xm.series("generacion_recurso", date(2026, 9, 1), date(2026, 9, 2), entity="Planta"))
    with pytest.raises(MarketDataError, match="Available entities: Agente, Sistema"):
        run(xm.series("DemaReal", date(2026, 9, 1), date(2026, 9, 2), entity="Embalse"))
    with pytest.raises(MarketDataError, match="at most 366"):
        run(xm.series("PrecBolsNaci", date(2024, 1, 1), date(2026, 1, 1)))
    with pytest.raises(MarketDataError, match="at most 31"):
        run(xm.series("Gene", date(2026, 8, 1), date(2026, 9, 30), entity="Recurso"))
    with pytest.raises(MarketDataError, match="XM rejected"):
        run(xm.series("PrecBolsNaci", date(2026, 9, 1), date(2026, 9, 2)))
    with pytest.raises(MarketDataError, match="after the end date"):
        run(xm.series("PrecBolsNaci", date(2026, 11, 1), date(2026, 11, 2)))


def test_unavailable_api_is_a_provider_error_and_results_are_cached() -> None:
    fake = FakeXM({("hourly", "PrecBolsNaci"): httpx.ConnectError("down")})
    with pytest.raises(ProviderUnavailableError):
        run(service(fake).series("PrecBolsNaci", date(2026, 10, 1), date(2026, 10, 2)))

    fake = FakeXM({("hourly", "PrecBolsNaci"): fixture("prec")})
    xm = service(fake)

    async def twice():
        await xm.series("PrecBolsNaci", date(2026, 10, 1), date(2026, 10, 2))
        await xm.series("PrecBolsNaci", date(2026, 10, 1), date(2026, 10, 2))

    run(twice())
    assert sum(endpoint == "hourly" for endpoint, _ in fake.requests) == 1


def test_codes_are_found_by_plant_name() -> None:
    xm = service(FakeXM())
    found = run(xm.find_codes("Recurso", "Ituango"))
    assert [(item.code, item.name) for item in found] == [("PES1", "ITUANGO")]
    assert run(xm.find_codes("Recurso", "Hidroituango"))[0].code == "PES1"
    assert "HIDRAULICA" in found[0].detail
    with pytest.raises(MarketDataError, match="no code list"):
        run(xm.find_codes("Planeta", "x"))


def test_guessed_ids_resolve_to_the_closest_metric_of_the_same_entity() -> None:
    xm = service(FakeXM({("hourly", "Gene"): fixture("gene_recurso")}))
    series = run(
        xm.series(
            "generacion", date(2026, 10, 1), date(2026, 10, 1), entity="Recurso", codes=["GVIO"]
        )
    )
    assert series.metric.metric_id == "Gene"
    assert series.rows
    assert "'generacion' no es un id de XM" in series.summary_text()


def test_a_partially_published_last_day_is_not_summed() -> None:
    def day(date_text, hours):
        return {
            "Date": date_text,
            "HourlyEntities": [
                {
                    "Id": "Sistema",
                    "Values": {
                        "code": "Sistema",
                        **{f"Hour{hour:02d}": "1000" for hour in range(1, hours + 1)},
                    },
                }
            ],
        }

    payload = {"Items": [day("2026-10-16", 24), day("2026-10-17", 24), day("2026-10-18", 6)]}
    xm = service(FakeXM({("hourly", "DemaReal"): payload}))
    series = run(xm.series("DemaReal", date(2026, 10, 16), date(2026, 10, 18)))
    assert [label for label, _ in series.rows] == ["2026-10-16", "2026-10-17"]
