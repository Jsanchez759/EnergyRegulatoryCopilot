import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.services import agent_service as agent_module
from app.services import market_service
from app.services.market_service import MarketService
from tests.test_agent_service import FakeLLM, SilentHandler
from tests.test_market_service import FakeXM, fixture

DATA = {
    ("hourly", "PrecBolsNaci"): fixture("prec"),
    ("daily", "PorcVoluUtilDiar"): fixture("embalse_daily"),
}


@pytest.fixture
def market(client: TestClient, monkeypatch):
    class Today(market_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return market_service.datetime(2026, 10, 2, 12, tzinfo=tz)

    monkeypatch.setattr(market_service, "datetime", Today)
    fake = FakeXM(DATA)
    service = MarketService(transport=httpx.MockTransport(fake.handler))
    monkeypatch.setattr(client.app.state, "market_service", service)
    monkeypatch.setattr(client.app.state.agent_service, "_market", service)
    return fake


def test_market_requires_authentication(client: TestClient) -> None:
    anonymous = TestClient(client.app)
    assert anonymous.get("/api/v1/market/metrics").status_code == 401


def test_catalog_and_search(make_user, market) -> None:
    user = make_user()
    catalog = user.get("/api/v1/market/metrics").json()["metrics"]
    assert len(catalog) > 150
    found = user.get("/api/v1/market/metrics", params={"q": "precio de bolsa"}).json()["metrics"]
    assert found[0]["metric_id"] == "PrecBolsNaci"
    assert found[0]["recommended"]


def test_series_is_ready_to_chart(make_user, market) -> None:
    user = make_user()
    response = user.get(
        "/api/v1/market/series",
        params={"metric_id": "PrecBolsNaci", "start": "2026-10-01", "end": "2026-10-02"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["unit"] == "COP/kWh"
    assert len(body["chart"]["x"]) == 48
    assert body["chart"]["series"][0]["name"] == "Sistema"
    assert body["metric"]["granularity"] == "hourly"


def test_series_errors_are_explained_in_spanish(make_user, market) -> None:
    user = make_user()
    response = user.get(
        "/api/v1/market/series",
        params={"metric_id": "PrecBolsNaci", "start": "2024-01-01", "end": "2026-10-01"},
    )
    assert response.status_code == 422
    assert response.json()["detail"].startswith("El rango tiene")


def test_codes_by_name(make_user, market) -> None:
    user = make_user()
    codes = user.get("/api/v1/market/codes", params={"entity": "Recurso", "q": "Hidroituango"})
    assert codes.json()["codes"][0]["code"] == "PES1"
    bad = user.get("/api/v1/market/codes", params={"entity": "Planeta", "q": "xx"})
    assert bad.status_code == 422


def test_overview_reports_latest_values_and_weekly_change(make_user, market) -> None:
    indicators = make_user().get("/api/v1/market/overview").json()["indicators"]
    by_key = {item["key"]: item for item in indicators}

    reservoirs = by_key["embalses"]
    assert (reservoirs["unit"], reservoirs["change_unit"]) == ("%", "pp")
    assert reservoirs["latest_date"] == "2026-09-30"
    assert reservoirs["latest_value"] == pytest.approx(76.236)
    assert len(reservoirs["spark"]["x"]) == 30
    assert "demanda" not in by_key, "indicators without data are left out"


def test_assistant_streams_answer_and_chart_sources(client, make_user, market, monkeypatch) -> None:
    contexts: list[str] = []

    class MarketAgent:
        def __init__(self, tools, system_prompt, **_kwargs) -> None:
            self.tools = {tool.__name__: tool for tool in tools}
            assert "interactive chart canvas" in system_prompt
            assert "search_uploaded_documents" not in self.tools

        def run(self, user_msg, **_kwargs):
            contexts.append(user_msg)
            get_market_data = self.tools["get_market_data"]

            class Handler(SilentHandler):
                def __await__(self):
                    async def result():
                        await get_market_data("PrecBolsNaci", "2026-10-01", "2026-10-02")
                        return type(
                            "Result",
                            (),
                            {"response": type("R", (), {"content": "Subió [M1]."})(), "raw": None},
                        )()

                    return result().__await__()

            return Handler()

    monkeypatch.setattr(agent_module, "FunctionAgent", MarketAgent)
    monkeypatch.setattr(client.app.state.agent_service, "_llm", FakeLLM())
    user = make_user()
    with user.stream(
        "POST",
        "/api/v1/market/assistant",
        json={"question": "¿Cómo va el precio?", "canvas": ["Embalses: 76 % el 30 de septiembre"]},
    ) as response:
        events = [
            json.loads(line[5:]) for line in response.iter_lines() if line.startswith("data:")
        ]

    assert "Charts on the user's canvas:\nEmbalses: 76 %" in contexts[0]
    assert next(event["answer"] for event in events if "answer" in event) == "Subió [M1]."
    audit = next(event["audit"] for event in events if "audit" in event)
    assert audit["sources"][0]["chart"]["unit"] == "COP/kWh"
