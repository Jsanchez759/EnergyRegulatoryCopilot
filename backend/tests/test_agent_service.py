import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.workflow.errors import WorkflowTimeoutError

from app.schemas.agent import ChatMessage, EvidenceSource
from app.services import agent_service as agent_module
from app.services.agent_service import AgentService
from app.services.errors import ProviderUnavailableError


class SilentHandler:
    """Agent run that finishes without tool calls or text, like some routed free models."""

    async def stream_events(self):
        return
        yield

    def __await__(self):
        async def result():
            return SimpleNamespace(response=SimpleNamespace(content=""), raw=None)

        return result().__await__()


class SilentAgent:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, **_kwargs) -> SilentHandler:
        return SilentHandler()


class TimedOutHandler(SilentHandler):
    def __await__(self):
        async def result():
            raise WorkflowTimeoutError("Operation timed out after 90 seconds")

        return result().__await__()


class FakeLLM:
    def __init__(self, reply: str = "Respuesta con evidencia [D1].") -> None:
        self.reply = reply
        self.prompts: list[str] = []

        self.queued: list[str] = []

    async def achat(self, messages):
        self.prompts.append(messages[-1].content)
        content = self.queued.pop(0) if self.queued else self.reply
        return SimpleNamespace(message=SimpleNamespace(content=content))


class FakeDocuments:
    async def search(self, query: str, user_id: str, top_k: int = 6):
        return [
            EvidenceSource(
                id="D1",
                kind="document",
                title="creg-015-2018.pdf",
                page=4,
                snippet="El operador de red responde en nueve días hábiles.",
            )
        ]


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> tuple[AgentService, FakeLLM]:
    monkeypatch.setattr(agent_module, "FunctionAgent", SilentAgent)
    agent = AgentService(FakeDocuments())
    llm = FakeLLM()
    agent._llm = llm
    return agent, llm


def collect(agent: AgentService, question="¿Plazo de conexión?", history=None):
    async def run() -> list[dict[str, object]]:
        events = [event async for event in agent.stream_answer(question, history or [], "u1")]
        await agent.close()
        return events

    return asyncio.run(run())


def test_agent_that_skips_tools_still_answers_from_retrieved_evidence(service) -> None:
    agent, llm = service
    events = collect(agent)

    searched = [
        event["activity"]["tool"]
        for event in events
        if "activity" in event and event["activity"]["status"] == "complete"
    ]
    assert searched == ["search_uploaded_documents", "search_web"]
    assert {"answer": "Respuesta con evidencia [D1]."} in events
    assert "[D1] creg-015-2018.pdf" in llm.prompts[0]

    audit = next(event["audit"] for event in events if "audit" in event)
    assert [source["id"] for source in audit["sources"]] == ["D1"]
    assert events[-1] == {"done": True}


def test_inline_reasoning_tags_are_removed_from_the_answer(service) -> None:
    agent, llm = service
    llm.reply = "<thinking>Debo resumir las fuentes.</thinking>\n\nRespuesta limpia [D1]."
    events = collect(agent)
    assert {"answer": "Respuesta limpia [D1]."} in events


def test_agent_timeout_is_reported_as_provider_failure(service, monkeypatch) -> None:
    agent, _ = service
    monkeypatch.setattr(SilentAgent, "run", lambda self, **_kwargs: TimedOutHandler())
    with pytest.raises(ProviderUnavailableError):
        collect(agent)


def test_domain_filtered_web_search_omits_country() -> None:
    filtered = AgentService._web_search_payload("CREG 174", ["creg.gov.co"])
    assert filtered["include_domains"] == ["creg.gov.co"]
    assert "country" not in filtered
    assert AgentService._web_search_payload("CREG 174", None)["country"] == "Colombia"


class FakeTavily:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def post(self, url, headers, json):
        self.payloads.append(json)
        results = (
            []
            if json.get("include_domains")
            else [
                {
                    "title": "Ley 1715 de 2014",
                    "url": "https://www.funcionpublica.gov.co/ley-1715",
                    "content": "Incentivos tributarios para FNCE.",
                    "score": 0.9,
                }
            ]
        )
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"results": results})

    async def aclose(self) -> None:
        pass


def test_empty_domain_search_widens_to_open_web(service, monkeypatch) -> None:
    agent, _ = service
    tavily = FakeTavily()
    agent._web_client = tavily
    monkeypatch.setattr(agent_module.settings, "TAVILY_API_KEY", "test-key")

    class DomainSearchingAgent:
        def __init__(self, tools, **_kwargs) -> None:
            self.search_web = tools[1]

        def run(self, **_kwargs):
            search_web = self.search_web

            class Handler(SilentHandler):
                def __await__(self):
                    async def result():
                        await search_web("Ley 1715 beneficios", domains=["energia.gov.co"])
                        return SimpleNamespace(
                            response=SimpleNamespace(content="Hay incentivos tributarios [W1]."),
                            raw=None,
                        )

                    return result().__await__()

            return Handler()

    monkeypatch.setattr(agent_module, "FunctionAgent", DomainSearchingAgent)
    events = collect(agent)

    assert [payload.get("include_domains") for payload in tavily.payloads] == [
        ["energia.gov.co"],
        None,
    ]
    audit = next(event["audit"] for event in events if "audit" in event)
    assert audit["steps"][0]["summary"].startswith("No results on energia.gov.co.")
    assert [source["id"] for source in audit["sources"]] == ["W1"]


def test_follow_up_is_rewritten_into_a_standalone_search(service) -> None:
    agent, llm = service
    llm.queued = ["Resolución CREG 024 de 2015 autogeneración a gran escala"]
    history = [
        ChatMessage(role="user", content="¿Qué regula la autogeneración a gran escala?"),
        ChatMessage(role="assistant", content="La Resolución CREG 024 de 2015 [W1]."),
    ]
    events = collect(agent, question="dame un resumen más explicativo", history=history)

    queries = [
        event["activity"]["query"]
        for event in events
        if "activity" in event and event["activity"]["status"] == "complete"
    ]
    assert queries == ["Resolución CREG 024 de 2015 autogeneración a gran escala"] * 2
    assert "La Resolución CREG 024 de 2015 [W1]." in llm.prompts[-1]


class FakeCorpus:
    async def search(self, query: str, limit: int = 8):
        from app.services.corpus_service import CorpusHit

        return [
            CorpusHit(
                norm_id="n1",
                norm_key="creg:resolucion:174:2021",
                source="creg",
                norm_label="Resolución CREG 174 de 2021",
                title="Por la cual se regulan las actividades de autogeneración a pequeña escala",
                doc_type="resolucion",
                status="vigente",
                year=2021,
                issued_on=None,
                url="https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0174_2021.htm",
                article_id="a1",
                article_label="Artículo 1",
                content="Regular aspectos técnicos, operativos y comerciales.",
                score=0.03,
            )
        ]


def test_corpus_is_searched_first_and_cited_with_status(monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "FunctionAgent", SilentAgent)
    agent = AgentService(FakeDocuments(), FakeCorpus())
    llm = FakeLLM("La Resolución CREG 174 de 2021 regula la AGPE [R1].")
    agent._llm = llm
    events = collect(agent)

    searched = [
        event["activity"]["tool"]
        for event in events
        if "activity" in event and event["activity"]["status"] == "complete"
    ]
    assert searched[0] == "search_regulatory_corpus"
    assert "[R1] Resolución CREG 174 de 2021, Artículo 1" in llm.prompts[-1]
    audit = next(event["audit"] for event in events if "audit" in event)
    regulation = next(source for source in audit["sources"] if source["id"] == "R1")
    assert regulation["kind"] == "regulation"
    assert regulation["status"] == "vigente"


def test_market_tools_cite_xm_series_with_chart_data(monkeypatch) -> None:
    import httpx

    from app.services import market_service
    from app.services.market_service import MarketService
    from tests.test_market_service import FakeXM
    from tests.test_market_service import fixture as xm_fixture

    class Today(market_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return market_service.datetime(2026, 10, 20, 12, tzinfo=tz)

    monkeypatch.setattr(market_service, "datetime", Today)
    fake = FakeXM({("hourly", "PrecBolsNaci"): xm_fixture("prec")})
    market = MarketService(transport=httpx.MockTransport(fake.handler))
    outputs: list[str] = []

    class MarketAgent:
        def __init__(self, tools, system_prompt, **_kwargs) -> None:
            self.tools = {tool.__name__: tool for tool in tools}
            self.system_prompt = system_prompt

        def run(self, **_kwargs):
            tools = self.tools

            class Handler(SilentHandler):
                def __await__(self):
                    async def result():
                        outputs.append(await tools["find_market_metrics"]("precio de bolsa"))
                        outputs.append(
                            await tools["get_market_data"](
                                "PrecBolsNaci", "2026-10-01", "2026-10-02"
                            )
                        )
                        outputs.append(
                            await tools["get_market_data"](
                                "PrecBolsNaci", "2026-10-01", "2026-10-02", entity="Embalse"
                            )
                        )
                        outputs.append(
                            await tools["get_market_data"](
                                "PrecBolsNaci entity=Sistema unit=COP/kWh granularity=hourly",
                                "2026-10-01",
                                "2026-10-02",
                            )
                        )
                        return SimpleNamespace(
                            response=SimpleNamespace(content="El precio subió [M1]."), raw=None
                        )

                    return result().__await__()

            return Handler()

    monkeypatch.setattr(agent_module, "FunctionAgent", MarketAgent)
    agent = AgentService(FakeDocuments(), market_service=market)
    agent._llm = FakeLLM()
    events = collect(agent, question="¿Cómo estuvo el precio de bolsa el 1 y 2 de octubre?")

    assert "metric_id=PrecBolsNaci entity=Sistema" in outputs[0]
    assert outputs[1].startswith("[M1] XM · Precio Bolsa Nacional por Sistema")
    assert "Unknown metric 'PrecBolsNaci' for entity 'Embalse'" in outputs[2]
    assert outputs[3].startswith("[M1] XM · Precio Bolsa Nacional"), (
        "pasted catalog lines still work"
    )
    audit = next(event["audit"] for event in events if "audit" in event)
    market_source = next(source for source in audit["sources"] if source["id"] == "M1")
    assert market_source["kind"] == "market"
    assert market_source["chart"]["unit"] == "COP/kWh"
    assert len(market_source["chart"]["x"]) == 48
    assert "XM market data" in audit["method"]
