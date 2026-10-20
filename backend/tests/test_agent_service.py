import asyncio
from types import SimpleNamespace

import pytest

from app.schemas.agent import EvidenceSource
from app.services import agent_service as agent_module
from app.services.agent_service import AgentService


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


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def achat(self, messages):
        self.prompts.append(messages[-1].content)
        return SimpleNamespace(message=SimpleNamespace(content="Respuesta con evidencia [D1]."))


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


def collect(agent: AgentService) -> list[dict[str, object]]:
    async def run() -> list[dict[str, object]]:
        events = [event async for event in agent.stream_answer("¿Plazo de conexión?", [], "u1")]
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
