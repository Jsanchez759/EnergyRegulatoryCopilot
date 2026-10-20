import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.api.dependencies import get_agent_service
from app.services.errors import ProviderUnavailableError


class FakeAgentService:
    def __init__(self, events: list[dict[str, object]], error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.calls: list[tuple[str, list, str]] = []

    async def stream_answer(self, question, history, user_id) -> AsyncIterator[dict[str, object]]:
        self.calls.append((question, history, user_id))
        for event in self.events:
            yield event
        if self.error:
            raise self.error


def sse_payloads(body: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("data:").strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


def create_conversation(user: TestClient) -> str:
    response = user.post("/api/v1/conversations", json={})
    assert response.status_code == 201
    return response.json()["id"]


def test_users_cannot_read_each_others_conversations(make_user) -> None:
    owner, intruder = make_user(), make_user()
    conversation_id = create_conversation(owner)

    assert intruder.get(f"/api/v1/conversations/{conversation_id}/messages").status_code == 404
    assert intruder.delete(f"/api/v1/conversations/{conversation_id}").status_code == 404
    assert conversation_id not in {
        item["id"] for item in intruder.get("/api/v1/conversations").json()["conversations"]
    }


def test_stream_persists_question_answer_and_audit(make_user) -> None:
    user = make_user()
    conversation_id = create_conversation(user)
    audit = {"method": "test", "steps": [], "sources": []}
    fake = FakeAgentService(
        [{"delta": "Hola"}, {"answer": "Hola mundo"}, {"audit": audit}, {"done": True}]
    )
    user.app.dependency_overrides[get_agent_service] = lambda: fake
    try:
        response = user.post(
            "/api/v1/agent/stream",
            json={"conversation_id": conversation_id, "question": "¿Qué es la CREG?"},
        )
    finally:
        user.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert {"done": True} in sse_payloads(response.text)
    messages = user.get(f"/api/v1/conversations/{conversation_id}/messages").json()["messages"]
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "¿Qué es la CREG?"),
        ("assistant", "Hola mundo"),
    ]
    assert messages[1]["audit"] == audit
    conversations = user.get("/api/v1/conversations").json()["conversations"]
    assert conversations[0]["title"] == "¿Qué es la CREG?"


def test_stream_reports_and_persists_provider_failure(make_user) -> None:
    user = make_user()
    conversation_id = create_conversation(user)
    fake = FakeAgentService([], error=ProviderUnavailableError("Proveedor caído."))
    user.app.dependency_overrides[get_agent_service] = lambda: fake
    try:
        response = user.post(
            "/api/v1/agent/stream",
            json={"conversation_id": conversation_id, "question": "¿Qué es la CREG?"},
        )
    finally:
        user.app.dependency_overrides.clear()

    payloads = sse_payloads(response.text)
    assert {"error": "Proveedor caído."} in payloads
    messages = user.get(f"/api/v1/conversations/{conversation_id}/messages").json()["messages"]
    assert messages[-1]["content"] == "No se pudo completar la consulta. Proveedor caído."


def test_stream_rejects_foreign_conversation(make_user) -> None:
    owner, intruder = make_user(), make_user()
    conversation_id = create_conversation(owner)
    fake = FakeAgentService([{"done": True}])
    intruder.app.dependency_overrides[get_agent_service] = lambda: fake
    try:
        response = intruder.post(
            "/api/v1/agent/stream",
            json={"conversation_id": conversation_id, "question": "¿Qué es la CREG?"},
        )
    finally:
        intruder.app.dependency_overrides.clear()

    assert response.status_code == 404
    assert fake.calls == []
