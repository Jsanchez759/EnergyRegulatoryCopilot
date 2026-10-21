import asyncio
import os

import pytest
from fastapi.testclient import TestClient
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent

from app.api.dependencies import get_agent_service
from app.api.v1.endpoints import public
from app.core.config import settings
from tests.conftest import PASSWORD
from tests.test_conversations import create_conversation


def waitlist_signup(app, **extra) -> tuple[TestClient, str]:
    visitor = TestClient(app)
    email = f"wait-{os.urandom(4).hex()}@example.com"
    response = visitor.post(
        "/api/v1/auth/register",
        json={"display_name": "Analista", "email": email, "password": PASSWORD, **extra},
    )
    assert response.status_code == 201, response.text
    return visitor, email


@pytest.fixture
def admin_with_waitlist(make_admin, monkeypatch) -> TestClient:
    admin = make_admin()
    monkeypatch.setattr(settings, "REGISTRATION_MODE", "waitlist")
    return admin


def find_user(admin: TestClient, email: str) -> dict:
    users = admin.get("/api/v1/admin/users", params={"q": email}).json()["users"]
    assert len(users) == 1
    return users[0]


def test_waitlist_signup_gets_no_session_until_approved(admin_with_waitlist) -> None:
    admin = admin_with_waitlist
    visitor, email = waitlist_signup(
        admin.app, company="Comercializadora X", profile="comercializador", use_case="Tarifas"
    )
    assert visitor.get("/api/v1/auth/me").status_code == 401
    login = TestClient(admin.app).post(
        "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
    )
    assert login.status_code == 403
    assert "lista de espera" in login.json()["detail"]

    listed = admin.get("/api/v1/admin/users", params={"status": "waitlist"}).json()
    assert email in {user["email"] for user in listed["users"]}
    user = find_user(admin, email)
    assert (user["company"], user["profile"]) == ("Comercializadora X", "comercializador")

    approved = admin.patch(f"/api/v1/admin/users/{user['id']}", json={"status": "active"})
    assert approved.status_code == 200
    assert approved.json()["approved_at"] is not None
    member = TestClient(admin.app)
    assert (
        member.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD}).status_code
        == 200
    )
    me = member.get("/api/v1/auth/me").json()
    assert (me["status"], me["role"]) == ("active", "user")
    audit = admin.get("/api/v1/admin/audit").json()["entries"]
    assert any(entry["action"] == "approve" and entry["target_email"] == email for entry in audit)


def test_wrong_password_does_not_reveal_waitlist(admin_with_waitlist) -> None:
    _, email = waitlist_signup(admin_with_waitlist.app)
    response = TestClient(admin_with_waitlist.app).post(
        "/api/v1/auth/login", json={"email": email, "password": "wrong-password-123"}
    )
    assert response.status_code == 401


def test_suspending_signs_the_user_out(make_admin, make_user) -> None:
    admin, member = make_admin(), make_user()
    target = member.get("/api/v1/auth/me").json()

    response = admin.patch(f"/api/v1/admin/users/{target['id']}", json={"status": "suspended"})
    assert response.status_code == 200
    assert member.get("/api/v1/auth/me").status_code == 401
    login = TestClient(admin.app).post(
        "/api/v1/auth/login", json={"email": target["email"], "password": PASSWORD}
    )
    assert login.status_code == 403
    assert "suspendida" in login.json()["detail"]


def test_admin_routes_require_the_admin_role(make_user) -> None:
    member = make_user()
    assert member.get("/api/v1/admin/users").status_code == 403
    assert member.get("/api/v1/admin/overview").status_code == 403
    assert TestClient(member.app).get("/api/v1/admin/users").status_code == 401


def test_admin_cannot_lock_out_their_own_account(make_admin) -> None:
    admin = make_admin()
    own_id = admin.get("/api/v1/auth/me").json()["id"]
    assert admin.patch(f"/api/v1/admin/users/{own_id}", json={"role": "user"}).status_code == 400
    assert admin.delete(f"/api/v1/admin/users/{own_id}").status_code == 400
    assert admin.patch(f"/api/v1/admin/users/{own_id}", json={}).status_code == 422


def test_promoting_and_deleting_are_audited(make_admin, make_user) -> None:
    admin, member = make_admin(), make_user()
    target = member.get("/api/v1/auth/me").json()

    assert (
        admin.patch(f"/api/v1/admin/users/{target['id']}", json={"role": "admin"}).json()["role"]
        == "admin"
    )
    assert member.get("/api/v1/admin/users").status_code == 200
    assert admin.delete(f"/api/v1/admin/users/{target['id']}").status_code == 204
    assert admin.get(f"/api/v1/admin/users/{target['id']}").status_code == 404
    actions = [
        entry["action"]
        for entry in admin.get("/api/v1/admin/audit").json()["entries"]
        if entry["target_user_id"] == target["id"]
    ]
    assert actions[:2] == ["delete", "promote"]


class MeteredAgent:
    """Emits the instrumentation event a real model call would, from a child task."""

    async def stream_answer(self, question, history, user_id, **kwargs):
        async def model_call() -> None:
            get_dispatcher().event(
                LLMChatEndEvent(
                    messages=[],
                    response=ChatResponse(
                        message=ChatMessage(role="assistant", content="ok"),
                        raw={
                            "usage": {
                                "prompt_tokens": 1200,
                                "completion_tokens": 300,
                                "cost": 0.0042,
                            }
                        },
                    ),
                )
            )

        await asyncio.create_task(model_call())
        await asyncio.create_task(model_call())
        yield {"answer": "Respuesta"}
        yield {"done": True}


def test_assistant_runs_are_metered_per_user(make_admin, make_user) -> None:
    admin, member = make_admin(), make_user()
    member_id = member.get("/api/v1/auth/me").json()["id"]
    conversation_id = create_conversation(member)
    member.app.dependency_overrides[get_agent_service] = MeteredAgent
    try:
        response = member.post(
            "/api/v1/agent/stream",
            json={"conversation_id": conversation_id, "question": "¿Qué es la CREG?"},
        )
    finally:
        member.app.dependency_overrides.clear()
    assert response.status_code == 200

    detail = admin.get(f"/api/v1/admin/users/{member_id}").json()
    assert detail["usage_30d"]["runs"] == 1
    assert detail["usage_30d"]["prompt_tokens"] == 2400
    assert detail["usage_30d"]["cost_usd"] == pytest.approx(0.0084)
    assert detail["conversations"] == 1
    assert sum(day["runs"] for day in detail["daily"]) == 1
    overview = admin.get("/api/v1/admin/overview").json()
    assert overview["usage_30d"]["runs"] >= 1
    assert any(row["name"] == "research" for row in overview["by_feature"])
    assert any(row["id"] == member_id for row in overview["top_users"])


class NoMarket:
    async def overview(self):
        return []


def test_public_overview_needs_no_session(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(client.app.state, "market_service", NoMarket())
    public._cache.clear()
    response = TestClient(client.app).get("/api/v1/public/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["registration_mode"] == "open"
    assert {source["id"] for source in body["sources"]} >= {"creg", "minenergia"}
