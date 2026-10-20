import logging

from fastapi.testclient import TestClient

from tests.conftest import PASSWORD


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_sets_session_cookie_and_profile(make_user) -> None:
    user = make_user()
    response = user.get("/api/v1/auth/me")
    assert response.status_code == 200
    assert response.json()["display_name"] == "Analista"


def test_duplicate_email_is_rejected_case_insensitively(make_user) -> None:
    user = make_user()
    email = user.get("/api/v1/auth/me").json()["email"]
    response = TestClient(user.app).post(
        "/api/v1/auth/register",
        json={"display_name": "Otra", "email": email.upper(), "password": PASSWORD},
    )
    assert response.status_code == 409


def test_login_with_wrong_password_fails(make_user) -> None:
    email = make_user().get("/api/v1/auth/me").json()["email"]
    response = TestClient(make_user().app).post(
        "/api/v1/auth/login", json={"email": email, "password": "wrong-password-123"}
    )
    assert response.status_code == 401


def test_logout_revokes_session(make_user) -> None:
    user = make_user()
    token = user.cookies.get("enercol_session")
    assert user.post("/api/v1/auth/logout").status_code == 204

    stale = TestClient(user.app, cookies={"enercol_session": token})
    assert stale.get("/api/v1/auth/me").status_code == 401


def test_password_change_revokes_other_sessions(make_user) -> None:
    user = make_user()
    email = user.get("/api/v1/auth/me").json()["email"]
    other_device = TestClient(user.app)
    assert (
        other_device.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        ).status_code
        == 200
    )

    response = user.post(
        "/api/v1/auth/password",
        json={"current_password": PASSWORD, "new_password": "another-long-password"},
    )
    assert response.status_code == 204
    assert user.get("/api/v1/auth/me").status_code == 200
    assert other_device.get("/api/v1/auth/me").status_code == 401


def test_protected_routes_require_session(client: TestClient) -> None:
    anonymous = TestClient(client.app)
    assert anonymous.get("/api/v1/auth/me").status_code == 401
    assert anonymous.get("/api/v1/conversations").status_code == 401
    assert anonymous.get("/api/v1/sources").status_code == 401


def test_untrusted_origin_cannot_mutate(make_user) -> None:
    user = make_user()
    response = user.post(
        "/api/v1/conversations", json={}, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403


def test_startup_migrations_keep_app_logging_enabled(client: TestClient) -> None:
    assert logging.getLogger("app").disabled is False
