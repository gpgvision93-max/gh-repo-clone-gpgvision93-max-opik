"""Tests for the user profile API."""
import pytest
from fastapi.testclient import TestClient

from user_profile_api.app import app, get_store
from user_profile_api.store import UserStore


@pytest.fixture()
def client():
    """Return a TestClient backed by a fresh in-memory store."""
    store = UserStore()

    # Override the module-level store used by every endpoint
    import user_profile_api.app as app_module

    original = app_module._store
    app_module._store = store
    try:
        yield TestClient(app)
    finally:
        app_module._store = original


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


def test_create_user(client):
    resp = client.post(
        "/users",
        json={"username": "alice", "email": "alice@example.com"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "alice"
    assert data["email"] == "alice@example.com"
    assert data["email_verified"] is False
    assert "user_id" in data


def test_get_user(client):
    user_id = client.post(
        "/users",
        json={"username": "bob", "email": "bob@example.com"},
    ).json()["user_id"]

    resp = client.get(f"/users/{user_id}")
    assert resp.status_code == 200
    assert resp.json()["username"] == "bob"


def test_get_user_not_found(client):
    resp = client.get("/users/nonexistent-id")
    assert resp.status_code == 404


def test_update_user(client):
    user_id = client.post(
        "/users",
        json={"username": "carol", "email": "carol@example.com"},
    ).json()["user_id"]

    resp = client.patch(f"/users/{user_id}", json={"display_name": "Carol", "bio": "Hello!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["display_name"] == "Carol"
    assert data["bio"] == "Hello!"


def test_update_user_not_found(client):
    resp = client.patch("/users/nonexistent-id", json={"bio": "x"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


def test_request_email_verification(client):
    user_id = client.post(
        "/users",
        json={"username": "dave", "email": "dave@example.com"},
    ).json()["user_id"]

    resp = client.post(f"/users/{user_id}/request-email-verification")
    assert resp.status_code == 202
    data = resp.json()
    assert "token" in data
    assert "expires_at" in data


def test_request_email_verification_user_not_found(client):
    resp = client.post("/users/nonexistent-id/request-email-verification")
    assert resp.status_code == 404


def test_verify_email_success(client):
    user_id = client.post(
        "/users",
        json={"username": "eve", "email": "eve@example.com"},
    ).json()["user_id"]

    token = client.post(f"/users/{user_id}/request-email-verification").json()["token"]

    resp = client.post("/verify-email", json={"token": token})
    assert resp.status_code == 200
    assert resp.json()["email_verified"] is True


def test_verify_email_invalid_token(client):
    resp = client.post("/verify-email", json={"token": "bogus-token"})
    assert resp.status_code == 400


def test_verify_email_token_cannot_be_reused(client):
    user_id = client.post(
        "/users",
        json={"username": "frank", "email": "frank@example.com"},
    ).json()["user_id"]

    token = client.post(f"/users/{user_id}/request-email-verification").json()["token"]

    # First use – should succeed
    assert client.post("/verify-email", json={"token": token}).status_code == 200

    # Second use – should fail
    assert client.post("/verify-email", json={"token": token}).status_code == 400


def test_verify_email_expired_token_is_rejected():
    """Token that is already expired must not verify the email."""
    from datetime import datetime, timedelta, timezone

    from user_profile_api.models import EmailVerificationToken
    from user_profile_api.store import UserStore

    store = UserStore()
    user = store.create_user("grace", "grace@example.com")

    # Inject an already-expired token
    expired_token = EmailVerificationToken(
        token="expired-token",
        user_id=user.user_id,
        email=user.email,
        expires_at=datetime.now(tz=timezone.utc) - timedelta(seconds=1),
    )
    store._tokens["expired-token"] = expired_token

    result = store.verify_email("expired-token")
    assert result is None
    assert store.get_user(user.user_id).email_verified is False
