"""FastAPI application exposing the user profile API."""
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, EmailStr

from .store import UserStore

app = FastAPI(title="User Profile API", version="1.0.0")

# Module-level store (replaced during testing via dependency injection)
_store = UserStore()


def get_store() -> UserStore:
    return _store


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CreateUserRequest(BaseModel):
    username: str
    email: EmailStr
    display_name: str = ""
    bio: str = ""


class UpdateUserRequest(BaseModel):
    display_name: Optional[str] = None
    bio: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUserRequest) -> Dict[str, Any]:
    """Create a new user profile."""
    store = get_store()
    user = store.create_user(
        username=body.username,
        email=body.email,
        display_name=body.display_name,
        bio=body.bio,
    )
    return asdict(user)


@app.get("/users/{user_id}")
def get_user(user_id: str) -> Dict[str, Any]:
    """Retrieve a user profile by ID."""
    store = get_store()
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return asdict(user)


@app.patch("/users/{user_id}")
def update_user(user_id: str, body: UpdateUserRequest) -> Dict[str, Any]:
    """Update mutable fields on a user profile."""
    store = get_store()
    user = store.update_user(user_id, display_name=body.display_name, bio=body.bio)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return asdict(user)


@app.post("/users/{user_id}/request-email-verification", status_code=status.HTTP_202_ACCEPTED)
def request_email_verification(user_id: str) -> Dict[str, Any]:
    """
    Generate an email-verification token for the user.

    In a production system the token would be sent via email.  Here the token
    is returned in the response body so that the caller can pass it straight to
    ``POST /verify-email``.
    """
    store = get_store()
    token = store.create_verification_token(user_id)
    if token is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {
        "message": "Verification email sent",
        "token": token.token,
        "expires_at": token.expires_at.isoformat(),
    }


class VerifyEmailRequest(BaseModel):
    token: str


@app.post("/verify-email")
def verify_email(body: VerifyEmailRequest) -> Dict[str, Any]:
    """
    Confirm a user's email address using a verification token.

    Returns the updated user profile on success.
    """
    store = get_store()
    user = store.verify_email(body.token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or already-used token",
        )
    return asdict(user)
