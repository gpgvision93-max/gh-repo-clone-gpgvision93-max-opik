"""In-memory storage and business logic for the user profile API."""
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from .models import EmailVerificationToken, UserProfile

# Token expiry window (24 hours)
TOKEN_EXPIRY_HOURS = 24


class UserStore:
    """Simple in-memory store for user profiles and verification tokens."""

    def __init__(self) -> None:
        self._users: Dict[str, UserProfile] = {}
        self._tokens: Dict[str, EmailVerificationToken] = {}

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    def create_user(
        self,
        username: str,
        email: str,
        display_name: str = "",
        bio: str = "",
    ) -> UserProfile:
        """Create and store a new user profile."""
        user = UserProfile(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            display_name=display_name,
            bio=bio,
        )
        self._users[user.user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Return the user profile for *user_id*, or ``None``."""
        return self._users.get(user_id)

    def update_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        bio: Optional[str] = None,
    ) -> Optional[UserProfile]:
        """Update mutable fields on an existing user profile."""
        user = self._users.get(user_id)
        if user is None:
            return None
        if display_name is not None:
            user.display_name = display_name
        if bio is not None:
            user.bio = bio
        user.updated_at = datetime.now(tz=timezone.utc)
        return user

    # ------------------------------------------------------------------
    # Email verification
    # ------------------------------------------------------------------

    def create_verification_token(self, user_id: str) -> Optional[EmailVerificationToken]:
        """Generate a new email-verification token for *user_id*."""
        user = self._users.get(user_id)
        if user is None:
            return None
        token_value = secrets.token_urlsafe(32)
        token = EmailVerificationToken(
            token=token_value,
            user_id=user_id,
            email=user.email,
            expires_at=datetime.now(tz=timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS),
        )
        self._tokens[token_value] = token
        return token

    def verify_email(self, token_value: str) -> Optional[UserProfile]:
        """
        Mark a user's email as verified using *token_value*.

        Returns the updated ``UserProfile`` on success, or ``None`` when the
        token is unknown, already used, or expired.
        """
        token = self._tokens.get(token_value)
        if token is None or token.used:
            return None
        if datetime.now(tz=timezone.utc) > token.expires_at:
            return None

        token.used = True
        user = self._users.get(token.user_id)
        if user is None:
            return None
        user.email_verified = True
        user.updated_at = datetime.now(tz=timezone.utc)
        return user
