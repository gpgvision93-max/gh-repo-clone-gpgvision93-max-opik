"""Data models for the user profile API."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class UserProfile:
    """Represents a user profile with email verification state."""

    user_id: str
    username: str
    email: str
    display_name: str = ""
    bio: str = ""
    email_verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class EmailVerificationToken:
    """Token issued to verify a user's email address."""

    token: str
    user_id: str
    email: str
    expires_at: datetime
    used: bool = False
