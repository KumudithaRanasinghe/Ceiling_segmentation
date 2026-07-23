"""
Auth Request DTOs
==================
Pydantic models that represent incoming payloads for /auth endpoints.

Rules:
  - extra="forbid": rejects unknown fields (no silent data leakage).
  - All string fields are explicitly sized.
  - Password confirmation is UI responsibility — not repeated here.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


def _validate_password_strength(v: str) -> str:
    """Shared strength check — reused by Register and ChangePassword."""
    rules = [
        (r"[A-Z]",                    "at least one uppercase letter"),
        (r"[a-z]",                    "at least one lowercase letter"),
        (r"\d",                       "at least one digit"),
        (r'[!@#$%^&*(),.?":{}|<>]',  "at least one special character"),
    ]
    for pattern, msg in rules:
        if not re.search(pattern, v):
            raise ValueError(f"Password must contain {msg}.")
    return v


# ── Register ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """
    New user registration payload.

    Password confirmation is intentionally omitted — the UI is responsible
    for matching the two password fields before submitting the request.
    The backend only validates strength, not equality.
    """
    email: EmailStr = Field(..., description="Valid email address.")
    username: str = Field(..., min_length=3, max_length=64)
    full_name: Optional[str] = Field(None, max_length=255)
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Min 8 chars. Must include upper, lower, digit, and special character.",
    )

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Username may only contain letters, digits, _ or -.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)

    model_config = {"extra": "forbid"}


# ── Login ─────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    """Credentials for /auth/login."""
    username_or_email: str = Field(
        ..., description="Either the username or the registered email address."
    )
    password: str = Field(..., max_length=128)

    model_config = {"extra": "forbid"}


# ── Refresh token ─────────────────────────────────────────────────────────────

class RefreshRequest(BaseModel):
    """Refresh token rotation payload for /auth/refresh."""
    refresh_token: str = Field(
        ..., description="The refresh_token received from /auth/login or a previous /auth/refresh."
    )

    model_config = {"extra": "forbid"}


# ── Change password ───────────────────────────────────────────────────────────

class ChangePasswordRequest(BaseModel):
    """
    Password change payload for /auth/me/password.

    New-password confirmation is omitted — the UI handles field matching
    before submitting the request to the API.
    """
    current_password: str = Field(..., description="Your existing password.")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password — must satisfy strength requirements.",
    )

    @field_validator("new_password")
    @classmethod
    def new_password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)

    model_config = {"extra": "forbid"}
