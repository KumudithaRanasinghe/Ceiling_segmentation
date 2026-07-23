"""
Auth Response DTOs
==================
Pydantic models for /auth endpoint responses.

Rules:
  - NEVER expose hashed_password or refresh_token_hash.
  - All responses are from_attributes=True (ORM → DTO mapping).
  - Roles are stored as a JSON string in the DB; parse_roles handles
    the deserialization transparently so callers always receive a list.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── User profile ──────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    """
    Public user profile — safe to send to any authenticated client.
    Omits all credential fields.
    """
    id: str
    email: str
    username: str
    full_name: Optional[str]
    roles: List[str] = Field(description="Assigned roles, e.g. ['user', 'admin'].")
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    @field_validator("roles", mode="before")
    @classmethod
    def parse_roles(cls, v):
        """Deserialize roles from JSON string (as stored in DB) or pass through list."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    model_config = {"from_attributes": True}


# ── Token pair ────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    """
    Returned by /auth/login and /auth/refresh.

    access_token:  Short-lived JWT (15 min). Use in Authorization: Bearer header.
    refresh_token: Long-lived JWT (7 days).  Use only with /auth/refresh.
    token_type:    Always "bearer".
    expires_in:    Access token lifetime in seconds (for client-side timer).
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token lifetime in seconds.")


# ── Generic message ───────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    """Generic acknowledgement response (logout, password change, etc.)."""
    message: str


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Structured error body for 4xx/5xx responses."""
    error_code: str
    message: str
    details: Optional[dict] = None
