"""
Auth Pydantic Schemas
======================
Request bodies and response models for the /auth endpoints.
Passwords are validated for minimum strength at the schema level.
Response schemas NEVER include hashed_password or refresh_token_hash.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# ─────────────────────────────── Request schemas ──────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Valid email address.")
    username: str = Field(..., min_length=3, max_length=64)
    full_name: Optional[str] = Field(None, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    password_confirm: str = Field(..., description="Must match password.")

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Username may only contain letters, digits, _ or -.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character.")
        return v

    @model_validator(mode="after")
    def passwords_match(self) -> "RegisterRequest":
        if self.password != self.password_confirm:
            raise ValueError("password and password_confirm do not match.")
        return self

    model_config = {"extra": "forbid"}


class LoginRequest(BaseModel):
    username_or_email: str = Field(..., description="Either username or email address.")
    password: str = Field(..., max_length=128)

    model_config = {"extra": "forbid"}


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="The refresh token obtained at login.")

    model_config = {"extra": "forbid"}


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    new_password_confirm: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character.")
        return v

    @model_validator(mode="after")
    def passwords_match(self) -> "ChangePasswordRequest":
        if self.new_password != self.new_password_confirm:
            raise ValueError("new_password and new_password_confirm do not match.")
        return self

    model_config = {"extra": "forbid"}


# ─────────────────────────────── Response schemas ─────────────────────────────

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str]
    roles: List[str]
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    @field_validator("roles", mode="before")
    @classmethod
    def parse_roles(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    model_config = {"from_attributes": True}


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token lifetime in seconds.")


class MessageResponse(BaseModel):
    message: str
