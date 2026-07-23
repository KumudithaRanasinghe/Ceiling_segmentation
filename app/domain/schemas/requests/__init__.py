"""
Request DTOs — Package
========================
Re-exports all request DTOs for convenient one-line imports.

Usage:
    from app.domain.schemas.requests import (
        RegisterRequest,
        LoginRequest,
        RefreshRequest,
        ChangePasswordRequest,
        SegmentationRequest,
    )
"""
from app.domain.schemas.requests.auth_requests import (
    ChangePasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
)
from app.domain.schemas.requests.segmentation_requests import SegmentationRequest

__all__ = [
    "RegisterRequest",
    "LoginRequest",
    "RefreshRequest",
    "ChangePasswordRequest",
    "SegmentationRequest",
]
