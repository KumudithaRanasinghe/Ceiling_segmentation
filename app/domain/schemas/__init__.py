"""
Domain Schemas — Top-level Package
=====================================
Unified re-export of all request and response DTOs.

Structure:
  schemas/
    requests/
      auth_requests.py         ← RegisterRequest, LoginRequest, etc.
      segmentation_requests.py ← SegmentationRequest
    responses/
      auth_responses.py        ← UserResponse, TokenResponse, etc.
      segmentation_responses.py← SegmentationResponse, SegmentationJobSummary, etc.

Usage:
    # Import from specific sub-package (recommended for clarity):
    from app.domain.schemas.requests import RegisterRequest
    from app.domain.schemas.responses import UserResponse

    # Or from the top-level (convenience):
    from app.domain.schemas import RegisterRequest, UserResponse
"""
from app.domain.schemas.requests import (
    ChangePasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    SegmentationRequest,
)
from app.domain.schemas.responses import (
    CeilingMaterialType,
    DetectedRegion,
    ErrorResponse,
    MaterialEstimate,
    MessageResponse,
    RoofType,
    SegmentationJobSummary,
    SegmentationResponse,
    TokenResponse,
    UserResponse,
)

# Backward-compatibility alias (TokenPair → TokenResponse)
TokenPair = TokenResponse

__all__ = [
    # Requests
    "RegisterRequest",
    "LoginRequest",
    "RefreshRequest",
    "ChangePasswordRequest",
    "SegmentationRequest",
    # Responses
    "UserResponse",
    "TokenResponse",
    "TokenPair",
    "MessageResponse",
    "ErrorResponse",
    "SegmentationResponse",
    "SegmentationJobSummary",
    "CeilingMaterialType",
    "RoofType",
    "DetectedRegion",
    "MaterialEstimate",
]
