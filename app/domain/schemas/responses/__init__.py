"""
Response DTOs — Package
========================
Re-exports all response DTOs for convenient one-line imports.

Usage:
    from app.domain.schemas.responses import (
        UserResponse,
        TokenResponse,
        MessageResponse,
        ErrorResponse,
        SegmentationResponse,
        SegmentationJobSummary,
    )
"""
from app.domain.schemas.responses.auth_responses import (
    ErrorResponse,
    MessageResponse,
    TokenResponse,
    UserResponse,
)
from app.domain.schemas.responses.segmentation_responses import (
    CeilingMaterialType,
    DetectedRegion,
    MaterialEstimate,
    RoofType,
    SegmentationJobSummary,
    SegmentationResponse,
)

__all__ = [
    # Auth
    "UserResponse",
    "TokenResponse",
    "MessageResponse",
    "ErrorResponse",
    # Segmentation
    "CeilingMaterialType",
    "RoofType",
    "DetectedRegion",
    "MaterialEstimate",
    "SegmentationResponse",
    "SegmentationJobSummary",
]
