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
from app.domain.schemas.responses.admin_responses import (
    AdminJobListItem,
    AdminJobListResponse,
    AdminSummaryKPIResponse,
    AdminUserListItem,
    AdminUserListResponse,
    AuditLogItem,
    AuditLogListResponse,
    BucketItem,
    DistributionResponse,
    MaterialAnalyticsResponse,
    MaterialPricingConfigResponse,
    MaterialStatItem,
    ModelPerformanceItem,
    ModelPerformanceResponse,
    SystemHealthResponse,
    TimeseriesAnalyticsResponse,
    TimeseriesPoint,
)
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
    # Admin
    "AdminSummaryKPIResponse",
    "TimeseriesPoint",
    "TimeseriesAnalyticsResponse",
    "BucketItem",
    "DistributionResponse",
    "MaterialStatItem",
    "MaterialAnalyticsResponse",
    "ModelPerformanceItem",
    "ModelPerformanceResponse",
    "AdminUserListItem",
    "AdminUserListResponse",
    "AdminJobListItem",
    "AdminJobListResponse",
    "AuditLogItem",
    "AuditLogListResponse",
    "SystemHealthResponse",
    "MaterialPricingConfigResponse",
]
