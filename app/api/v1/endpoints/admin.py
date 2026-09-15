"""
Admin API Endpoints
====================
All administrative endpoints protected with Role-Based Access Control (RBAC: require_admin).
Provides:
  - Analytics & Visualizations (KPI summaries, timeseries, histograms, material trends, model stats)
  - User & Access Governance (IAM directory, role assignment, activation, password reset)
  - Global Job Management (Job explorer across users, inspection, deletion, TTL batch purge)
  - Material Pricing Catalog Management
  - System Health & Diagnostic Telemetry
  - Immutable Audit Trail Inspection
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.core.dependencies import require_admin
from app.domain.models.user import User
from app.domain.repositories.job_repository import JobRepository
from app.domain.schemas.requests.admin_requests import (
    AdminResetPasswordRequest,
    AdminUpdateUserStatusRequest,
    MaterialPricingUpdateRequest,
    PurgeJobsRequest,
    UpdateUserRoleRequest,
)
from app.domain.schemas.responses.admin_responses import (
    AdminJobListResponse,
    AdminSummaryKPIResponse,
    AdminUserListResponse,
    AuditLogListResponse,
    DistributionResponse,
    MaterialAnalyticsResponse,
    MaterialPricingConfigResponse,
    ModelPerformanceResponse,
    SystemHealthResponse,
    TimeseriesAnalyticsResponse,
)
from app.domain.schemas.responses.auth_responses import MessageResponse, UserResponse
from app.domain.schemas.responses.segmentation_responses import SegmentationResponse
from app.infrastructure.database.session import get_db
from app.services.admin_analytics_service import AdminAnalyticsService
from app.services.admin_management_service import AdminManagementService

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_client_ip(request: Request) -> Optional[str]:
    """Helper to retrieve client IP address (supporting reverse proxies)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Analytics & Visualizations (Dashboard Charts)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/dashboard/summary",
    response_model=AdminSummaryKPIResponse,
    summary="Get top-level KPI metrics for executive overview",
)
def get_dashboard_summary(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AdminSummaryKPIResponse:
    """Returns aggregated executive metrics (total jobs, success rate, DAU/MAU, total area, storage)."""
    svc = AdminAnalyticsService(db)
    return svc.get_summary_kpis()


@router.get(
    "/dashboard/timeseries",
    response_model=TimeseriesAnalyticsResponse,
    summary="Get timeseries trend data with gap-filling and smoothed EMA curve",
)
def get_timeseries_analytics(
    days: int = Query(14, ge=1, le=90, description="Number of past days to include."),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> TimeseriesAnalyticsResponse:
    """Returns chronological daily job volume, success/failure counts, and smoothed moving averages."""
    svc = AdminAnalyticsService(db)
    return svc.get_timeseries_analytics(days=days)


@router.get(
    "/dashboard/latency-distribution",
    response_model=DistributionResponse,
    summary="Get inference latency percentiles (p50/p90/p95/p99) and histogram buckets",
)
def get_latency_distribution(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> DistributionResponse:
    """Computes exact percentiles and binned histogram of AI inference durations."""
    svc = AdminAnalyticsService(db)
    return svc.get_latency_distribution()


@router.get(
    "/dashboard/area-distribution",
    response_model=DistributionResponse,
    summary="Get ceiling square meter (m²) area distribution and percentiles",
)
def get_area_distribution(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> DistributionResponse:
    """Returns histogram buckets and percentiles for ceiling surface dimensions."""
    svc = AdminAnalyticsService(db)
    return svc.get_area_distribution()


@router.get(
    "/dashboard/materials",
    response_model=MaterialAnalyticsResponse,
    summary="Get aggregated material estimation statistics",
)
def get_material_analytics(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MaterialAnalyticsResponse:
    """Returns overall material consumption quantities, costs, and project averages."""
    svc = AdminAnalyticsService(db)
    return svc.get_material_analytics()


@router.get(
    "/dashboard/models",
    response_model=ModelPerformanceResponse,
    summary="Get model observability telemetry per model version",
)
def get_model_performance(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> ModelPerformanceResponse:
    """Observability breakdown of inference counts, latencies, and error rates per model checkpoint."""
    svc = AdminAnalyticsService(db)
    return svc.get_model_performance()


# ─────────────────────────────────────────────────────────────────────────────
# 2. User & Access Governance (IAM)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/users",
    response_model=AdminUserListResponse,
    summary="List registered users with search, role filters, and job counts",
)
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by email, username, or full name"),
    is_active: Optional[bool] = Query(None, description="Filter by active/inactive status"),
    role: Optional[str] = Query(None, description="Filter by role e.g. admin or user"),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AdminUserListResponse:
    """Returns paginated user list with their job statistics."""
    svc = AdminManagementService(db, admin_actor=_admin)
    return svc.list_users(page=page, page_size=page_size, search=search, is_active=is_active, role=role)


@router.put(
    "/users/{user_id}/roles",
    response_model=UserResponse,
    summary="Update roles for a user [admin only]",
)
def update_user_roles(
    user_id: str,
    payload: UpdateUserRoleRequest,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Assign or revoke user roles (e.g. ['user', 'admin']). Automatically audited."""
    svc = AdminManagementService(db, admin_actor=admin_user, ip_address=_get_client_ip(request))
    try:
        user = svc.update_user_roles(user_id, payload)
        return UserResponse.model_validate(user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.put(
    "/users/{user_id}/status",
    response_model=UserResponse,
    summary="Activate or soft-deactivate a user account [admin only]",
)
def update_user_status(
    user_id: str,
    payload: AdminUpdateUserStatusRequest,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Activate or deactivate user account and invalidate active sessions."""
    svc = AdminManagementService(db, admin_actor=admin_user, ip_address=_get_client_ip(request))
    try:
        user = svc.set_user_status(user_id, payload)
        return UserResponse.model_validate(user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post(
    "/users/{user_id}/reset-password",
    response_model=MessageResponse,
    summary="Force reset a user's password [admin only]",
)
def reset_user_password(
    user_id: str,
    payload: AdminResetPasswordRequest,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Directly sets a new password and revokes existing refresh tokens."""
    svc = AdminManagementService(db, admin_actor=admin_user, ip_address=_get_client_ip(request))
    try:
        svc.reset_user_password(user_id, payload)
        return MessageResponse(message="User password has been successfully reset.")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Global Job Management
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/jobs",
    response_model=AdminJobListResponse,
    summary="Browse all segmentation jobs across the platform [admin only]",
)
def list_all_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status: success, failed, pending"),
    user_id: Optional[str] = Query(None, description="Filter by specific user ID"),
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    start_date: Optional[datetime] = Query(None, description="ISO start date filter"),
    end_date: Optional[datetime] = Query(None, description="ISO end date filter"),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AdminJobListResponse:
    """Global job inspector with pagination and multi-parameter filtering."""
    svc = AdminManagementService(db, admin_actor=_admin)
    return svc.list_all_jobs(
        page=page,
        page_size=page_size,
        status=status,
        user_id=user_id,
        model_version=model_version,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=SegmentationResponse,
    summary="Inspect full raw results and polygon contours for any job [admin only]",
)
def get_job_detail(
    job_id: str,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> SegmentationResponse:
    """Inspect full segmentation payload, masks, and material breakdowns."""
    job = JobRepository.get_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    result = JobRepository.result_from_job(job)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No result payload found for this job.")
    return result


@router.delete(
    "/jobs/{job_id}",
    response_model=MessageResponse,
    summary="Hard delete a job and clean up its stored masks on disk [admin only]",
)
def delete_job(
    job_id: str,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Deletes job database record and removes associated mask files."""
    svc = AdminManagementService(db, admin_actor=admin_user, ip_address=_get_client_ip(request))
    try:
        svc.delete_job(job_id)
        return MessageResponse(message=f"Job '{job_id}' and associated files have been permanently deleted.")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post(
    "/jobs/purge",
    response_model=MessageResponse,
    summary="Batch purge old or failed jobs beyond a retention cutoff [admin only]",
)
def purge_jobs(
    payload: PurgeJobsRequest,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Batch delete jobs older than N days to free storage space."""
    svc = AdminManagementService(db, admin_actor=admin_user, ip_address=_get_client_ip(request))
    deleted_count = svc.purge_jobs(payload)
    return MessageResponse(message=f"Purged {deleted_count} job records older than {payload.older_than_days} days.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Material Pricing & Dynamic System Configuration
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/pricing",
    response_model=MaterialPricingConfigResponse,
    summary="Get global ceiling material estimation unit prices and waste multipliers",
)
def get_pricing_config(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MaterialPricingConfigResponse:
    """Fetch active material pricing catalog."""
    svc = AdminManagementService(db, admin_actor=_admin)
    return svc.get_pricing_config()


@router.put(
    "/pricing",
    response_model=MaterialPricingConfigResponse,
    summary="Update global material estimation unit prices and waste multiplier [admin only]",
)
def update_pricing_config(
    payload: MaterialPricingUpdateRequest,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MaterialPricingConfigResponse:
    """Save new material prices and waste buffer. Automatically audited."""
    svc = AdminManagementService(db, admin_actor=admin_user, ip_address=_get_client_ip(request))
    return svc.update_pricing_config(payload)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Audit Log Inspection & System Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/audit-logs",
    response_model=AuditLogListResponse,
    summary="View immutable audit trail of administrative activities [admin only]",
)
def list_audit_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    action: Optional[str] = Query(None, description="Filter by action name e.g. USER_ROLE_UPDATED"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type e.g. user, job"),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AuditLogListResponse:
    """Fetch paginated audit records."""
    svc = AdminManagementService(db, admin_actor=_admin)
    return svc.list_audit_logs(page=page, page_size=page_size, action=action, resource_type=resource_type)


@router.get(
    "/system/health",
    response_model=SystemHealthResponse,
    summary="System diagnostic telemetry, storage usage, and DB status [admin only]",
)
def get_system_health(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> SystemHealthResponse:
    """System health check and diagnostic metrics."""
    svc = AdminManagementService(db, admin_actor=_admin)
    return svc.get_system_health()
