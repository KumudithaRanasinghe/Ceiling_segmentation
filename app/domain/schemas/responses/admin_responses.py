"""
Admin Response Schemas
=======================
Pydantic schemas for Admin Dashboard analytics, telemetry, data listings, and system stats.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.domain.schemas.responses.segmentation_responses import SegmentationResponse


class AdminSummaryKPIResponse(BaseModel):
    """Top-level executive summary metrics."""
    total_jobs: int = Field(..., description="Total inference jobs processed.")
    successful_jobs: int = Field(..., description="Total successful inference jobs.")
    failed_jobs: int = Field(..., description="Total failed inference jobs.")
    success_rate_pct: float = Field(..., description="Inference success rate percentage.")
    active_users_24h: int = Field(..., description="Unique users with jobs in past 24 hours.")
    active_users_30d: int = Field(..., description="Unique users with jobs in past 30 days.")
    total_registered_users: int = Field(..., description="Total registered accounts.")
    avg_inference_latency_ms: float = Field(..., description="Average inference latency in ms.")
    total_ceiling_area_m2: float = Field(..., description="Aggregated ceiling area segmented (m²).")
    total_estimated_cost_usd: float = Field(..., description="Aggregated material estimation cost.")
    storage_used_mb: float = Field(..., description="Disk space used by output masks in MB.")


class TimeseriesPoint(BaseModel):
    """Single time interval point for trend charts."""
    date: str = Field(..., description="Date / timestamp bucket string (e.g. YYYY-MM-DD).")
    total_jobs: int = 0
    success_jobs: int = 0
    failed_jobs: int = 0
    avg_latency_ms: float = 0.0
    total_area_m2: float = 0.0
    smoothed_trend: Optional[float] = None


class TimeseriesAnalyticsResponse(BaseModel):
    """Timeseries telemetry for dashboard charts."""
    interval: str = Field(..., description="Interval type: 'daily' or 'hourly'.")
    points: List[TimeseriesPoint] = Field(..., description="Contiguous chronologically filled data points.")


class BucketItem(BaseModel):
    """Histogram bucket item."""
    label: str
    count: int
    percentage: float


class DistributionResponse(BaseModel):
    """Distribution histogram + percentiles for latency or area."""
    metric_name: str
    p50: float
    p90: float
    p95: float
    p99: float
    min_value: float
    max_value: float
    buckets: List[BucketItem]


class MaterialStatItem(BaseModel):
    """Statistics for a specific material type."""
    material_name: str
    total_units: float
    unit: str
    estimated_total_cost: float
    frequency_count: int


class MaterialAnalyticsResponse(BaseModel):
    """Aggregated material estimation analytics."""
    top_materials: List[MaterialStatItem]
    average_ceiling_area_m2: float
    total_projects_analyzed: int
    default_waste_factor: float


class ModelPerformanceItem(BaseModel):
    """Performance breakdown per model version."""
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    total_inferences: int
    avg_latency_ms: float
    error_count: int
    error_rate_pct: float


class ModelPerformanceResponse(BaseModel):
    """Model registry observability stats."""
    active_encoder: str
    device: str
    models: List[ModelPerformanceItem]


class AdminUserListItem(BaseModel):
    """User account summary for admin user directory."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    username: str
    full_name: Optional[str]
    roles: List[str]
    is_active: bool
    is_superuser: bool
    job_count: int = 0
    last_job_at: Optional[datetime] = None
    created_at: datetime


class AdminUserListResponse(BaseModel):
    """Paginated user directory response."""
    total: int
    page: int
    page_size: int
    users: List[AdminUserListItem]


class AdminJobListItem(BaseModel):
    """Segmentation job row for admin global job browser."""
    model_config = ConfigDict(protected_namespaces=())

    id: str
    user_id: str
    user_email: Optional[str] = None
    status: str
    image_filename: Optional[str] = None
    image_size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    pixels_per_meter: Optional[float] = None
    inference_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    total_area_m2: Optional[float] = None
    created_at: datetime
    expires_at: Optional[datetime] = None


class AdminJobListResponse(BaseModel):
    """Paginated global job list."""
    total: int
    page: int
    page_size: int
    jobs: List[AdminJobListItem]


class AuditLogItem(BaseModel):
    """Single audit log record."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    actor_id: str
    actor_email: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    details_json: Optional[str]
    ip_address: Optional[str]
    created_at: datetime


class AuditLogListResponse(BaseModel):
    """Paginated audit trail response."""
    total: int
    page: int
    page_size: int
    logs: List[AuditLogItem]


class SystemHealthResponse(BaseModel):
    """Infrastructure and diagnostic health report."""
    model_config = ConfigDict(protected_namespaces=())

    database_status: str
    database_driver: str
    total_users_count: int
    total_jobs_count: int
    total_audit_logs_count: int
    results_directory_files_count: int
    results_directory_size_mb: float
    model_loaded: bool
    active_device: str
    timestamp: datetime


class MaterialPricingConfigResponse(BaseModel):
    """Global pricing configuration response."""
    unit_prices: Dict[str, float]
    waste_factor: float
    currency: str
    updated_at: datetime
    updated_by: Optional[str] = None
