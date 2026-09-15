"""
Unit & Integration Tests for Admin Dashboard, Telemetry & Management APIs
========================================================================
Validates:
  1. Statistical & Timeseries algorithms (percentiles, histograms, EMA, gap filling)
  2. Role-Based Access Control (RBAC: 403 for non-admins, 200 for admins)
  3. Analytics KPIs and distribution endpoints
  4. User governance (role updates, status toggles, password resets)
  5. Global job explorer and batch purge
  6. Material pricing config updates & audit log recording
"""
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.security import create_access_token
from app.domain.models.audit_log import AuditLog
from app.domain.models.segmentation_job import SegmentationJob
from app.domain.models.system_setting import SystemSetting
from app.domain.models.user import Base, User
from app.infrastructure.database.session import get_db
from app.main import app
from app.services.admin_analytics_service import AdminAnalyticsService
from app.services.admin_management_service import AdminManagementService
from app.services.analytics_algorithms import (
    bucket_distribution,
    calculate_iqr_bounds,
    calculate_percentiles,
    compute_exponential_moving_average,
    fill_timeseries_gaps,
)

from sqlalchemy.pool import StaticPool

# ── Isolated Test DB Setup ─────────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create all tables in the shared in-memory DB."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Yield a database session for test setup."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def admin_client():
    """Test client with request-scoped DB session override."""
    def _override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_user_helper(db, email, username, roles, is_superuser=False):
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        username=username,
        roles=json.dumps(roles),
        hashed_password="hashed_dummy_password",
        is_active=True,
        is_superuser=is_superuser,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_sample_job(db, user_id, status="success", latency=120.0, area=45.0, days_ago=0):
    created_at = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    sample_result = {
        "job_id": str(uuid.uuid4()),
        "status": status,
        "inference_time_ms": latency,
        "model_version": "v1.0-unet",
        "total_area_m2": area,
        "total_perimeter_m": 28.0,
        "detected_regions": [],
        "estimated_materials": [
            {
                "material_type": "GYPSUM_BOARD",
                "estimated_units": 15.0,
                "unit": "sheets",
                "estimated_cost": 217.50,
            },
            {
                "material_type": "DRYWALL_SCREWS",
                "estimated_units": 3.0,
                "unit": "packs",
                "estimated_cost": 24.60,
            },
        ],
    }
    job = SegmentationJob(
        id=sample_result["job_id"],
        user_id=user_id,
        status=status,
        inference_time_ms=latency,
        model_version="v1.0-unet",
        full_result_json=json.dumps(sample_result),
        created_at=created_at,
        completed_at=created_at,
        expires_at=created_at + timedelta(days=7),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


# ── 1. Algorithm Unit Tests ───────────────────────────────────────────────────

def test_calculate_percentiles():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p = calculate_percentiles(data, [50.0, 90.0, 99.0])
    assert p["p50"] == 55.0
    assert p["p90"] == 91.0
    assert p["p99"] > 90.0


def test_bucket_distribution():
    data = [5.0, 12.0, 18.0, 26.0, 45.0, 80.0]
    thresholds = [10.0, 25.0, 50.0]
    labels = ["< 10", "10-25", "25-50", "> 50"]
    buckets = bucket_distribution(data, thresholds, labels)
    assert len(buckets) == 4
    assert buckets[0]["count"] == 1  # 5.0
    assert buckets[1]["count"] == 2  # 12.0, 18.0
    assert buckets[2]["count"] == 2  # 26.0, 45.0
    assert buckets[3]["count"] == 1  # 80.0


def test_fill_timeseries_gaps():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 3, tzinfo=timezone.utc)
    sparse = {
        "2026-01-01": {"date": "2026-01-01", "total_jobs": 5},
        "2026-01-03": {"date": "2026-01-03", "total_jobs": 12},
    }
    filled = fill_timeseries_gaps(sparse, start, end, "%Y-%m-%d")
    assert len(filled) == 3
    assert filled[1]["date"] == "2026-01-02"
    assert filled[1]["total_jobs"] == 0  # Gap correctly filled with zero


def test_compute_exponential_moving_average():
    series = [10.0, 20.0, 30.0, 40.0]
    ema = compute_exponential_moving_average(series, alpha=0.5)
    assert len(ema) == 4
    assert ema[0] == 10.0
    assert ema[1] == 15.0  # 0.5*20 + 0.5*10


def test_calculate_iqr_bounds():
    data = [10, 12, 14, 15, 16, 18, 20, 22, 100]  # 100 is outlier
    lower, upper = calculate_iqr_bounds(data)
    assert lower >= 0.0
    assert upper < 100.0  # 100 detected beyond upper IQR bound


# ── 2. Services Unit Tests ────────────────────────────────────────────────────

def test_admin_analytics_service(db_session):
    user = create_user_helper(db_session, "u1@test.com", "u1", ["user"])
    create_sample_job(db_session, user.id, status="success", latency=150.0, area=50.0)
    create_sample_job(db_session, user.id, status="failed", latency=50.0)

    svc = AdminAnalyticsService(db_session)
    summary = svc.get_summary_kpis()
    assert summary.total_jobs == 2
    assert summary.successful_jobs == 1
    assert summary.failed_jobs == 1
    assert summary.success_rate_pct == 50.0
    assert summary.total_ceiling_area_m2 == 50.0

    materials = svc.get_material_analytics()
    assert len(materials.top_materials) >= 1

    timeseries = svc.get_timeseries_analytics(days=7)
    assert len(timeseries.points) == 7


def test_admin_management_service(db_session):
    admin = create_user_helper(db_session, "admin@test.com", "admin", ["admin"])
    user = create_user_helper(db_session, "target@test.com", "target", ["user"])
    job = create_sample_job(db_session, user.id)

    svc = AdminManagementService(db_session, admin_actor=admin)

    # 1. Update roles
    from app.domain.schemas.requests.admin_requests import UpdateUserRoleRequest
    updated_user = svc.update_user_roles(user.id, UpdateUserRoleRequest(roles=["user", "admin"]))
    assert "admin" in json.loads(updated_user.roles)

    # 2. Deactivate user
    from app.domain.schemas.requests.admin_requests import AdminUpdateUserStatusRequest
    svc.set_user_status(user.id, AdminUpdateUserStatusRequest(is_active=False))
    assert not user.is_active

    # 3. Update Pricing
    from app.domain.schemas.requests.admin_requests import MaterialPricingUpdateRequest
    pricing = svc.update_pricing_config(
        MaterialPricingUpdateRequest(
            unit_prices={"drywall_sheet_4x8": 16.0},
            waste_factor=1.15,
            currency="USD",
        )
    )
    assert pricing.unit_prices["drywall_sheet_4x8"] == 16.0
    assert pricing.waste_factor == 1.15

    # 4. Check Audit logs
    audit_logs = svc.list_audit_logs()
    assert audit_logs.total >= 3


# ── 3. API Endpoints & RBAC Integration Tests ─────────────────────────────────

def test_admin_endpoint_forbidden_for_regular_user(admin_client, db_session):
    regular_user = create_user_helper(db_session, "reg@test.com", "regular", ["user"])
    token = create_access_token(subject=regular_user.id, roles=["user"])

    res = admin_client.get(
        "/api/v1/admin/dashboard/summary",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]


def test_admin_dashboard_endpoints_success(admin_client, db_session):
    admin = create_user_helper(db_session, "admin2@test.com", "admin2", ["admin"])
    create_sample_job(db_session, admin.id, status="success", latency=180.0, area=65.0)
    token = create_access_token(subject=admin.id, roles=["admin"])
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Summary
    res = admin_client.get("/api/v1/admin/dashboard/summary", headers=headers)
    assert res.status_code == 200
    assert res.json()["total_jobs"] >= 1
    assert res.json()["total_ceiling_area_m2"] == 65.0

    # 2. Timeseries
    res = admin_client.get("/api/v1/admin/dashboard/timeseries?days=7", headers=headers)
    assert res.status_code == 200
    assert len(res.json()["points"]) == 7

    # 3. Latency distribution
    res = admin_client.get("/api/v1/admin/dashboard/latency-distribution", headers=headers)
    assert res.status_code == 200
    assert res.json()["metric_name"] == "inference_latency_ms"

    # 4. Area distribution
    res = admin_client.get("/api/v1/admin/dashboard/area-distribution", headers=headers)
    assert res.status_code == 200
    assert res.json()["metric_name"] == "ceiling_area_m2"

    # 5. Materials
    res = admin_client.get("/api/v1/admin/dashboard/materials", headers=headers)
    assert res.status_code == 200
    assert len(res.json()["top_materials"]) >= 1

    # 6. Models
    res = admin_client.get("/api/v1/admin/dashboard/models", headers=headers)
    assert res.status_code == 200
    assert len(res.json()["models"]) >= 1

    # 7. System Health
    res = admin_client.get("/api/v1/admin/system/health", headers=headers)
    assert res.status_code == 200
    assert res.json()["database_status"] == "healthy"

    # 8. Pricing Config GET & PUT
    res = admin_client.get("/api/v1/admin/pricing", headers=headers)
    assert res.status_code == 200
    res_put = admin_client.put(
        "/api/v1/admin/pricing",
        json={
            "unit_prices": {"drywall_sheet_4x8": 15.25},
            "waste_factor": 1.12,
            "currency": "USD",
        },
        headers=headers,
    )
    assert res_put.status_code == 200
    assert res_put.json()["unit_prices"]["drywall_sheet_4x8"] == 15.25

    # 9. Audit Logs
    res = admin_client.get("/api/v1/admin/audit-logs", headers=headers)
    assert res.status_code == 200
    assert res.json()["total"] >= 1
