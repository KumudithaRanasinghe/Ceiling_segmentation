"""
Admin Analytics Service
========================
High-performance business and operational telemetry aggregation service.
Leverages statistical and timeseries algorithms to generate chart data and KPIs.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.domain.models.segmentation_job import SegmentationJob
from app.domain.models.user import User
from app.domain.repositories.job_repository import JobRepository
from app.domain.repositories.setting_repository import SettingRepository
from app.domain.repositories.user_repository import UserRepository
from app.domain.schemas.responses.admin_responses import (
    AdminSummaryKPIResponse,
    BucketItem,
    DistributionResponse,
    MaterialAnalyticsResponse,
    MaterialStatItem,
    ModelPerformanceItem,
    ModelPerformanceResponse,
    TimeseriesAnalyticsResponse,
    TimeseriesPoint,
)
from app.services.analytics_algorithms import (
    bucket_distribution,
    calculate_percentiles,
    compute_exponential_moving_average,
    fill_timeseries_gaps,
)


class AdminAnalyticsService:
    """Provides aggregated analytics and telemetry for the Admin Dashboard."""

    def __init__(self, db: Session):
        self.db = db

    def get_summary_kpis(self) -> AdminSummaryKPIResponse:
        """Calculate high-level summary KPIs."""
        now = datetime.now(tz=timezone.utc)
        since_24h = now - timedelta(hours=24)
        since_30d = now - timedelta(days=30)

        # Total counts
        status_counts = JobRepository.count_by_status(self.db)
        total_jobs = sum(status_counts.values())
        successful_jobs = status_counts.get("success", 0)
        failed_jobs = status_counts.get("failed", 0)
        success_rate = round((successful_jobs / total_jobs * 100.0), 2) if total_jobs > 0 else 100.0

        # Active users in 24h / 30d
        active_24h = (
            self.db.query(func.count(func.distinct(SegmentationJob.user_id)))
            .filter(SegmentationJob.created_at >= since_24h)
            .scalar() or 0
        )
        active_30d = (
            self.db.query(func.count(func.distinct(SegmentationJob.user_id)))
            .filter(SegmentationJob.created_at >= since_30d)
            .scalar() or 0
        )
        total_users = UserRepository.count_total(self.db)

        # Latency, area, and estimated cost aggregation
        all_jobs = self.db.query(SegmentationJob).all()
        latencies = [j.inference_time_ms for j in all_jobs if j.inference_time_ms is not None]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        total_area = 0.0
        total_cost = 0.0

        for job in all_jobs:
            if job.status == "success" and job.full_result_json:
                try:
                    data = json.loads(job.full_result_json)
                    total_area += data.get("total_area_m2", 0.0)
                    for mat in data.get("estimated_materials", []):
                        total_cost += mat.get("estimated_cost", 0.0)
                except Exception:
                    continue

        # Storage calculation for results directory
        storage_mb = self._calculate_results_storage_mb()

        return AdminSummaryKPIResponse(
            total_jobs=total_jobs,
            successful_jobs=successful_jobs,
            failed_jobs=failed_jobs,
            success_rate_pct=success_rate,
            active_users_24h=active_24h,
            active_users_30d=active_30d,
            total_registered_users=total_users,
            avg_inference_latency_ms=avg_latency,
            total_ceiling_area_m2=round(total_area, 2),
            total_estimated_cost_usd=round(total_cost, 2),
            storage_used_mb=round(storage_mb, 2),
        )

    def get_timeseries_analytics(
        self, days: int = 14
    ) -> TimeseriesAnalyticsResponse:
        """
        Generate daily timeseries for job volume, failure rate, latency, and segmented area.
        Applies chronological gap-filling and Exponential Moving Average trend line.
        """
        now = datetime.now(tz=timezone.utc)
        start_date = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        jobs = JobRepository.get_jobs_in_timerange(self.db, start_date, now)

        # Aggregate raw database points by YYYY-MM-DD
        grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_jobs": 0,
            "success_jobs": 0,
            "failed_jobs": 0,
            "latency_list": [],
            "total_area_m2": 0.0,
        })

        for job in jobs:
            day_str = job.created_at.strftime("%Y-%m-%d")
            grouped[day_str]["total_jobs"] += 1
            if job.status == "success":
                grouped[day_str]["success_jobs"] += 1
            elif job.status == "failed":
                grouped[day_str]["failed_jobs"] += 1

            if job.inference_time_ms is not None:
                grouped[day_str]["latency_list"].append(job.inference_time_ms)

            if job.status == "success" and job.full_result_json:
                try:
                    payload = json.loads(job.full_result_json)
                    grouped[day_str]["total_area_m2"] += payload.get("total_area_m2", 0.0)
                except Exception:
                    pass

        sparse_points: Dict[str, Dict[str, Any]] = {}
        for day_str, info in grouped.items():
            l_list = info["latency_list"]
            avg_lat = round(sum(l_list) / len(l_list), 2) if l_list else 0.0
            sparse_points[day_str] = {
                "date": day_str,
                "total_jobs": info["total_jobs"],
                "success_jobs": info["success_jobs"],
                "failed_jobs": info["failed_jobs"],
                "avg_latency_ms": avg_lat,
                "total_area_m2": round(info["total_area_m2"], 2),
            }

        # Fill missing days algorithmically
        filled_timeline = fill_timeseries_gaps(
            sparse_points=sparse_points,
            start_dt=start_date,
            end_dt=now,
            date_format="%Y-%m-%d",
            step_delta=timedelta(days=1),
        )

        # Compute smoothed trend curve using EMA algorithm
        daily_volumes = [float(p["total_jobs"]) for p in filled_timeline]
        ema_trend = compute_exponential_moving_average(daily_volumes, alpha=0.35)

        points = [
            TimeseriesPoint(
                date=p["date"],
                total_jobs=p["total_jobs"],
                success_jobs=p["success_jobs"],
                failed_jobs=p["failed_jobs"],
                avg_latency_ms=p["avg_latency_ms"],
                total_area_m2=p["total_area_m2"],
                smoothed_trend=ema_trend[i] if i < len(ema_trend) else None,
            )
            for i, p in enumerate(filled_timeline)
        ]

        return TimeseriesAnalyticsResponse(interval="daily", points=points)

    def get_latency_distribution(self) -> DistributionResponse:
        """Compute latency percentiles (p50, p90, p95, p99) and histogram buckets."""
        jobs = self.db.query(SegmentationJob.inference_time_ms).filter(
            SegmentationJob.inference_time_ms.isnot(None)
        ).all()
        latencies = [j[0] for j in jobs if j[0] is not None]

        if not latencies:
            return DistributionResponse(
                metric_name="inference_latency_ms",
                p50=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                min_value=0.0,
                max_value=0.0,
                buckets=[],
            )

        percentiles = calculate_percentiles(latencies, (50.0, 90.0, 95.0, 99.0))
        thresholds = [100.0, 250.0, 500.0, 1000.0, 2000.0]
        labels = ["< 100ms", "100 - 250ms", "250 - 500ms", "500 - 1000ms", "1000 - 2000ms", "> 2000ms"]
        raw_buckets = bucket_distribution(latencies, thresholds, labels)

        return DistributionResponse(
            metric_name="inference_latency_ms",
            p50=percentiles["p50"],
            p90=percentiles["p90"],
            p95=percentiles["p95"],
            p99=percentiles["p99"],
            min_value=round(min(latencies), 2),
            max_value=round(max(latencies), 2),
            buckets=[BucketItem(**b) for b in raw_buckets],
        )

    def get_area_distribution(self) -> DistributionResponse:
        """Compute ceiling area distribution histogram and percentiles."""
        all_jobs = self.db.query(SegmentationJob.full_result_json).filter(
            SegmentationJob.status == "success"
        ).all()

        areas: List[float] = []
        for j in all_jobs:
            if j[0]:
                try:
                    payload = json.loads(j[0])
                    area = payload.get("total_area_m2")
                    if area is not None and area > 0:
                        areas.append(float(area))
                except Exception:
                    pass

        if not areas:
            return DistributionResponse(
                metric_name="ceiling_area_m2",
                p50=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                min_value=0.0,
                max_value=0.0,
                buckets=[],
            )

        percentiles = calculate_percentiles(areas, (50.0, 90.0, 95.0, 99.0))
        thresholds = [15.0, 30.0, 60.0, 120.0]
        labels = ["< 15 m²", "15 - 30 m²", "30 - 60 m²", "60 - 120 m²", "> 120 m²"]
        raw_buckets = bucket_distribution(areas, thresholds, labels)

        return DistributionResponse(
            metric_name="ceiling_area_m2",
            p50=percentiles["p50"],
            p90=percentiles["p90"],
            p95=percentiles["p95"],
            p99=percentiles["p99"],
            min_value=round(min(areas), 2),
            max_value=round(max(areas), 2),
            buckets=[BucketItem(**b) for b in raw_buckets],
        )

    def get_material_analytics(self) -> MaterialAnalyticsResponse:
        """Aggregate material consumption trends across all completed projects."""
        success_jobs = self.db.query(SegmentationJob.full_result_json).filter(
            SegmentationJob.status == "success"
        ).all()

        material_totals: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_units": 0.0,
            "unit": "",
            "estimated_total_cost": 0.0,
            "frequency_count": 0,
        })

        total_area = 0.0
        project_count = 0

        for j in success_jobs:
            if not j[0]:
                continue
            try:
                payload = json.loads(j[0])
                total_area += payload.get("total_area_m2", 0.0)
                project_count += 1
                for mat in payload.get("estimated_materials", []):
                    name = mat.get("material_type", "unknown")
                    material_totals[name]["total_units"] += mat.get("estimated_units", 0.0)
                    material_totals[name]["unit"] = mat.get("unit", "units")
                    material_totals[name]["estimated_total_cost"] += mat.get("estimated_cost", 0.0)
                    material_totals[name]["frequency_count"] += 1
            except Exception:
                continue

        top_materials = [
            MaterialStatItem(
                material_name=mat_name,
                total_units=round(data["total_units"], 2),
                unit=data["unit"],
                estimated_total_cost=round(data["estimated_total_cost"], 2),
                frequency_count=data["frequency_count"],
            )
            for mat_name, data in sorted(
                material_totals.items(),
                key=lambda x: x[1]["frequency_count"],
                reverse=True,
            )
        ]

        avg_area = round(total_area / project_count, 2) if project_count > 0 else 0.0
        pricing_config = SettingRepository.get_value(self.db, "material_pricing", {})
        default_waste = pricing_config.get("waste_factor", settings.MATERIAL_OVERHEAD_FACTOR)

        return MaterialAnalyticsResponse(
            top_materials=top_materials,
            average_ceiling_area_m2=avg_area,
            total_projects_analyzed=project_count,
            default_waste_factor=default_waste,
        )

    def get_model_performance(self) -> ModelPerformanceResponse:
        """Observability report for AI model versions."""
        jobs = self.db.query(
            SegmentationJob.model_version,
            SegmentationJob.status,
            SegmentationJob.inference_time_ms,
        ).all()

        model_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total": 0,
            "errors": 0,
            "latencies": [],
        })

        for version, status, latency in jobs:
            v_key = version or "default_v1"
            model_stats[v_key]["total"] += 1
            if status == "failed":
                model_stats[v_key]["errors"] += 1
            if latency is not None:
                model_stats[v_key]["latencies"].append(latency)

        performance_items = []
        for version, data in model_stats.items():
            tot = data["total"]
            err = data["errors"]
            l_list = data["latencies"]
            avg_l = round(sum(l_list) / len(l_list), 2) if l_list else 0.0
            err_rate = round((err / tot * 100.0), 2) if tot > 0 else 0.0
            performance_items.append(
                ModelPerformanceItem(
                    model_version=version,
                    total_inferences=tot,
                    avg_latency_ms=avg_l,
                    error_count=err,
                    error_rate_pct=err_rate,
                )
            )

        return ModelPerformanceResponse(
            active_encoder=settings.ENCODER,
            device=settings.DEVICE,
            models=performance_items,
        )

    def _calculate_results_storage_mb(self) -> float:
        """Sum file sizes in results/ directory."""
        results_path = Path("results")
        if not results_path.exists():
            return 0.0
        total_bytes = 0
        for root, _, files in os.walk(results_path):
            for f in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total_bytes / (1024 * 1024)
