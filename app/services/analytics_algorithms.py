"""
Analytics & Statistical Algorithms
===================================
Reusable, high-efficiency statistical and timeseries algorithms for Admin Dashboard
visualizations, telemetry metrics, and anomaly detection.
Avoids code duplication across analytics services.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Sequence, Tuple


def calculate_percentiles(
    values: Sequence[float],
    percentile_targets: Sequence[float] = (50.0, 90.0, 95.0, 99.0),
) -> Dict[str, float]:
    """
    Computes exact percentiles from a sequence of numeric values using linear interpolation.
    
    Args:
        values: Non-empty sequence of floats.
        percentile_targets: List of percentiles (e.g. [50.0, 90.0, 95.0, 99.0]).
        
    Returns:
        Dictionary mapping percentile labels (e.g., 'p50', 'p90') to interpolated values.
    """
    if not values:
        return {f"p{int(p)}": 0.0 for p in percentile_targets}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    results: Dict[str, float] = {}

    for p in percentile_targets:
        if n == 1:
            results[f"p{int(p)}"] = round(sorted_vals[0], 2)
            continue

        rank = (p / 100.0) * (n - 1)
        lower_idx = int(math.floor(rank))
        upper_idx = int(math.ceil(rank))
        weight = rank - lower_idx

        val = (1.0 - weight) * sorted_vals[lower_idx] + weight * sorted_vals[upper_idx]
        results[f"p{int(p)}"] = round(val, 2)

    return results


def bucket_distribution(
    values: Sequence[float],
    thresholds: Sequence[float],
    labels: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Bins numeric data into histogram buckets based on cutoff thresholds.
    
    Example:
        thresholds = [10.0, 25.0, 50.0]
        labels = ["< 10 m²", "10 - 25 m²", "25 - 50 m²", "> 50 m²"]
    """
    if len(labels) != len(thresholds) + 1:
        raise ValueError("labels count must be exactly thresholds count + 1")

    counts = [0] * len(labels)
    total = len(values)

    for val in values:
        placed = False
        for i, cutoff in enumerate(thresholds):
            if val < cutoff:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    return [
        {
            "label": label,
            "count": count,
            "percentage": round((count / total * 100.0), 1) if total > 0 else 0.0,
        }
        for label, count in zip(labels, counts)
    ]


def fill_timeseries_gaps(
    sparse_points: Dict[str, Dict[str, Any]],
    start_dt: datetime,
    end_dt: datetime,
    date_format: str = "%Y-%m-%d",
    step_delta: timedelta = timedelta(days=1),
    default_factory: Callable[[str], Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """
    Ensures a continuous chronological timeline without missing intervals.
    Fills days/hours that had 0 database events with structured zero-data defaults.
    """
    if default_factory is None:
        default_factory = lambda key: {
            "date": key,
            "total_jobs": 0,
            "success_jobs": 0,
            "failed_jobs": 0,
            "avg_latency_ms": 0.0,
            "total_area_m2": 0.0,
        }

    timeline: List[Dict[str, Any]] = []
    current = start_dt

    while current <= end_dt:
        key = current.strftime(date_format)
        if key in sparse_points:
            timeline.append(sparse_points[key])
        else:
            timeline.append(default_factory(key))
        current += step_delta

    return timeline


def compute_exponential_moving_average(
    values: Sequence[float],
    alpha: float = 0.3,
) -> List[float]:
    """
    Computes an Exponential Moving Average (EMA) to extract smooth trend lines.
    
    Formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    """
    if not values:
        return []

    ema: List[float] = [round(values[0], 2)]
    for i in range(1, len(values)):
        val = alpha * values[i] + (1.0 - alpha) * ema[-1]
        ema.append(round(val, 2))
    return ema


def calculate_iqr_bounds(values: Sequence[float]) -> Tuple[float, float]:
    """
    Computes Interquartile Range (IQR) lower and upper bounds for anomaly detection.
    Values outside [lower_bound, upper_bound] are potential anomalies.
    """
    if len(values) < 4:
        return 0.0, float("inf")

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[int(n * 0.25)]
    q3 = sorted_vals[int(n * 0.75)]
    iqr = q3 - q1

    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    return round(lower_bound, 2), round(upper_bound, 2)
