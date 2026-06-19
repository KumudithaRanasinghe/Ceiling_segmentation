"""
Health endpoints — used by Docker/Kubernetes liveness and readiness probes.

GET /api/v1/health/live   → 200 if process is alive
GET /api/v1/health/ready  → 200 only if models are loaded and inference is possible
"""
import time

import torch
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter()

_start_time = time.time()


@router.get("/live", status_code=status.HTTP_200_OK, summary="Liveness probe")
async def liveness():
    """Returns 200 immediately. If this fails, the process has crashed."""
    return {"status": "alive", "uptime_s": round(time.time() - _start_time, 1)}


@router.get("/ready", status_code=status.HTTP_200_OK, summary="Readiness probe")
async def readiness(request: Request):
    """
    Returns 200 only when both models are loaded and a dummy forward pass succeeds.
    Returns 503 during startup or after a model load failure.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "model_registry not initialised"},
        )

    try:
        # Quick smoke-test: tiny dummy tensor through both models
        dummy = torch.zeros(1, 3, 64, 64, device=registry.device)
        with torch.no_grad():
            denoised = registry.denoiser(dummy)
            _ = registry.segmenter(denoised)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": str(exc)},
        )

    return {
        "status": "ready",
        "device": str(registry.device),
        "model_version": registry.model_version,
    }
