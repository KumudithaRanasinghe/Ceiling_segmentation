"""
Segmentation Endpoint.
========================
POST /api/v1/segment/
  - Accepts a multipart image upload.
  - Runs the full inference pipeline.
  - Persists the result as a SegmentationJob in the database.
  - Returns the full SegmentationResponse including job_id.

GET /api/v1/segment/{job_id}
  - Retrieves a previously completed job by its job_id.
  - Only the owner (authenticated user who created the job) can retrieve it.
  - Returns 404 if the job doesn't exist or belongs to another user.

GET /api/v1/segment/
  - Lists the current user's most recent jobs (paginated).

Security:
  - All routes require a valid Bearer access token.
  - Job retrieval is scoped to the requesting user's ID.
  - Rate limited to 5 requests/min/user by RateLimitMiddleware.
"""
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.core.exceptions import ImageValidationError, InferenceError
from app.domain.models.user import User
from app.domain.repositories.job_repository import JobRepository
from app.domain.schemas.requests import SegmentationRequest
from app.domain.schemas.responses import (
    ErrorResponse,
    SegmentationJobSummary,
    SegmentationResponse,
)
from app.infrastructure.database.session import get_db
from app.services.segmentation_service import SegmentationService

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Job summary schema ────────────────────────────────────────────────────────

class SegmentationJobSummary(BaseModel):
    """Lightweight job list item — no heavy result payload."""
    job_id: str
    status: str
    image_filename: Optional[str]
    inference_time_ms: Optional[float]
    model_version: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ── POST / — Run segmentation ─────────────────────────────────────────────────

@router.post(
    "/",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment a roof area image",
    description=(
        "Upload a roof image. The system runs noise reduction, semantic segmentation, "
        "geometry analysis, and returns ceiling material estimates with area in m².\n\n"
        "The result is stored under the returned `job_id` and can be retrieved "
        "later via `GET /segment/{job_id}`.\n\n"
        "**Rate limit:** 5 requests per minute per user."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid image or parameters"},
        429: {"description": "Rate limit exceeded (5 req/min/user)"},
        500: {"model": ErrorResponse, "description": "Inference failure"},
    },
)
async def segment_image(
    request: Request,
    image: UploadFile = File(..., description="Roof image (JPEG/PNG/WebP, max 10 MB)"),
    pixels_per_meter: Optional[float] = Form(
        default=None,
        gt=0,
        description=(
            "Scale calibration: how many pixels correspond to 1 metre. "
            "Include a ruler or known-length reference object in the image "
            "and measure it in pixels. Leave blank to use the server default."
        ),
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SegmentationResponse:
    """
    Main segmentation endpoint.

    Pipeline:
      validate → preprocess → denoise → segment → geometry → material estimate → persist
    """
    # ── Validate MIME type ────────────────────────────────────────────────────
    if image.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type: {image.content_type}. "
                f"Allowed: {settings.ALLOWED_MIME_TYPES}"
            ),
        )

    # ── Read & size-check image bytes ─────────────────────────────────────────
    image_bytes = await image.read()
    size_bytes = len(image_bytes)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image too large: {size_mb:.1f} MB. Maximum: {settings.MAX_IMAGE_SIZE_MB} MB",
        )

    logger.info(
        "Segment request: user=%s file=%r size=%.2fMB type=%s",
        current_user.id,
        image.filename,
        size_mb,
        image.content_type,
    )

    # ── Run segmentation pipeline ─────────────────────────────────────────────
    try:
        service = SegmentationService(model_registry=request.app.state.model_registry)
        result = await service.segment(
            image_bytes=image_bytes,
            pixels_per_meter=pixels_per_meter,
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except InferenceError as exc:
        logger.error("Inference failed for user=%s: %s", current_user.id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Segmentation inference failed. See server logs.",
        )

    # ── Persist result in database ────────────────────────────────────────────
    try:
        JobRepository.create_from_response(
            db,
            user_id=current_user.id,
            result=result,
            image_filename=image.filename,
            image_size_bytes=size_bytes,
            content_type=image.content_type,
            pixels_per_meter=pixels_per_meter,
        )
        logger.info("Job %s persisted for user %s", result.job_id, current_user.id)
    except Exception as exc:
        # Log but don't fail the request — the client already has their result
        logger.error("Failed to persist job %s: %s", result.job_id, exc, exc_info=True)

    return result


# ── GET /{job_id} — Retrieve a stored result ──────────────────────────────────

@router.get(
    "/{job_id}",
    response_model=SegmentationResponse,
    summary="Retrieve a previous segmentation result by job_id",
    description=(
        "Returns the stored segmentation result for the given `job_id`. "
        "Only the user who created the job can retrieve it. "
        "Jobs expire after 7 days (configurable via `JOB_RESULT_TTL_DAYS`)."
    ),
    responses={
        404: {"description": "Job not found, expired, or belongs to another user."},
    },
)
def get_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SegmentationResponse:
    """
    Retrieve a completed segmentation job by ID.

    Ownership is enforced: only the user who submitted the job can fetch it.
    This prevents users from enumerating or stealing other users' results.
    """
    job = JobRepository.get_by_id_and_user(db, job_id=job_id, user_id=current_user.id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found, has expired, or does not belong to your account.",
        )

    result = JobRepository.result_from_job(job)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' result data is unavailable.",
        )

    return result


# ── GET / — List user's jobs ──────────────────────────────────────────────────

@router.get(
    "/",
    response_model=List[SegmentationJobSummary],
    summary="List your segmentation jobs",
    description="Returns the most recent jobs for the authenticated user, newest first.",
)
def list_jobs(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[SegmentationJobSummary]:
    """Return a paginated list of the current user's segmentation jobs."""
    jobs = JobRepository.list_by_user(db, user_id=current_user.id, limit=limit, offset=offset)
    return [
        SegmentationJobSummary(
            job_id=j.id,
            status=j.status,
            image_filename=j.image_filename,
            inference_time_ms=j.inference_time_ms,
            model_version=j.model_version,
            created_at=j.created_at,
            expires_at=j.expires_at,
        )
        for j in jobs
    ]



