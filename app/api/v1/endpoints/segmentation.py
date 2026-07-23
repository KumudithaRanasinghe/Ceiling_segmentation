"""
Segmentation Endpoint.

POST /api/v1/segment/
  Accepts a multipart image upload.
  Returns full segmentation results + material estimates.

GET  /api/v1/segment/{job_id}
  Retrieve a cached result by job ID.

Design decisions:
  • UploadFile.read() is awaited (async I/O)
  • Heavy inference is off-loaded to thread pool in SegmentationService
  • ModelRegistry injected via request.app.state (no globals)
  • All validation is Pydantic — no manual if/else type checks
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.core.exceptions import ImageValidationError, InferenceError
from app.domain.models.user import User
from app.domain.schemas.segmentation import ErrorResponse, SegmentationResponse
from app.services.segmentation_service import SegmentationService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment a roof area image",
    description=(
        "Upload a roof image. The system runs noise reduction, semantic segmentation, "
        "geometry analysis, and returns ceiling material estimates with area in m²."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid image or parameters"},
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
    current_user: User = Depends(get_current_user),   # ← JWT auth required
) -> SegmentationResponse:
    """
    Main segmentation endpoint.

    Pipeline:
      validate → preprocess → denoise → segment → geometry → material estimate
    """
    # ── Validate file metadata ────────────────────────────────────────────────
    if image.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {image.content_type}. "
                   f"Allowed: {settings.ALLOWED_MIME_TYPES}",
        )

    # ── Read image bytes ──────────────────────────────────────────────────────
    image_bytes = await image.read()
    size_mb = len(image_bytes) / (1024 * 1024)

    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image too large: {size_mb:.1f} MB. Maximum: {settings.MAX_IMAGE_SIZE_MB} MB",
        )

    logger.info(
        f"Segment request: file={image.filename!r} "
        f"size={size_mb:.2f}MB type={image.content_type}"
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
        logger.error(f"Inference failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Segmentation inference failed. See server logs.",
        )

    return result


@router.get(
    "/{job_id}",
    response_model=SegmentationResponse,
    summary="Retrieve a previous segmentation result",
)
async def get_result(
    job_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),   # ← JWT auth required
) -> SegmentationResponse:
    """
    Retrieve a cached segmentation result by job_id.
    Useful for async flows where mobile app polls for completion.
    """
    # In production, fetch from Redis or database.
    # Placeholder 404 for now.
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Job {job_id!r} not found or has expired.",
    )
