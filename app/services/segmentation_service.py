"""
SegmentationService — end-to-end inference orchestrator.

Pipeline:
  bytes → preprocess → denoise → segment → postprocess → save images → response

All heavy work runs in a ThreadPoolExecutor so the asyncio event loop
stays free for accepting new connections while inference is in progress.
"""
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.core.exceptions import InferenceError, ImageValidationError
from app.domain.schemas.segmentation import (
    CeilingMaterialType,
    DetectedRegion,
    MaterialEstimate,
    RoofType,
    SegmentationResponse,
)
from app.infrastructure.ml.model_registry import ModelRegistry
from app.infrastructure.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from app.services.geometry_service import GeometryService
from app.services.material_estimator import MaterialEstimator

logger = logging.getLogger(__name__)

# ── Class index → material type (must match your training label encoding) ──────
CLASS_TO_MATERIAL = {
    0: None,                              # background
    1: CeilingMaterialType.GYPSUM_BOARD,
    2: CeilingMaterialType.MODULAR_GRID,
    3: CeilingMaterialType.SUSPENDED,
    4: CeilingMaterialType.PVC_PANEL,
    5: CeilingMaterialType.WOOD_PANEL,
}

RESULTS_DIR = Path("results")


class SegmentationService:
    """
    Stateless application service. Safe to instantiate once and reuse,
    or create per-request — no mutable state after __init__.
    """

    def __init__(self, model_registry: ModelRegistry):
        self.registry    = model_registry
        self.preprocessor = ImagePreprocessor()
        self.geometry    = GeometryService()
        self.estimator   = MaterialEstimator()

    async def segment(
        self,
        image_bytes: bytes,
        pixels_per_meter: Optional[float] = None,
    ) -> SegmentationResponse:
        """
        Async entry point called by the API router.
        All blocking CPU/GPU work is offloaded to a thread-pool executor.
        """
        job_id = str(uuid.uuid4())
        ppm    = pixels_per_meter or settings.DEFAULT_PIXELS_PER_METER
        t0     = time.perf_counter()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_pipeline_sync,
            image_bytes, ppm, job_id,
        )

        result.inference_time_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("Job %s done in %.1f ms", job_id, result.inference_time_ms)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronous pipeline — runs inside ThreadPoolExecutor
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pipeline_sync(
        self,
        image_bytes: bytes,
        pixels_per_meter: float,
        job_id: str,
    ) -> SegmentationResponse:
        warnings: list[str] = []

        # ── Step 1: Preprocess ────────────────────────────────────────────────
        try:
            input_tensor, original_size = self.preprocessor.process(image_bytes)
        except ImageValidationError:
            raise
        except Exception as exc:
            raise InferenceError(f"Preprocessing failed: {exc}") from exc

        input_tensor = input_tensor.to(self.registry.device)   # [1,3,H,W]

        # ── Step 2: Denoise (your optimized model) ────────────────────────────
        try:
            with torch.no_grad():
                denoised_tensor = self.registry.denoiser(input_tensor)  # [1,3,H,W]
        except Exception as exc:
            raise InferenceError(f"Denoiser forward pass failed: {exc}") from exc

        # ── Step 3: Segmentation U-Net forward pass ───────────────────────────
        try:
            with torch.no_grad():
                logits = self.registry.segmenter(denoised_tensor)   # [1,C,H,W]
        except Exception as exc:
            raise InferenceError(f"Segmenter forward pass failed: {exc}") from exc

        # ── Step 4: Decode logits → mask + confidence map ─────────────────────
        probs          = F.softmax(logits, dim=1)                # [1,C,H,W]
        pred_class     = torch.argmax(probs, dim=1).squeeze(0)   # [H,W]
        max_confidence = probs.max(dim=1).values.squeeze(0)      # [H,W]

        mask_np = pred_class.cpu().numpy().astype(np.uint8)      # [H,W] uint8
        conf_np = max_confidence.cpu().numpy().astype(np.float32) # [H,W]

        # ── Step 5: Geometry analysis ─────────────────────────────────────────
        regions, roof_type, class_map = self.geometry.analyze_mask(
            mask_np         = mask_np,
            conf_map        = conf_np,
            pixels_per_meter = pixels_per_meter,
            min_area_px     = settings.MIN_REGION_AREA_PX,
        )

        # ── Step 6: Confidence warning ─────────────────────────────────────────
        foreground = mask_np > 0
        if foreground.any():
            overall_conf = float(conf_np[foreground].mean())
            if overall_conf < 0.60:
                warnings.append(
                    f"Low segmentation confidence ({overall_conf:.0%}). "
                    "Retake photo with better lighting or less obstruction."
                )
        else:
            warnings.append("No roof area detected. Ensure the image shows a clear roof view.")

        # ── Step 7: Material estimation ───────────────────────────────────────
        materials = self.estimator.estimate(regions, class_map)

        total_area = round(sum(r.area_m2 for r in regions), 3)
        if 0 < total_area < 1.0:
            warnings.append("Estimated area is very small (<1 m²). Verify scale reference.")

        # ── Step 8: Render and save mask + overlay images ─────────────────────
        mask_url, overlay_url = self._save_result_images(
            job_id        = job_id,
            mask_np       = mask_np,
            image_bytes   = image_bytes,
            original_size = original_size,
            input_size    = settings.INPUT_SIZE,
        )

        return SegmentationResponse(
            job_id               = job_id,
            roof_type            = roof_type,
            total_area_m2        = total_area,
            regions              = regions,
            materials            = materials,
            segmentation_mask_url = mask_url,
            overlay_image_url    = overlay_url,
            inference_time_ms    = 0.0,          # filled by async caller
            model_version        = self.registry.model_version,
            warnings             = warnings,
            pixels_per_meter     = pixels_per_meter,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Image rendering helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _save_result_images(
        self,
        job_id: str,
        mask_np: np.ndarray,
        image_bytes: bytes,
        original_size: Tuple[int, int],
        input_size: int,
    ) -> Tuple[str, str]:
        """
        Saves the coloured segmentation mask and the overlay image to disk.
        Returns (mask_url, overlay_url) as relative URL paths.
        """
        out_dir = RESULTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Coloured mask (saved at model resolution, resized to original)
        colored_mask_model = self.geometry.render_colored_mask(mask_np)  # [H,W,3] BGR
        orig_w, orig_h     = original_size
        colored_mask_orig  = cv2.resize(
            colored_mask_model, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        mask_path = out_dir / "mask.png"
        cv2.imwrite(str(mask_path), colored_mask_orig)

        # 2. Overlay: blend mask onto original image
        original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_bgr = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        overlay_bgr  = self.geometry.create_overlay(original_bgr, mask_np)
        overlay_path = out_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), overlay_bgr)

        mask_url    = f"/results/{job_id}/mask.png"
        overlay_url = f"/results/{job_id}/overlay.png"
        return mask_url, overlay_url
