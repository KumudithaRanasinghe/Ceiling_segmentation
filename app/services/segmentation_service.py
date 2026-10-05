"""
SegmentationService — end-to-end inference orchestrator (Dual-Model Pipeline).

Pipeline:
  bytes → preprocess → denoise →
    ┌─ V1: cluster segmentation (rooms, kitchens, etc.)   ─┐  (parallel)
    └─ V2: floor vs background binary mask                 ─┘
  → combine: mask V1 clusters with V2 interior mask → postprocess → response

Both models run in PARALLEL via concurrent.futures.ThreadPoolExecutor.
V2's binary mask (inside floor vs background) gates V1's cluster output,
so cluster labels only survive where V2 says "inside floor".

All heavy work runs in a ThreadPoolExecutor so the asyncio event loop
stays free for accepting new connections while inference is in progress.
"""
import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import cv2
import io
import numpy as np
import torch
from torchvision import transforms  # REMOVE AFTER DEBUGGING
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.core.exceptions import InferenceError, ImageValidationError
from app.domain.schemas.responses import (
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
    Stateless application service — dual-model inference pipeline.

    V1 detects cluster areas (rooms, kitchens, floor zones).
    V2 detects inside-floor vs background (binary).

    Both run in parallel. V2's foreground mask gates V1's clusters:
    only clusters inside the detected floor area survive.
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
        t0     = time.perf_counter()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_pipeline_sync,
            image_bytes, pixels_per_meter, job_id,
        )

        result.inference_time_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("Job %s done in %.1f ms", job_id, result.inference_time_ms)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Parallel inference helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_v1_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """V1: Cluster segmentation forward pass → logits [1, C, H, W]"""
        with torch.no_grad():
            return self.registry.segmenter(input_tensor)

    def _run_v2_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """V2: Floor vs background forward pass → logits [1, 2, H, W]"""
        with torch.no_grad():
            return self.registry.segmenter_v2(input_tensor)

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronous pipeline — runs inside ThreadPoolExecutor
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pipeline_sync(
        self,
        image_bytes: bytes,
        pixels_per_meter: Optional[float],
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

        # ── Step 2: Denoise (optional) ────────────────────────────────────────
        try:
            with torch.no_grad():
                denoised_tensor = self.registry.denoiser(input_tensor)  # [1,3,H,W]
                # REMOVE LATER
                denoised_tensor_export = transforms.ToPILImage()(torch.clamp(denoised_tensor, 0, 1).squeeze(0).cpu())
                denoised_tensor_export.save(f"results/denoised_debug.png")
                # REMOVE LATER
        except Exception as exc:
            raise InferenceError(f"Denoiser forward pass failed: {exc}") from exc

        # ── Step 3: PARALLEL inference — V1 + V2 ─────────────────────────────
        #   V1: cluster segmentation (rooms, kitchens, floor areas)
        #   V2: binary floor vs background detection
        #   Both share the same preprocessed tensor, run concurrently.
        try:
            if self.registry.has_v2:
                # Run V1 and V2 in parallel using threads
                with ThreadPoolExecutor(max_workers=2) as pool:
                    future_v1 = pool.submit(self._run_v1_inference, denoised_tensor)
                    future_v2 = pool.submit(self._run_v2_inference, denoised_tensor)

                    v1_logits = future_v1.result()  # [1, NUM_CLASSES, H, W]
                    v2_logits = future_v2.result()  # [1, 2, H, W]

                logger.info("Parallel inference complete: V1 + V2")
            else:
                # V2 not available — fallback to V1-only pipeline
                v1_logits = self._run_v1_inference(denoised_tensor)
                v2_logits = None
                logger.info("V1-only inference (V2 model not loaded)")

        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc

        # ── Step 4: Decode V1 logits → cluster mask + confidence ──────────────
        v1_probs       = F.softmax(v1_logits, dim=1)               # [1,C,H,W]
        v1_pred_class  = torch.argmax(v1_probs, dim=1).squeeze(0)  # [H,W]
        v1_confidence  = v1_probs.max(dim=1).values.squeeze(0)     # [H,W]

        mask_np = v1_pred_class.cpu().numpy().astype(np.uint8)     # [H,W] uint8
        conf_np = v1_confidence.cpu().numpy().astype(np.float32)   # [H,W]

        # ── Step 5: Decode V2 logits → binary floor mask ──────────────────────
        v2_floor_mask_np = None
        if v2_logits is not None:
            v2_probs      = F.softmax(v2_logits, dim=1)               # [1,C,H,W]
            v2_pred_class = torch.argmax(v2_probs, dim=1).squeeze(0)  # [H,W]
            v2_floor_mask_np = v2_pred_class.cpu().numpy().astype(np.uint8)  # [H,W]

            # V2 class mapping: 0=background, any class > 0 = inside floor area
            # Create binary mask: True where V2 says "inside floor"
            floor_interior = (v2_floor_mask_np > 0)

            logger.info(
                "V2 floor detection: %d/%d pixels = %.1f%% inside floor",
                floor_interior.sum(),
                floor_interior.size,
                100.0 * floor_interior.sum() / floor_interior.size,
            )

        # ── Step 6: COMBINE — gate V1 clusters with V2 floor mask ─────────────
        #   Set V1 cluster labels to 0 (background) wherever V2 says "background".
        #   This ensures only clusters inside the detected floor area survive.
        if v2_floor_mask_np is not None:
            background_mask = (v2_floor_mask_np == 0)  # V2 background
            mask_np[background_mask] = 0               # Zero out V1 clusters outside floor
            conf_np[background_mask] = 0.0             # Zero confidence for background

            # Count removed pixels
            removed = background_mask.sum()
            total   = background_mask.size
            logger.info(
                "Combined masks: V2 removed %d/%d background pixels (%.1f%%) from V1 clusters",
                removed, total, 100.0 * removed / total,
            )

        # ── Step 7: Geometry analysis (on combined mask) ──────────────────────
        regions, roof_type, class_map = self.geometry.analyze_mask(
            mask_np          = mask_np,
            conf_map         = conf_np,
            pixels_per_meter = pixels_per_meter,
            original_size    = original_size,   # (w, h) before letterbox resize
            min_area_px      = settings.MIN_REGION_AREA_PX,
        )

        # ── Step 8: Confidence warning ────────────────────────────────────────
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

        # ── Step 9: Material estimation ───────────────────────────────────────
        materials = self.estimator.estimate(regions, class_map)

        total_area = round(sum(r.area_m2 for r in regions), 3)
        if 0 < total_area < 1.0:
            warnings.append("Estimated area is very small (<1 m²). Verify scale reference.")

        # ── Step 10: Render and save mask + overlay + V2 debug images ─────────
        mask_url, overlay_url = self._save_result_images(
            job_id        = job_id,
            mask_np       = mask_np,
            v2_mask_np    = v2_floor_mask_np,
            image_bytes   = image_bytes,
            original_size = original_size,
            input_size    = settings.INPUT_SIZE,
        )

        orig_w, orig_h = original_size
        scale_factor = min(settings.INPUT_SIZE / orig_w, settings.INPUT_SIZE / orig_h) if (orig_w > 0 and orig_h > 0) else 1.0
        if pixels_per_meter is not None and pixels_per_meter > 0:
            effective_ppm = float(pixels_per_meter)
        else:
            effective_ppm = round(settings.DEFAULT_PIXELS_PER_METER / scale_factor, 1) if scale_factor > 0 else float(settings.DEFAULT_PIXELS_PER_METER)

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
            pixels_per_meter     = effective_ppm,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Image rendering helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _save_result_images(
        self,
        job_id: str,
        mask_np: np.ndarray,
        v2_mask_np: Optional[np.ndarray],
        image_bytes: bytes,
        original_size: Tuple[int, int],
        input_size: int,
    ) -> Tuple[str, str]:
        """
        Saves the coloured segmentation mask, overlay image, and V2 debug
        images to disk. Returns (mask_url, overlay_url) as relative URL paths.
        """
        out_dir = RESULTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        orig_w, orig_h = original_size

        # 1. Coloured combined mask (saved at model resolution, resized to original)
        colored_mask_model = self.geometry.render_colored_mask(mask_np)  # [H,W,3] BGR
        colored_mask_orig  = cv2.resize(
            colored_mask_model, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        mask_path = out_dir / "mask.png"
        cv2.imwrite(str(mask_path), colored_mask_orig)

        # 2. Overlay: blend combined mask onto original image
        original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_bgr = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        overlay_bgr  = self.geometry.create_overlay(original_bgr, mask_np)
        overlay_path = out_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), overlay_bgr)

        # 3. V2 binary floor mask debug image (green = inside floor, dark = background)
        if v2_mask_np is not None:
            v2_colored = np.zeros((*v2_mask_np.shape, 3), dtype=np.uint8)
            v2_colored[v2_mask_np == 0] = (30, 30, 30)     # background — near-black
            v2_colored[v2_mask_np == 1] = (80, 220, 80)    # inside floor — green
            v2_colored_orig = cv2.resize(
                v2_colored, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            v2_mask_path = out_dir / "v2_floor_mask.png"
            cv2.imwrite(str(v2_mask_path), v2_colored_orig)

            # V2 overlay on original
            v2_overlay = cv2.addWeighted(original_bgr, 0.55, v2_colored_orig, 0.45, 0)
            # Restore background pixels to original
            bg_mask_resized = cv2.resize(
                v2_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            v2_overlay[bg_mask_resized == 0] = original_bgr[bg_mask_resized == 0]
            v2_overlay_path = out_dir / "v2_floor_overlay.png"
            cv2.imwrite(str(v2_overlay_path), v2_overlay)

            logger.info(
                "Saved V2 debug images: %s, %s",
                v2_mask_path, v2_overlay_path,
            )

        mask_url    = f"/results/{job_id}/mask.png"
        overlay_url = f"/results/{job_id}/overlay.png"
        return mask_url, overlay_url
