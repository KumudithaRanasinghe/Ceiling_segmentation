"""
GeometryService — deterministic computational geometry.

Core principle: every measurement traces back to pixel counting
via cv2.contourArea(), then converts through a fixed scale factor.
No neural net guessing for measurements — only for segmentation masks.

Pipeline per class:
  binary mask → morphological cleanup → contour detection
  → area (px²) → area (m²) → DetectedRegion
"""
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.domain.schemas.responses import CeilingMaterialType, DetectedRegion, RoofType

logger = logging.getLogger(__name__)

# BGR colors for mask visualization — one per class index
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (30,  30,  30),   # background — near-black
    1: (255, 120,  80),  # gypsum board — blue-ish
    2: (80,  220, 120),  # modular grid — green
    3: (80,  120, 255),  # suspended    — red
    4: (255, 200,  80),  # pvc panel    — amber
    5: (200,  80, 255),  # wood panel   — violet
}

# Class index → material type (mirrors segmentation_service mapping)
CLASS_TO_MATERIAL: Dict[int, CeilingMaterialType] = {
    1: CeilingMaterialType.GYPSUM_BOARD,
    2: CeilingMaterialType.MODULAR_GRID,
    3: CeilingMaterialType.SUSPENDED,
    4: CeilingMaterialType.PVC_PANEL,
    5: CeilingMaterialType.WOOD_PANEL,
}


class GeometryService:

    def analyze_mask(
        self,
        mask_np: np.ndarray,           # [H, W] uint8 — class index per pixel
        conf_map: np.ndarray,          # [H, W] float32 — max softmax conf per pixel
        pixels_per_meter: Optional[float] = None,  # calibration: px in 1 m in ORIGINAL image space (None = default)
        original_size: Tuple[int, int] = (512, 512),  # (width, height) of original image
        min_area_px: int = 500,        # reject regions smaller than this
    ) -> Tuple[List[DetectedRegion], RoofType, Dict[int, CeilingMaterialType]]:
        """
        Returns:
            regions:   DetectedRegion list (sorted area desc)
            roof_type: inferred RoofType
            class_map: {region_id: CeilingMaterialType} for the estimator

        Geometric scale calibration:
            1. If `pixels_per_meter` is provided by user (calibrated on original image):
               scale_factor = min(mask_w / orig_w, mask_h / orig_h)
               ppm_mask = pixels_per_meter * scale_factor
            2. If uncalibrated (`pixels_per_meter` is None):
               Uses `settings.DEFAULT_PIXELS_PER_METER` (34.0 px/m in 512x512 space),
               representing a standard residential building span (~12-16m across ~400-450px).
               This ensures realistic room areas (10–45 m²) and house areas (60–150 m²).
        """
        mask_h, mask_w = mask_np.shape
        orig_w, orig_h = original_size

        scale_w = mask_w / orig_w if orig_w > 0 else 1.0
        scale_h = mask_h / orig_h if orig_h > 0 else 1.0
        scale_factor = min(scale_w, scale_h)

        if pixels_per_meter is not None and pixels_per_meter > 0:
            ppm_mask = pixels_per_meter * scale_factor
            ppm_orig = pixels_per_meter
        else:
            ppm_mask = settings.DEFAULT_PIXELS_PER_METER
            ppm_orig = round(ppm_mask / scale_factor, 1) if scale_factor > 0 else ppm_mask

        ppm2   = ppm_mask ** 2   # px² per m²
        logger.info(
            "Geometry calibration: ppm_orig=%.2f scale=%.4f ppm_mask=%.2f ppm2=%.2f",
            ppm_orig, scale_factor, ppm_mask, ppm2,
        )
        regions: List[DetectedRegion]          = []
        class_map: Dict[int, CeilingMaterialType] = {}
        region_id = 0

        for class_idx, material_type in CLASS_TO_MATERIAL.items():
            # Extract binary mask for this class
            binary = ((mask_np == class_idx).astype(np.uint8)) * 255

            # ── Morphological cleanup ─────────────────────────────────────
            # CLOSE: fill small holes inside regions
            # OPEN:  remove isolated noise speckles
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k7)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3)

            # ── Contour detection ─────────────────────────────────────────
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area_px = int(cv2.contourArea(contour))
                if area_px < min_area_px:
                    continue

                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Centroid via image moments (exact, not bbox centre)
                M  = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                # Mean confidence inside this contour
                local_mask = np.zeros(mask_np.shape, dtype=np.uint8)
                cv2.drawContours(local_mask, [contour], -1, 255, cv2.FILLED)
                region_conf = float(conf_map[local_mask > 0].mean())

                # Convert pixels² → m²
                area_m2 = area_px / ppm2

                # Sanity gate: reject architecturally impossible measurements
                # Upper limit: largest buildings are ~500,000 m² floor area;
                # a single ceiling region >50,000 m² is implausible.
                # Lower limit: anything below 0.01 m² is sub-10cm noise.
                if area_m2 > 50_000:
                    logger.warning(
                        "Region %d area %.1f m² exceeds physical limit — "
                        "check pixels_per_meter calibration (current ppm_orig=%.1f, ppm_mask=%.2f).",
                        region_id, area_m2, pixels_per_meter, ppm_mask,
                    )
                    continue
                if area_m2 < 0.01:
                    logger.debug(
                        "Region %d area %.4f m² is sub-10cm noise — skipping.",
                        region_id, area_m2,
                    )
                    continue

                regions.append(DetectedRegion(
                    region_id    = region_id,
                    area_pixels  = area_px,
                    area_m2      = round(area_m2, 4),
                    confidence   = round(region_conf, 4),
                    bounding_box = [int(x), int(y), int(w), int(h)],
                    centroid     = [round(cx, 1), round(cy, 1)],
                ))
                class_map[region_id] = material_type
                region_id += 1

        regions.sort(key=lambda r: r.area_m2, reverse=True)
        roof_type = self._infer_roof_type(regions)
        return regions, roof_type, class_map

    # ── Visualization helpers ─────────────────────────────────────────────────

    def render_colored_mask(self, mask_np: np.ndarray) -> np.ndarray:
        """
        [H, W] uint8 class map → [H, W, 3] BGR uint8 coloured mask.
        Fast vectorised assignment (no Python loop over pixels).
        """
        h, w   = mask_np.shape
        output = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, bgr in CLASS_COLORS_BGR.items():
            output[mask_np == class_idx] = bgr
        return output

    def create_overlay(
        self,
        original_bgr: np.ndarray,
        mask_np: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Alpha-blend the coloured mask onto the original image.
        Resizes mask to match original if model output size differs.
        """
        colored = self.render_colored_mask(mask_np)
        if colored.shape[:2] != original_bgr.shape[:2]:
            oh, ow = original_bgr.shape[:2]
            colored = cv2.resize(colored, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # Only blend non-background pixels — keep background transparent
        bg_mask = (mask_np == 0)
        if bg_mask.shape[:2] != original_bgr.shape[:2]:
            bg_mask = cv2.resize(
                bg_mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        blended   = cv2.addWeighted(original_bgr, 1 - alpha, colored, alpha, 0)
        blended[bg_mask] = original_bgr[bg_mask]   # restore background pixels
        return blended

    # ── Private helpers ───────────────────────────────────────────────────────

    def _infer_roof_type(self, regions: List[DetectedRegion]) -> RoofType:
        """
        Heuristic classification from dominant region geometry.
        Real implementation: add a dedicated classifier head to U-Net.
        """
        if not regions:
            return RoofType.UNKNOWN

        largest  = regions[0]
        _, _, w, h = largest.bounding_box
        aspect   = (w / h) if h > 0 else 1.0
        n        = len(regions)

        if n == 1:
            return RoofType.HIP  if (0.75 < aspect < 1.33) else RoofType.FLAT
        if n == 2:
            return RoofType.GABLE
        return RoofType.SHED
