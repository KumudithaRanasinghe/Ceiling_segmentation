"""
MaterialEstimator — converts segmented regions into purchase-ready quantities.

Receives the class_map {region_id: CeilingMaterialType} from GeometryService
so every region is correctly attributed to its material type.

Applies:
  - Aggregation:  sum area per material type
  - Waste buffer: configured MATERIAL_OVERHEAD_FACTOR (default 10%)
  - Unit count:   ceil(gross_area / standard_panel_size)
"""
import math
import logging
from collections import defaultdict
from typing import Dict, List

from app.core.config import settings
from app.domain.schemas.segmentation import (
    CeilingMaterialType,
    DetectedRegion,
    MaterialEstimate,
)

logger = logging.getLogger(__name__)

# Standard panel / sheet sizes in m² (Sri Lankan / global market norms)
PANEL_SIZE_M2: Dict[CeilingMaterialType, float] = {
    CeilingMaterialType.GYPSUM_BOARD: 2.88,   # 1200 × 2400 mm
    CeilingMaterialType.MODULAR_GRID: 0.36,   # 600 × 600 mm
    CeilingMaterialType.SUSPENDED:    0.36,   # 600 × 600 mm
    CeilingMaterialType.PVC_PANEL:    0.24,   # 200 × 1200 mm
    CeilingMaterialType.WOOD_PANEL:   0.60,   # 600 × 1000 mm
    CeilingMaterialType.UNKNOWN:      1.00,   # generic
}


class MaterialEstimator:

    def estimate(
        self,
        regions: List[DetectedRegion],
        class_map: Dict[int, CeilingMaterialType],
    ) -> List[MaterialEstimate]:
        """
        Args:
            regions:   all detected regions with area_m2 and confidence
            class_map: {region_id → CeilingMaterialType}

        Returns:
            One MaterialEstimate per material type found, sorted by area desc.
        """
        if not regions:
            return []

        # Aggregate net area and confidence samples per material type
        area_by_type: Dict[CeilingMaterialType, float]      = defaultdict(float)
        conf_by_type: Dict[CeilingMaterialType, List[float]] = defaultdict(list)

        for region in regions:
            mat = class_map.get(region.region_id, CeilingMaterialType.UNKNOWN)
            area_by_type[mat] += region.area_m2
            conf_by_type[mat].append(region.confidence)

        overhead = settings.MATERIAL_OVERHEAD_FACTOR
        estimates: List[MaterialEstimate] = []

        for mat_type, net_area in area_by_type.items():
            gross_area  = round(net_area * overhead, 3)
            panel_size  = PANEL_SIZE_M2.get(mat_type, 1.0)
            unit_count  = math.ceil(gross_area / panel_size)
            mean_conf   = sum(conf_by_type[mat_type]) / len(conf_by_type[mat_type])

            estimates.append(MaterialEstimate(
                material_type          = mat_type,
                quantity_m2            = round(net_area, 3),
                quantity_with_waste_m2 = gross_area,
                unit_count             = unit_count,
                unit_size_m2           = panel_size,
                confidence             = round(mean_conf, 4),
            ))
            logger.debug(
                "Material %s: %.2f m² net → %.2f m² gross → %d units",
                mat_type, net_area, gross_area, unit_count,
            )

        estimates.sort(key=lambda e: e.quantity_m2, reverse=True)
        return estimates
