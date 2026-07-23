"""
Test suite — covers the full request lifecycle without needing real model weights.

Strategy: mock the ModelRegistry so tests run on CPU without .pt files.
The denoiser and segmenter are replaced with identity-pass stubs.
Everything else (preprocessing, geometry, material estimation, API routing)
runs exactly as in production.
"""
import io
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_png_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a minimal valid RGB PNG in memory."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), "RGB"
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_stub_registry(num_classes: int = 6, input_size: int = 512) -> MagicMock:
    """
    Build a ModelRegistry stub whose denoiser and segmenter are
    real nn.Modules (identity + constant output) so tensor shapes are correct.
    """
    import torch.nn as nn

    class IdentityDenoiser(nn.Module):
        def forward(self, x):
            return x

    class ConstantSegmenter(nn.Module):
        """Always predicts class 1 (GYPSUM_BOARD) for the entire image."""
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x):
            b, _, h, w = x.shape
            logits = torch.zeros(b, self.num_classes, h, w)
            logits[:, 1, :, :] = 10.0   # force class 1 everywhere
            return logits

    registry = MagicMock()
    registry.device      = torch.device("cpu")
    registry.model_version = "test-1.0"
    registry.denoiser    = IdentityDenoiser()
    registry.segmenter   = ConstantSegmenter(num_classes)
    return registry


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestImagePreprocessor:
    def test_valid_jpeg_returns_correct_shape(self):
        from app.infrastructure.preprocessing.image_preprocessor import ImagePreprocessor
        pp = ImagePreprocessor(target_size=512)
        png = make_png_bytes(640, 480)
        tensor, orig_size = pp.process(png)
        assert tensor.shape == (1, 3, 512, 512)
        assert orig_size == (640, 480)

    def test_float32_normalized_range(self):
        from app.infrastructure.preprocessing.image_preprocessor import ImagePreprocessor
        pp = ImagePreprocessor(target_size=512)
        png = make_png_bytes(128, 128)
        tensor, _ = pp.process(png)
        # After ImageNet normalization values can go below 0 or above 1
        assert tensor.dtype == torch.float32
        assert tensor.shape[1] == 3

    def test_too_small_image_raises(self):
        from app.infrastructure.preprocessing.image_preprocessor import ImagePreprocessor
        from app.core.exceptions import ImageValidationError
        pp = ImagePreprocessor()
        tiny = make_png_bytes(32, 32)
        with pytest.raises(ImageValidationError, match="too small"):
            pp.process(tiny)

    def test_corrupt_bytes_raises(self):
        from app.infrastructure.preprocessing.image_preprocessor import ImagePreprocessor
        from app.core.exceptions import ImageValidationError
        pp = ImagePreprocessor()
        with pytest.raises(ImageValidationError):
            pp.process(b"not_an_image_at_all")


# ── Geometry tests ────────────────────────────────────────────────────────────

class TestGeometryService:
    def test_empty_mask_returns_no_regions(self):
        from app.services.geometry_service import GeometryService
        gs = GeometryService()
        mask = np.zeros((512, 512), dtype=np.uint8)
        conf = np.ones((512, 512), dtype=np.float32) * 0.9
        regions, roof_type, class_map = gs.analyze_mask(mask, conf, pixels_per_meter=100.0)
        assert regions == []
        assert roof_type.value == "unknown"

    def test_full_class1_mask_returns_one_region(self):
        from app.services.geometry_service import GeometryService
        gs = GeometryService()
        mask = np.ones((512, 512), dtype=np.uint8)   # all class 1
        conf = np.ones((512, 512), dtype=np.float32) * 0.95
        regions, roof_type, class_map = gs.analyze_mask(mask, conf, pixels_per_meter=100.0)
        assert len(regions) >= 1
        assert regions[0].area_m2 > 0
        assert class_map[regions[0].region_id].value == "gypsum_board"

    def test_area_conversion_is_consistent(self):
        from app.services.geometry_service import GeometryService
        gs = GeometryService()
        # 200×200 block of class 1 in a 512×512 mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[156:356, 156:356] = 1
        conf = np.ones((512, 512), dtype=np.float32) * 0.9
        regions, _, _ = gs.analyze_mask(mask, conf, pixels_per_meter=100.0)
        # Expected: 200×200 px / (100 px/m)² = 4.0 m²
        assert len(regions) == 1
        assert abs(regions[0].area_m2 - 4.0) < 0.5   # allow for morphological effects

    def test_render_colored_mask_shape(self):
        from app.services.geometry_service import GeometryService
        gs = GeometryService()
        mask = np.zeros((64, 64), dtype=np.uint8)
        colored = gs.render_colored_mask(mask)
        assert colored.shape == (64, 64, 3)
        assert colored.dtype == np.uint8


# ── Material estimator tests ──────────────────────────────────────────────────

class TestMaterialEstimator:
    def _make_region(self, region_id, area_m2, confidence=0.9):
        from app.domain.schemas.responses import DetectedRegion
        return DetectedRegion(
            region_id=region_id, area_pixels=1000,
            area_m2=area_m2, confidence=confidence,
            bounding_box=[0, 0, 100, 100], centroid=[50.0, 50.0],
        )

    def test_empty_input_returns_empty(self):
        from app.services.material_estimator import MaterialEstimator
        est = MaterialEstimator()
        assert est.estimate([], {}) == []

    def test_waste_buffer_applied(self):
        from app.services.material_estimator import MaterialEstimator
        from app.domain.schemas.responses import CeilingMaterialType
        est = MaterialEstimator()
        region  = self._make_region(0, 10.0)
        result  = est.estimate([region], {0: CeilingMaterialType.GYPSUM_BOARD})
        assert len(result) == 1
        assert result[0].quantity_with_waste_m2 > result[0].quantity_m2

    def test_unit_count_is_ceiling_division(self):
        import math
        from app.services.material_estimator import MaterialEstimator, PANEL_SIZE_M2
        from app.domain.schemas.responses import CeilingMaterialType
        est = MaterialEstimator()
        net = 7.5
        region = self._make_region(0, net)
        result = est.estimate([region], {0: CeilingMaterialType.MODULAR_GRID})
        panel  = PANEL_SIZE_M2[CeilingMaterialType.MODULAR_GRID]
        # waste factor applied then ceil
        expected_units = math.ceil((net * 1.10) / panel)
        assert result[0].unit_count == expected_units


# ── Full API integration test ─────────────────────────────────────────────────

class TestSegmentationEndpoint:
    @pytest.fixture
    def client(self):
        """TestClient with a stubbed ModelRegistry injected via app.state and a mock auth token + DB user."""
        from app.main import app
        from app.core.security import create_access_token
        from app.infrastructure.database.session import init_db, get_db
        from app.domain.repositories.user_repository import UserRepository

        init_db()
        db = next(get_db())
        user = UserRepository.get_by_id(db, "test-user-id")
        if not user:
            UserRepository.create(
                db,
                email="testuser@example.com",
                username="testuser",
                full_name=None,
                plain_password="Password123!",
            )
            # Ensure ID is test-user-id
            u = UserRepository.get_by_username(db, "testuser")
            u.id = "test-user-id"
            db.commit()
        db.close()

        registry = make_stub_registry()
        app.state.model_registry = registry
        token = create_access_token(subject="test-user-id", roles=["user"])
        with TestClient(app, raise_server_exceptions=True) as c:
            c.headers.update({"Authorization": f"Bearer {token}"})
            yield c

    def test_health_live(self, client):
        r = client.get("/api/v1/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_segment_valid_png_returns_200(self, client, tmp_path):
        png_bytes = make_png_bytes(512, 512)
        r = client.post(
            "/api/v1/segment/",
            files={"image": ("roof.png", png_bytes, "image/png")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "job_id" in body
        assert "total_area_m2" in body
        assert isinstance(body["materials"], list)
        assert isinstance(body["warnings"], list)

    def test_segment_wrong_mime_returns_422(self, client):
        r = client.post(
            "/api/v1/segment/",
            files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 422

    def test_segment_with_pixels_per_meter_param(self, client):
        png_bytes = make_png_bytes(512, 512)
        r = client.post(
            "/api/v1/segment/",
            files={"image": ("roof.png", png_bytes, "image/png")},
            data={"pixels_per_meter": "150.0"},
        )
        assert r.status_code == 200
        assert r.json()["pixels_per_meter"] == 150.0

    def test_segment_response_schema_complete(self, client):
        """All required fields must be present in the response."""
        required = {
            "job_id", "status", "roof_type", "total_area_m2", "regions",
            "materials", "segmentation_mask_url", "overlay_image_url",
            "inference_time_ms", "model_version", "warnings", "pixels_per_meter",
        }
        png_bytes = make_png_bytes(512, 512)
        r = client.post(
            "/api/v1/segment/",
            files={"image": ("roof.png", png_bytes, "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        missing = required - set(body.keys())
        assert not missing, f"Missing fields in response: {missing}"
