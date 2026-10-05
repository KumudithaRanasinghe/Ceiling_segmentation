"""

Model Registry — Singleton pattern for ML model lifecycle.

Loads two segmentation models at startup:
  V1 (segmenter)      — plan-area segmentation into rooms, kitchens, floor areas
  V2 (segmenter_v2)   — binary inside-floor vs background detection

YOUR MODEL FILES:
  V1: app/domain/models/V1/best_model_optimized.pth
  V2: app/domain/models/V2/best_model_optimized (1).pth

HOW LOADING WORKS:
  The registry inspects the checkpoint dictionary keys to handle all common
  PyTorch save formats automatically:
    torch.save(model, path)                     → raw model object
    torch.save(model.state_dict(), path)        → raw state_dict
    torch.save({"model_state_dict": ...}, path) → training checkpoint
    torch.save({"state_dict": ...}, path)       → Lightning checkpoint
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn

from app.core.config import settings
from app.infrastructure.ml.unet import UNet

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Loads all models once at application startup (via FastAPI lifespan).
    Every request reads from the already-loaded in-memory objects.
    Zero per-request loading.
    """

    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self._segmenter: Optional[nn.Module] = None
        self._segmenter_v2: Optional[nn.Module] = None
        self._denoiser: Optional[nn.Module] = None
        self._model_version: str = "1.0.0"

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def segmenter(self) -> nn.Module:
        """V1 — plan area cluster segmentation (rooms, kitchens, etc.)"""
        if self._segmenter is None:
            raise RuntimeError("Segmenter V1 not loaded — check startup logs.")
        return self._segmenter

    @property
    def segmenter_v2(self) -> nn.Module:
        """V2 — binary inside-floor vs background detection."""
        if self._segmenter_v2 is None:
            raise RuntimeError("Segmenter V2 not loaded — check startup logs.")
        return self._segmenter_v2

    @property
    def has_v2(self) -> bool:
        """True if V2 model was successfully loaded."""
        return self._segmenter_v2 is not None

    @property
    def denoiser(self) -> nn.Module:
        if self._denoiser is None:
            # If no separate denoiser is configured, use an identity module as a pass-through
            return nn.Identity()
        return self._denoiser
    @property
    def model_version(self) -> str:
        return self._model_version

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def load_all(self) -> None:
        """Called once by the FastAPI lifespan at startup."""
        loop = asyncio.get_event_loop()

        # Load V1 segmenter
        logger.info("Loading V1 segmentation model (cluster segmentation)...")
        t0 = time.perf_counter()
        await loop.run_in_executor(None, self._load_sync_segmenter)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("V1 segmenter ready in %.0f ms on %s", ms, self.device)

        # Load V2 segmenter (floor vs background)
        if settings.V2_SEGMENTER_MODEL_PATH:
            logger.info("Loading V2 segmentation model (floor vs background)...")
            t0 = time.perf_counter()
            await loop.run_in_executor(None, self._load_sync_segmenter_v2)
            ms = (time.perf_counter() - t0) * 1000
            logger.info("V2 segmenter ready in %.0f ms on %s", ms, self.device)
        else:
            logger.warning(
                "V2_SEGMENTER_MODEL_PATH not set. "
                "Floor/background detection will be skipped — V1 clusters only."
            )

        # Load denoiser (optional)
        if settings.DENOISER_MODEL_PATH:
            logger.info("Loading denoiser model...")
            t0 = time.perf_counter()
            await loop.run_in_executor(None, self._load_sync_denoiser)
            ms = (time.perf_counter() - t0) * 1000
            logger.info("Denoiser ready in %.0f ms on %s", ms, self.device)

        logger.info("All models loaded successfully.")

    # ── V1: Cluster segmentation (rooms, kitchens, etc.) ──────────────────────

    def _load_sync_segmenter(self) -> None:
        """Synchronous load — runs in thread pool so event loop stays free."""
        path = Path(settings.SEGMENTER_MODEL_PATH)

        if not path.exists():
            logger.warning(
                "V1 model file not found at '%s'. "
                "Place best_model_optimized.pth in the V1/ folder "
                "and set SEGMENTER_MODEL_PATH in your .env file. "
                "Starting with an untrained U-Net for now.",
                path,
            )
            self._segmenter = UNet(
                in_channels=3,
                num_classes=settings.NUM_CLASSES,
            ).to(self.device)
            self._segmenter.eval()
            # Warm-up for segmenter
            dummy = torch.zeros(
                1, 3, settings.INPUT_SIZE, settings.INPUT_SIZE, device=self.device
            )
            with torch.no_grad():
                _ = self._segmenter(dummy)
            logger.info("V1 segmenter warm-up complete. Model is ready.")
            return

        logger.info("Loading V1 weights from '%s'", path)
        checkpoint = torch.load(path, map_location=self.device)

        # ── Auto-detect checkpoint format ─────────────────────────────────────
        state_dict = self._extract_state_dict(checkpoint, path)

        # ── Build architecture and load weights ───────────────────────────────
        self._segmenter = smp.UnetPlusPlus(
            encoder_name        = settings.ENCODER,        # efficientnet-b4
            encoder_weights     = None,
            in_channels         = 3,
            classes             = settings.NUM_CLASSES,
            activation          = None,               # raw logits
            decoder_attention_type = "scse",          # Squeeze-Excitation in decoder
        ).to(self.device)

        missing, unexpected = self._segmenter.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in V1 checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in V1 checkpoint: %s", unexpected)

        self._segmenter.eval()   # CRITICAL — disables dropout/batchnorm train mode

        # ── Warm-up forward pass ──────────────────────────────────────────────
        dummy = torch.zeros(
            1, 3, settings.INPUT_SIZE, settings.INPUT_SIZE, device=self.device
        )
        with torch.no_grad():
            _ = self._segmenter(dummy)

        logger.info("V1 segmenter warm-up complete. Model is ready.")

    # ── V2: Floor vs background binary segmentation ───────────────────────────

    def _load_sync_segmenter_v2(self) -> None:
        """Load V2 model — binary (inside floor vs background) segmentation."""
        path = Path(settings.V2_SEGMENTER_MODEL_PATH)

        if not path.exists():
            logger.warning(
                "V2 model file not found at '%s'. "
                "Floor/background detection will be unavailable.",
                path,
            )
            return

        logger.info("Loading V2 weights from '%s'", path)
        checkpoint = torch.load(path, map_location=self.device)

        # ── Auto-detect checkpoint format ─────────────────────────────────────
        state_dict = self._extract_state_dict(checkpoint, path)

        # ── Build V2 architecture (same UnetPlusPlus, different num_classes) ──
        self._segmenter_v2 = smp.UnetPlusPlus(
            encoder_name        = settings.ENCODER,        # efficientnet-b4
            encoder_weights     = None,
            in_channels         = 3,
            classes             = settings.V2_NUM_CLASSES,  # 2 = background + inside floor
            activation          = None,               # raw logits
            decoder_attention_type = "scse",
        ).to(self.device)

        missing, unexpected = self._segmenter_v2.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in V2 checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in V2 checkpoint: %s", unexpected)

        self._segmenter_v2.eval()

        # ── Warm-up forward pass ──────────────────────────────────────────────
        dummy = torch.zeros(
            1, 3, settings.INPUT_SIZE, settings.INPUT_SIZE, device=self.device
        )
        with torch.no_grad():
            _ = self._segmenter_v2(dummy)

        logger.info("V2 segmenter warm-up complete. Model is ready.")

    # ── Denoiser ──────────────────────────────────────────────────────────────

    def _load_sync_denoiser(self) -> None:
        """Synchronous load for denoiser — runs in thread pool."""
        path = Path(settings.DENOISER_MODEL_PATH)

        if not path.exists():
            logger.warning(
                "Denoiser model file not found at '%s'. "
                "Starting with an untrained U-Net for now.",
                path,
            )
            self._denoiser = UNet(
                in_channels=3,
                num_classes=3, # Assuming denoiser outputs 3 channels (RGB)
            ).to(self.device)
            self._denoiser.eval()
            # Warm-up for denoiser
            dummy = torch.zeros(
                1, 3, settings.INPUT_SIZE, settings.INPUT_SIZE, device=self.device
            )
            with torch.no_grad():
                _ = self._denoiser(dummy)
            logger.info("Denoiser warm-up complete. Model is ready.")
            return

        logger.info("Loading denoiser weights from '%s'", path)
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint, path)

        self._denoiser = UNet(
            in_channels=3,
            num_classes=3, # Assuming denoiser outputs 3 channels (RGB)
        ).to(self.device)

        missing, unexpected = self._denoiser.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in denoiser checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in denoiser checkpoint: %s", unexpected)

        self._denoiser.eval()

        dummy = torch.zeros(
            1, 3, settings.INPUT_SIZE, settings.INPUT_SIZE, device=self.device
        )
        with torch.no_grad():
            _ = self._denoiser(dummy)

        logger.info("Denoiser warm-up complete. Model is ready.")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _extract_state_dict(self, checkpoint, path: Path) -> dict:
        """
        Handle all common PyTorch checkpoint formats:
          - Raw model object (torch.save(model, ...))
          - Raw state_dict (torch.save(model.state_dict(), ...))
          - Training checkpoint dict with 'model_state_dict' key
          - PyTorch Lightning checkpoint with 'state_dict' key
        """
        if isinstance(checkpoint, nn.Module):
            # Saved with torch.save(model, path) — whole model object
            logger.info("Checkpoint contains full model object.")
            logger.info("Extracting state_dict from full model object.")
            return checkpoint.state_dict()

        if isinstance(checkpoint, dict):
            # Extract version if available
            self._model_version = str(checkpoint.get("version", checkpoint.get("epoch", "1.0")))

            for key in ("model_state", "model_state_dict", "state_dict", "model"):
                if key in checkpoint:
                    logger.info("Using '%s' key from checkpoint.", key)
                    return checkpoint[key]

            # Assume the dict itself IS the state_dict
            logger.info("Treating checkpoint dict directly as state_dict.")
            return checkpoint

        raise ValueError(
            f"Unrecognised checkpoint format in '{path}'. "
            f"Expected nn.Module or dict, got {type(checkpoint)}."
        )

    async def unload_all(self) -> None:
        self._segmenter = None
        self._segmenter_v2 = None
        self._denoiser = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded.")