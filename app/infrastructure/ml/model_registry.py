"""

Model Registry — Singleton pattern for ML model lifecycle.

YOUR MODEL FILE:
  1. Rename "segmentation optimized.pth" → "segmentation_optimized.pth"
  2. Place it at:  ceiling_ai/models/segmentation_optimized.pth
  3. Set in .env:  SEGMENTER_MODEL_PATH=models/segmentation_optimized.pth

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
        self._denoiser: Optional[nn.Module] = None
        self._model_version: str = "1.0.0"

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def segmenter(self) -> nn.Module:
        if self._segmenter is None:
            raise RuntimeError("Segmenter not loaded — check startup logs.")
        return self._segmenter

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
        logger.info("Loading segmentation model...")
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync_segmenter)
        if settings.DENOISER_MODEL_PATH:
            logger.info("Loading denoiser model...")
            t0 = time.perf_counter()
            await loop.run_in_executor(None, self._load_sync_denoiser)
            ms = (time.perf_counter() - t0) * 1000
            logger.info("Denoiser ready in %.0f ms on %s", ms, self.device)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("Model ready in %.0f ms on %s", ms, self.device)

    def _load_sync_segmenter(self) -> None:
        """Synchronous load — runs in thread pool so event loop stays free."""
        path = Path(settings.SEGMENTER_MODEL_PATH)

        if not path.exists():
            logger.warning(
                "Model file not found at '%s'. "
                "Place segmentation_optimized.pth in the models/ folder "
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
            logger.info("Segmenter warm-up complete. Model is ready.")
            return

        logger.info("Loading weights from '%s'", path)
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
            logger.warning("Missing keys in checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

        self._segmenter.eval()   # CRITICAL — disables dropout/batchnorm train mode

        # ── Warm-up forward pass ──────────────────────────────────────────────
        # Fills CUDA caches so the first real request is not slower than normal.
        dummy = torch.zeros(
            1, 3, settings.INPUT_SIZE, settings.INPUT_SIZE, device=self.device
        )
        with torch.no_grad():
            _ = self._segmenter(dummy)

        logger.info("Segmenter warm-up complete. Model is ready.")

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")