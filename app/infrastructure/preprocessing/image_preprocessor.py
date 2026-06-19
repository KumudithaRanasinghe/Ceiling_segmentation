"""
Image Preprocessing Pipeline.

Responsibilities:
  1. Decode raw bytes → PIL Image
  2. Validate dimensions and mode
  3. Resize to model input size (preserving aspect ratio with padding)
  4. Normalize pixel values to match ImageNet pre-training stats
  5. Convert to PyTorch tensor

DESIGN: Stateless service — no instance variables mutated per call.
        Safe for concurrent async use.
"""
import io
import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from app.core.config import settings
from app.core.exceptions import ImageValidationError

logger = logging.getLogger(__name__)

# ImageNet normalization constants (used by pretrained ResNet backbones)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ImagePreprocessor:
    """
    Transforms raw image bytes into a normalized float32 tensor
    ready for the denoiser and segmentation model.
    """

    def __init__(self, target_size: int = settings.INPUT_SIZE):
        self.target_size = target_size

    def process(self, image_bytes: bytes) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            image_bytes: raw file bytes from the upload

        Returns:
            tensor:        float32 [1, 3, H, W] normalized, on CPU
            original_size: (width, height) of the input image (for postprocess rescaling)

        Raises:
            ImageValidationError: on corrupt, unsupported, or too-small images
        """
        pil_image, original_size = self._decode_and_validate(image_bytes)
        resized = self._resize_with_padding(pil_image)
        tensor = self._to_tensor_normalized(resized)
        return tensor.unsqueeze(0), original_size   # add batch dimension

    # ── Private helpers ───────────────────────────────────────────────────────

    def _decode_and_validate(self, image_bytes: bytes) -> Tuple[Image.Image, Tuple[int, int]]:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()                   # detect truncated files
            img = Image.open(io.BytesIO(image_bytes))  # reopen after verify
            img = ImageOps.exif_transpose(img)         # correct camera rotation
        except Exception as exc:
            raise ImageValidationError(f"Cannot decode image: {exc}") from exc

        if img.mode not in ("RGB", "RGBA", "L"):
            raise ImageValidationError(f"Unsupported image mode: {img.mode}")

        img = img.convert("RGB")
        w, h = img.size
        if w < 64 or h < 64:
            raise ImageValidationError(f"Image too small: {w}×{h}. Minimum 64×64.")

        return img, (w, h)

    def _resize_with_padding(self, img: Image.Image) -> Image.Image:
        """
        Letterbox resize: fit image into target_size×target_size square
        while preserving aspect ratio. Pad remainder with zeros (black).
        This avoids geometric distortion that would corrupt area measurements.
        """
        target = self.target_size
        img.thumbnail((target, target), Image.LANCZOS)
        padded = Image.new("RGB", (target, target), (0, 0, 0))
        offset_x = (target - img.width)  // 2
        offset_y = (target - img.height) // 2
        padded.paste(img, (offset_x, offset_y))
        return padded

    def _to_tensor_normalized(self, img: Image.Image) -> torch.Tensor:
        """PIL → [3, H, W] float32 tensor, ImageNet-normalized."""
        arr = np.array(img, dtype=np.float32) / 255.0         # [H, W, 3] in [0,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)       # [3, H, W]
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor
