"""
Shared image preprocessing for the knee metal/regular classifier.

The same `standardize_exposure` function is used at training time, in
evaluate_folder.py, and in app.py, so a model never sees pixels prepared a
different way than it was trained on.

Knee X-rays come from different machines/settings, so raw pixel intensity
and contrast vary a lot image to image. "Standardizing exposure" here means:
  1. Percentile clipping - the darkest/brightest ~1% of pixels (sensor
     artifacts, burned-in text/markers, extreme over/under-exposure) are
     clipped before rescaling, so a few outlier pixels can't skew the
     min/max used for normalization.
  2. Min-max rescale the clipped range back to the full 0-255 range.
  3. CLAHE (contrast-limited adaptive histogram equalization) to bring
     local contrast onto a comparable footing across images regardless of
     how the original exposure was set.
"""

import cv2
import numpy as np
from PIL import Image

CLIP_LOW_PERCENTILE = 1.0
CLIP_HIGH_PERCENTILE = 99.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Native input resolution for EfficientNet-B1.
IMAGE_SIZE = 240
RESIZE_SIZE = 255  # matches EfficientNet_B1_Weights.IMAGENET1K_V2's eval transform

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def standardize_exposure_gray(img: Image.Image) -> np.ndarray:
    """Grayscale + outlier-clip + CLAHE an image. Returns a uint8 single-channel array."""
    gray = np.array(img.convert("L"), dtype=np.float32)

    lo, hi = np.percentile(gray, [CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE])
    if hi <= lo:
        # Degenerate (near-blank) image - fall back to the full range.
        lo, hi = float(gray.min()), float(gray.max())
        if hi <= lo:
            hi = lo + 1.0

    clipped = np.clip(gray, lo, hi)
    rescaled = (clipped - lo) / (hi - lo) * 255.0
    rescaled = rescaled.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(rescaled)


def standardize_exposure(img: Image.Image) -> Image.Image:
    """Same as standardize_exposure_gray, but returns a 3-channel RGB PIL image
    (for feeding into the CNN, which expects 3-channel input)."""
    equalized = standardize_exposure_gray(img)
    rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)
