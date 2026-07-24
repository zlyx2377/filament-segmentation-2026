"""Physical-prior preprocessing for GONG H-Alpha filament images.

Key ideas:
  * Images are GRAYSCALE 8-bit JPEG (NOT RGB).
  * Filaments are dark absorbing features -> low local contrast vs background.
  * Limb darkening makes the solar-disk edge look like a false dark ring ->
    we estimate the solar disk and mask everything outside it.
  * We assemble a 3-channel input [raw, CLAHE, gradient] so ImageNet-pretrained
    encoders (ConvNeXt/Swin) can be used directly while carrying more signal.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return cl.apply(gray)


def estimate_disk_mask(gray: np.ndarray) -> np.ndarray:
    """Return a boolean mask (H, W): True inside the solar disk."""
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    # Disk is brighter than space -> Otsu separates them; disk = 255.
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    if num <= 1:
        return np.ones((H, W), dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == largest
    mask = ndimage.binary_fill_holes(mask)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.ones((H, W), dtype=bool)
    cx, cy = np.mean(xs), np.mean(ys)
    r = float(np.percentile(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2), 95))
    Y, X = np.ogrid[:H, :W]
    circ = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    return circ


def to_3channel(gray: np.ndarray) -> np.ndarray:
    """Stack [raw, CLAHE, gradient-magnitude] -> (3, H, W) uint8."""
    cl = clahe(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    p99 = np.percentile(gm + 1e-6, 99)
    gm = np.clip(gm / max(p99, 1e-6), 0, 1) * 255.0
    three = np.stack([gray, cl, gm.astype(np.uint8)], axis=0)
    return three.astype(np.uint8)


def preprocess_image(gray: np.ndarray):
    """Full preprocessing. Returns (image_3ch (3,H,W) uint8, disk_mask (H,W) bool)."""
    disk = estimate_disk_mask(gray)
    three = to_3channel(gray)
    return three, disk
