"""RLE / mask / polygon utilities (pycocotools-based).

Competition rule: submit RLE *counts only* (no size), per-instance, using
pycocotools `encode`. Each filament -> one row with key `{image_base}_{n}`.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as maskUtils


def mask_to_rle(mask: np.ndarray) -> str:
    """mask: (H, W) bool/uint8 (0/1). Returns RLE counts string (no size)."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return counts


def rle_to_mask(rle_counts: str, height: int, width: int) -> np.ndarray:
    """Decode an RLE counts string back to a (H, W) uint8 binary mask."""
    rle = {"counts": rle_counts.encode("ascii"), "size": [int(height), int(width)]}
    return maskUtils.decode(rle)


def polygons_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    """segmentation: list with ONE polygon -> [[x0,y0, x1,y1, ...]] (COCO format)."""
    if len(segmentation) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    rles = maskUtils.frPyObjects(segmentation, int(height), int(width))
    if isinstance(rles, list):
        rles = maskUtils.merge(rles)
    return maskUtils.decode(rles)


def mask_area_rle(rle_counts: str, height: int, width: int) -> int:
    rle = {"counts": rle_counts.encode("ascii"), "size": [int(height), int(width)]}
    return int(maskUtils.area(rle))
