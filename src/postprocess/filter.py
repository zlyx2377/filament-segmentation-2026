"""Post-filtering: remove fragments (cut one-to-many) and lightly smooth edges.

We deliberately do NOT merge touching masks (that would cause many-to-one).
Area thresholds are the main lever for the fragmentation penalty.
"""
from __future__ import annotations

import cv2
import numpy as np


def filter_instances(masks: list, cfg: dict) -> list:
    pcfg = cfg["inference"]["postprocess"]
    min_area = int(pcfg.get("min_area", 80))
    max_area = int(pcfg.get("max_area", 900000))
    smooth = int(pcfg.get("morphological_smooth", 1))

    out = []
    for m in masks:
        area = int(m.sum())
        if area < min_area or area > max_area:
            continue
        if smooth > 0:
            k = np.ones((3, 3), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=smooth)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=smooth)
        if m.sum() >= min_area:
            out.append((m > 0).astype(np.uint8))
    return out
