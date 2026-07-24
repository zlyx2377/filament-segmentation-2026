"""Instance separation: turn a fused filament probability map into per-filament
binary masks. Two strategies (config-selectable):

  * watershed      : distance-transform + local-maxima markers (classic)
  * spine_seed     : watershed seeded by the PREDICTED SPINE points (uses the
                     auxiliary head). This better separates touching filaments
                     and reduces many-to-one errors — our competitive edge.

Either way we avoid blind morphological closing (which would merge filaments
and trigger many-to-one penalties).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import label as sklabel
from skimage.segmentation import watershed


def separate_instances(prob: np.ndarray, spine: np.ndarray | None, cfg: dict) -> list:
    thr = float(cfg["inference"]["threshold"])
    bin_mask = (prob > thr).astype(np.uint8)
    if bin_mask.sum() == 0:
        return []

    icfg = cfg["inference"]["instance"]
    use_spine = icfg.get("use_spine_seed", False) and spine is not None

    dist = ndimage.distance_transform_edt(bin_mask)
    sigma = float(icfg.get("watershed_smooth_sigma", 1.0))
    if sigma > 0:
        dist = ndimage.gaussian_filter(dist, sigma)

    if use_spine:
        spine_thr = float(cfg["inference"].get("spine_threshold", 0.5))
        seeds = (spine > spine_thr).astype(np.uint8)
        # ensure seed connectivity inside the binary mask
        seeds = seeds * bin_mask
        markers = sklabel(seeds)
        if markers.max() == 0:
            # fallback to distance peaks
            coords = peak_local_max(dist, min_distance=int(icfg.get("min_distance", 12)), labels=bin_mask)
            markers = np.zeros_like(dist, dtype=np.int32)
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    else:
        coords = peak_local_max(dist, min_distance=int(icfg.get("min_distance", 12)), labels=bin_mask)
        markers = np.zeros_like(dist, dtype=np.int32)
        if len(coords) > 0:
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    if markers.max() == 0:
        # whole connected region -> single instance
        return [bin_mask.copy()]

    labels = watershed(-dist, markers, mask=bin_mask)
    out = []
    for lid in np.unique(labels):
        if lid == 0:
            continue
        m = (labels == lid).astype(np.uint8)
        if m.sum() > 0:
            out.append(m)
    return out
