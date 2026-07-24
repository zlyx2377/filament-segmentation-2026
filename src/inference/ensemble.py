"""Ensemble multiple predictors by averaging probability maps."""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from src.inference.predictor import Predictor


class EnsemblePredictor:
    def __init__(self, cfg: dict, checkpoint_paths: List[str], device: str = "cuda"):
        self.predictors = [Predictor(cfg, p, device) for p in checkpoint_paths]

    def predict(self, gray: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        fil_sum = None
        spi_sum = None
        disk = None
        n = 0
        for pred in self.predictors:
            fil, spi, disk = pred.predict(gray)
            fil_sum = fil if fil_sum is None else fil_sum + fil
            if spi is not None:
                spi_sum = spi if spi_sum is None else spi_sum + spi
            n += 1
        fil = fil_sum / n
        spi = (spi_sum / n) if spi_sum is not None else None
        return fil, spi, disk
