"""Small IO / helper utilities: seeding, JSON load, image read, logging."""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def read_gray(path: str) -> np.ndarray:
    """Read an image as grayscale uint8 (H, W). Competition images are grayscale."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return img


def image_base_name(filename: str) -> str:
    """`20160920230134Lh.jpeg` -> `20160920230134Lh`."""
    return os.path.splitext(os.path.basename(filename))[0]


def get_logger(name: str = "filament") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                         "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def to_cpu(t: Any):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else t
