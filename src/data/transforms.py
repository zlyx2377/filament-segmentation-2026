"""Augmentation pipeline (Albumentations).

We deliberately do NOT put Normalize/ToTensor here; the Dataset converts to a
normalized float tensor itself (keeps dtype handling explicit and testable).
The `spine` target is an additional mask ('mask' type) so it shares transforms.
"""
from __future__ import annotations

import albumentations as A


def get_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        ],
        additional_targets={"spine": "mask"},
        is_check_shapes=False,
    )
