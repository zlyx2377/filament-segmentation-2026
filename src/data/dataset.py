"""PyTorch Dataset for MAGFiLO (COCO-style) filament annotations.

Key handling:
  * Multi-annotator: each annotated image is already a SEPARATE entry in the COCO
    json (image_id includes the annotator batch). We simply iterate them; the
    GroupKFold in training/folds.py groups by physical image base to avoid leakage.
  * Spine annotations (provided but not scored) are rasterized as an auxiliary
    target when cfg['data']['use_spine_aux'] is True.
  * Training samples random crops (full 2048 is too large for most GPUs); masks
    are rasterized directly into the crop window for efficiency.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_image
from src.utils.io_utils import image_base_name, load_json, read_gray

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FilamentDataset(Dataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        images_dir: str,
        cfg: dict,
        mode: str = "train",
        transforms=None,
        fold_indices: Optional[list] = None,
    ):
        self.cfg = cfg
        self.mode = mode
        self.transforms = transforms
        self.use_spine = bool(cfg["data"].get("use_spine_aux", True))
        self.crop_sizes = cfg["data"].get("crop_sizes", [cfg["data"]["crop_size"]])

        data = load_json(os.path.join(root, ann_file))
        self.images = data["images"]
        self.anns = data.get("annotations", [])

        self.img_by_id = {im["id"]: im for im in self.images}
        self.anns_by_img: dict = {}
        for a in self.anns:
            if a.get("iscrowd", 0) == 1:
                continue
            self.anns_by_img.setdefault(a["image_id"], []).append(a)

        self.img_dir = os.path.join(root, images_dir)
        self._cache: dict = {}

        if fold_indices is not None:
            self.images = [self.images[i] for i in fold_indices]

    # ------------------------------------------------------------------ #
    def _load(self, im: dict):
        if im["id"] in self._cache:
            return self._cache[im["id"]]
        path = os.path.join(self.img_dir, im["file_name"])
        gray = read_gray(path)
        three, disk = preprocess_image(gray)
        H, W = three.shape[1], three.shape[2]
        self._cache[im["id"]] = (three, disk, H, W)
        return self._cache[im["id"]]

    @staticmethod
    def _rasterize(anns, x, y, w, h, use_spine=True):
        fil = np.zeros((h, w), np.uint8)
        spi = np.zeros((h, w), np.uint8)
        for a in anns:
            for poly in a.get("segmentation", []):
                pts = np.asarray(poly, np.float32).reshape(-1, 2)
                pts[:, 0] -= x
                pts[:, 1] -= y
                if len(pts) >= 3:
                    cv2.fillPoly(fil, [pts.astype(np.int32)], 1)
            if use_spine:
                sp = a.get("spine", [])
                if sp and len(sp) >= 4:
                    pts = np.asarray(sp, np.float32).reshape(-1, 2)
                    pts[:, 0] -= x
                    pts[:, 1] -= y
                    cv2.polylines(spi, [pts.astype(np.int32)], isClosed=False, color=1, thickness=2)
        return fil, spi

    def _getitem_train(self, im, anns):
        three, disk, H, W = self._load(im)
        cs = random.choice(self.crop_sizes)
        cs = min(cs, H, W)
        x = random.randint(0, max(0, W - cs))
        y = random.randint(0, max(0, H - cs))
        img_crop = three[:, y : y + cs, x : x + cs]
        disk_crop = disk[y : y + cs, x : x + cs]
        fil, spi = self._rasterize(anns, x, y, cs, cs, self.use_spine)
        fil = fil * disk_crop.astype(np.uint8)
        spi = spi * disk_crop.astype(np.uint8)

        if self.transforms is not None:
            out = self.transforms(image=img_crop.transpose(1, 2, 0), mask=fil, spine=spi)
            img_crop = out["image"]
            fil = out["mask"]
            spi = out["spine"]

        img_t = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - MEAN[:, None, None]) / STD[:, None, None]
        fil_t = torch.from_numpy(fil).float().unsqueeze(0)
        spi_t = torch.from_numpy(spi).float().unsqueeze(0) if self.use_spine else torch.zeros_like(fil_t)
        disk_t = torch.from_numpy(disk_crop.astype(np.uint8)).float().unsqueeze(0)
        return {"image": img_t, "mask": fil_t, "spine": spi_t, "disk": disk_t}

    def __getitem__(self, idx: int):
        im = self.images[idx]
        anns = self.anns_by_img.get(im["id"], [])
        return self._getitem_train(im, anns)

    def __len__(self):
        return len(self.images)

    # ------------------------------------------------------------------ #
    # Helpers for evaluation (full-res GT instances for a given image id)
    def gt_instances(self, image_id, H, W):
        """Return list of full-res (H,W) binary masks, one per filament annotation."""
        from src.utils.rle import polygons_to_mask
        anns = self.anns_by_img.get(image_id, [])
        masks = []
        for a in anns:
            m = polygons_to_mask(a["segmentation"], H, W)
            if m.sum() > 0:
                masks.append(m)
        return masks
