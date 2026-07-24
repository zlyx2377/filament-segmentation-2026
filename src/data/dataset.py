"""PyTorch Dataset for MAGFiLO (COCO-style) filament annotations.

Real-data structure (discovered from the released dataset):
  * The COCO json lists 1154 image entries but only 707 unique files exist on
    disk. The same physical file is listed multiple times under *different*
    `image_id` prefixes (e.g. ``050101-...``, ``050102-...``, ``050103-...``),
    each with its OWN independent set of annotations (different categories!).
    These prefixes are annotation passes / observers.
  * We **group annotations by physical file** (``file_name``) and treat one file
    = one training sample. By default we MERGE all passes (union of polygons +
    spines) to build a high-recall, consensus foreground target. This also
    removes the train/val leakage risk that per-pass splitting would create.

Rasterization is done with **pycocotools** (the exact COCO convention the
competition uses) at full resolution, ONCE per file, and cached. Crops are then
simple slices of the cached full-res masks — exact AND fast (no per-crop cv2
fillPoly, which disagrees with the official rasterizer by ~9% area).
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pycocotools import mask as maskUtils

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
        # union | first  (how to combine multiple annotation passes per file)
        self.merge = cfg["data"].get("annotation_merge", "union")

        data = load_json(os.path.join(root, ann_file))
        self.images = data["images"]
        self.anns = data.get("annotations", [])

        # image_id (str) -> file_name (str, unique physical file)
        file_by_id = {im["id"]: im["file_name"] for im in self.images}

        # group every annotation under its physical file
        anns_by_file: dict[str, list] = defaultdict(list)
        for a in self.anns:
            if a.get("iscrowd", 0) == 1:
                continue
            fn = file_by_id.get(a["image_id"])
            if fn:
                anns_by_file[fn].append(a)

        self.img_dir = os.path.join(root, images_dir)
        disk_files = set(os.listdir(self.img_dir)) if os.path.isdir(self.img_dir) else set()

        # one sample per file that exists on disk AND has annotations
        self.files = sorted(f for f in disk_files if f in anns_by_file)
        self.anns_by_file = {f: anns_by_file[f] for f in self.files}
        self.image_bases = [image_base_name(f) for f in self.files]
        self._cache: dict = {}

        if fold_indices is not None:
            self.files = [self.files[i] for i in fold_indices]

    # ------------------------------------------------------------------ #
    def _build_full_masks(self, fname: str, H: int, W: int):
        """Exact (pycocotools) full-res union filament + spine masks for a file."""
        anns = self.anns_by_file.get(fname, [])
        if self.merge == "first":
            anns = anns[:1]
        fil_polys, spi_polys = [], []
        for a in anns:
            for poly in a.get("segmentation", []):
                if len(poly) >= 6:
                    fil_polys.append(poly)
            sp = a.get("spine", [])
            if sp and len(sp) >= 4:
                spi_polys.append(sp)
        fil = np.zeros((H, W), np.uint8)
        if fil_polys:
            rle = maskUtils.merge(maskUtils.frPyObjects(fil_polys, H, W))
            fil = maskUtils.decode(rle)
        spi = np.zeros((H, W), np.uint8)
        if self.use_spine and spi_polys:
            for sp in spi_polys:
                pts = np.asarray(sp, np.float32).reshape(-1, 2)
                if len(pts) >= 2:
                    cv2.polylines(spi, [pts.astype(np.int32)], isClosed=False, color=1, thickness=2)
        return fil, spi

    def _load(self, fname: str):
        if fname in self._cache:
            return self._cache[fname]
        path = os.path.join(self.img_dir, fname)
        gray = read_gray(path)
        three, disk = preprocess_image(gray)
        H, W = three.shape[1], three.shape[2]
        fil, spi = self._build_full_masks(fname, H, W)
        self._cache[fname] = (three, disk, H, W, fil, spi)
        return self._cache[fname]

    def _getitem_train(self, fname, _anns):
        three, disk, H, W, fil, spi = self._load(fname)
        cs = random.choice(self.crop_sizes)
        cs = min(cs, H, W)
        x = random.randint(0, max(0, W - cs))
        y = random.randint(0, max(0, H - cs))
        img_crop = three[:, y : y + cs, x : x + cs]
        disk_crop = disk[y : y + cs, x : x + cs]
        fil_crop = fil[y : y + cs, x : x + cs] * disk_crop.astype(np.uint8)
        spi_crop = spi[y : y + cs, x : x + cs] * disk_crop.astype(np.uint8)

        if self.transforms is not None:
            out = self.transforms(image=img_crop.transpose(1, 2, 0), mask=fil_crop, spine=spi_crop)
            img_crop = out["image"]
            fil_crop = out["mask"]
            spi_crop = out["spine"]

        img_t = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - MEAN[:, None, None]) / STD[:, None, None]
        fil_t = torch.from_numpy(fil_crop).float().unsqueeze(0)
        spi_t = torch.from_numpy(spi_crop).float().unsqueeze(0) if self.use_spine else torch.zeros_like(fil_t)
        disk_t = torch.from_numpy(disk_crop.astype(np.uint8)).float().unsqueeze(0)
        return {"image": img_t, "mask": fil_t, "spine": spi_t, "disk": disk_t}

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        anns = self.anns_by_file.get(fname, [])
        return self._getitem_train(fname, anns)

    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------------------ #
    # Helpers for evaluation.
    def gt_full_mask(self, image_base: str, H: int = 2048, W: int = 2048) -> np.ndarray:
        """Exact union filament mask (H,W) for a file, built from all passes."""
        fname = None
        for f in self.files:
            if image_base_name(f) == image_base:
                fname = f
                break
        if fname is None:
            return np.zeros((H, W), np.uint8)
        _, _, _, _, fil, _ = self._load(fname)
        return fil

    def gt_instances(self, image_base: str, H: int = 2048, W: int = 2048) -> list:
        """GT instances = connected components of the union mask.

        Using connected components (not per-annotation masks) is the standard
        instance definition and avoids false fragmentation/merge penalties when
        multiple annotation passes overlap on the same filament.
        """
        from scipy import ndimage
        mask = self.gt_full_mask(image_base, H, W)
        if mask.sum() == 0:
            return []
        labels, n = ndimage.label(mask)
        return [((labels == k).astype(np.uint8)) for k in range(1, n + 1)]
