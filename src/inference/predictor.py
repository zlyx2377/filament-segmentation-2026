"""Sliding-window + TTA inference for a single model checkpoint.

Produces fused filament probability maps (and spine probability if aux enabled),
masked to the solar disk. Designed for 2048x2048 inputs.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import MEAN, STD
from src.data.preprocessing import preprocess_image
from src.models.builder import build_model, split_output
from src.utils.io_utils import read_gray


class Predictor:
    def __init__(self, cfg: dict, checkpoint_path: str, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_spine = bool(cfg["data"].get("use_spine_aux", True))
        self.tile_size = int(cfg["inference"]["tile_size"])
        self.tile_overlap = float(cfg["inference"]["tile_overlap"])
        self.threshold = float(cfg["inference"]["threshold"])
        self.spine_threshold = float(cfg["inference"].get("spine_threshold", 0.5))
        self.tta = self._build_tta_modes()

        model = build_model(cfg)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        self.model = model

    def _build_tta_modes(self) -> List[str]:
        modes = ["none"]
        for f in self.cfg["inference"]["tta"].get("flips", ["none"]):
            if f != "none":
                modes.append(f)
        if self.cfg["inference"]["tta"].get("rot90", False):
            modes += ["r1", "r2", "r3"]
        return modes

    @staticmethod
    def _apply_tta(arr3: np.ndarray, mode: str) -> np.ndarray:
        a = arr3
        if mode == "h":
            return np.flip(a, 2)
        if mode == "v":
            return np.flip(a, 1)
        if mode == "hv":
            return np.flip(np.flip(a, 2), 1)
        if mode == "r1":
            return np.rot90(a, 1, axes=(1, 2))
        if mode == "r2":
            return np.rot90(a, 2, axes=(1, 2))
        if mode == "r3":
            return np.rot90(a, 3, axes=(1, 2))
        return a

    @staticmethod
    def _inverse_tta(prob2d: np.ndarray, mode: str) -> np.ndarray:
        p = prob2d
        if mode == "h":
            return np.flip(p, 1)
        if mode == "v":
            return np.flip(p, 0)
        if mode == "hv":
            return np.flip(np.flip(p, 1), 0)
        if mode == "r1":
            return np.rot90(p, -1, axes=(0, 1))
        if mode == "r2":
            return np.rot90(p, -2, axes=(0, 1))
        if mode == "r3":
            return np.rot90(p, -3, axes=(0, 1))
        return p

    def _infer_tiles(self, arr3: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ts = min(self.tile_size, arr3.shape[1], arr3.shape[2])
        H, W = arr3.shape[1], arr3.shape[2]
        ov = self.tile_overlap
        stride = max(1, int(ts * (1 - ov)))
        ys = list(range(0, H - ts + 1, stride))
        xs = list(range(0, W - ts + 1, stride))
        if not ys or ys[-1] != H - ts:
            ys.append(H - ts)
        if not xs or xs[-1] != W - ts:
            xs.append(W - ts)
        ys = [y for y in ys if y >= 0]
        xs = [x for x in xs if x >= 0]

        fil = np.zeros((H, W), np.float32)
        spi = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        mean = torch.from_numpy(MEAN).to(self.device).view(3, 1, 1)
        std = torch.from_numpy(STD).to(self.device).view(3, 1, 1)

        with torch.no_grad():
            for y in ys:
                for x in xs:
                    tile = arr3[:, y : y + ts, x : x + ts]
                    t = torch.from_numpy(tile).float() / 255.0
                    t = (t - mean) / std
                    out = self.model(t.unsqueeze(0).to(self.device))
                    fo, so = split_output(out, self.use_spine)
                    pf = torch.sigmoid(fo)[0, 0].cpu().numpy()
                    ps = torch.sigmoid(so)[0, 0].cpu().numpy() if so is not None else None
                    fil[y : y + ts, x : x + ts] += pf
                    wsum[y : y + ts, x : x + ts] += 1
                    if ps is not None:
                        spi[y : y + ts, x : x + ts] += ps
        fil /= np.maximum(wsum, 1)
        spi = spi / np.maximum(wsum, 1) if self.use_spine else None
        return fil, (spi if self.use_spine else None)

    def predict(self, gray: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """gray: (H,W) uint8. Returns (fil_prob, spine_prob|None, disk_mask)."""
        three, disk = preprocess_image(gray)
        acc_fil = np.zeros(three.shape[1:], np.float32)
        acc_spi = np.zeros(three.shape[1:], np.float32)
        w = 0
        for mode in self.tta:
            arr = self._apply_tta(three, mode)
            fil, spi = self._infer_tiles(arr)
            fil = self._inverse_tta(fil, mode)
            spi = self._inverse_tta(spi, mode) if spi is not None else None
            acc_fil += fil
            if spi is not None:
                acc_spi += spi
            w += 1
        fil = acc_fil / max(w, 1)
        spi = acc_spi / max(w, 1) if self.use_spine else None
        fil = fil * disk.astype(np.float32)
        return fil, spi, disk
