"""Build the competition submission CSV.

For each test image: predict -> separate instances -> filter -> encode each
instance as an RLE-counts string -> one row `{image_base}_{n}`.
"""
from __future__ import annotations

import csv
import glob
import os

from src.inference.ensemble import EnsemblePredictor
from src.inference.predictor import Predictor
from src.postprocess.instance import separate_instances
from src.postprocess.filter import filter_instances
from src.utils.io_utils import image_base_name, read_gray
from src.utils.rle import mask_to_rle


def _find_checkpoints(cfg: dict) -> list:
    out_dir = cfg["experiment"]["out_dir"]
    pattern = os.path.join(out_dir, "checkpoints", f"{cfg['experiment']['name']}_fold*_best.pth")
    ckpts = sorted(glob.glob(pattern))
    return ckpts


def build_submission(cfg: dict, predictor, out_csv: str) -> int:
    test_dir = os.path.join(cfg["data"]["root"], cfg["data"]["test_images_dir"])
    files = sorted(
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )
    rows = []
    for fp in files:
        gray = read_gray(fp)
        fil, spi, _ = predictor.predict(gray)
        insts = separate_instances(fil, spi, cfg)
        insts = filter_instances(insts, cfg)
        base = image_base_name(fp)
        for i, m in enumerate(insts, 1):
            rows.append((f"{base}_{i}", mask_to_rle(m)))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(["filament_id", "segmentation_rle"])
        for r in rows:
            w.writerow(r)
    return len(rows)


def make_predictor(cfg: dict, device: str = "cuda"):
    ckpts = _find_checkpoints(cfg)
    if not ckpts:
        raise FileNotFoundError("No checkpoints found. Train first (run_train.py).")
    if len(ckpts) > 1:
        return EnsemblePredictor(cfg, ckpts, device)
    return Predictor(cfg, ckpts[0], device)
