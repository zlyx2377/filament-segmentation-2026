"""Check whether the competition TEST images overlap the PUBLIC MAGFiLO release.

Discussion board signals the test set may be a subset of the public MAGFiLO
dataset (Nature Scientific Data, 10.1038/s41597-024-03876-y). If so, their
ground-truth masks might already be public -> a potential "free labels" angle.

HOW TO RUN (on Kaggle, after you have the test images):
  1. Download the public MAGFiLO index (a CSV mapping image base-name -> has_mask)
     e.g. from the MAGFiLO GitHub / paper supplementary.
  2. python src/submission/check_test_overlap.py \
        --test-dir /kaggle/input/.../test/test_images \
        --magfilo-index magfilo_public_index.csv

This prints how many test images appear in the public index. If the overlap is
high, investigate whether using public GT is permitted under the rules BEFORE
doing so.
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--magfilo-index", required=True, help="CSV with column 'image_base'")
    args = ap.parse_args()

    import pandas as pd
    from src.utils.io_utils import image_base_name

    test_bases = {
        image_base_name(f)
        for f in os.listdir(args.test_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    }
    idx = pd.read_csv(args.magfilo_index)
    public_bases = set(idx["image_base"].astype(str))
    overlap = test_bases & public_bases
    print(f"test images : {len(test_bases)}")
    print(f"public index: {len(public_bases)}")
    print(f"OVERLAP     : {len(overlap)} ({100*len(overlap)/max(len(test_bases),1):.1f}%)")
    if overlap:
        print("Sample overlapping bases:", sorted(overlap)[:10])


if __name__ == "__main__":
    main()
