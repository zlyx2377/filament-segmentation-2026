"""Validate a submission CSV: decode every RLE, check shape (2048x2048) and
that it is a non-empty binary mask. Catches format bugs before uploading."""
from __future__ import annotations

import sys
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.rle import rle_to_mask

EXPECTED = (2048, 2048)


def main(csv_path: str):
    total = 0
    bad = 0
    areas = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        assert r.fieldnames == ["filament_id", "segmentation_rle"], r.fieldnames
        for row in r:
            total += 1
            try:
                m = rle_to_mask(row["segmentation_rle"], *EXPECTED)
                a = int(m.sum())
                assert a > 0, "empty mask"
                areas.append(a)
            except Exception as e:  # noqa
                bad += 1
                if bad <= 5:
                    print(f"  BAD {row['filament_id']}: {e}")
    print(f"rows={total} bad={bad} masks_ok={total-bad}")
    if areas:
        print(f"area min/mean/max = {min(areas)}/{sum(areas)//len(areas)}/{max(areas)}")
    print("OK" if bad == 0 else "FAILED validation")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    main(path)
