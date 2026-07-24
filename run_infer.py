"""Entry point: inference -> submission CSV."""
from __future__ import annotations

import sys
import os
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.submission.make_submission import build_submission, make_predictor


def main(cfg_path: str = "configs/base.yaml"):
    from src.utils.config import load_config
    cfg = load_config(cfg_path)
    predictor = make_predictor(cfg)
    out_csv = cfg["submission"]["out_csv"]
    n = build_submission(cfg, predictor, out_csv)
    print(f"[run_infer] wrote {n} filament rows -> {out_csv}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"
    main(cfg)
