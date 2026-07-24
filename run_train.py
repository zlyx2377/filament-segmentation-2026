"""Entry point: train (single holdout fold by default)."""
from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train import main as train_main

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else None
    train_main(cfg, fold)
