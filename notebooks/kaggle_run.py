"""End-to-end orchestrator for a Kaggle Notebook.

Usage (in a Kaggle notebook cell):
    from notebooks.kaggle_run import main
    main("configs/base.yaml", do_train=True, do_infer=True)   # single holdout fold
    # For an ensemble, train several folds/backbones first, then infer:
    #   main("configs/base.yaml", do_train=True, do_infer=False, folds=[0,1,2])
    #   main("configs/base.yaml", do_train=False, do_infer=True)

The inference step auto-discovers all `{name}_fold*_best.pth` checkpoints and
ensembles them (see src/submission/make_submission.make_predictor).
"""
from __future__ import annotations

import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train import train_fold
from src.utils.io_utils import get_logger


def main(config_path: str = "configs/base.yaml", do_train: bool = True,
         do_infer: bool = True, folds=None):
    from src.utils.config import load_config
    cfg = load_config(config_path)
    logger = get_logger("kaggle_run")

    if do_train:
        if folds is None:
            folds = [int(cfg["data"].get("holdout_fold", 0))]
        for fd in folds:
            logger.info(f"=== training fold {fd} ===")
            train_fold(cfg, fd, logger)

    if do_infer:
        from run_infer import main as infer_main
        infer_main(config_path)

    logger.info("done.")


if __name__ == "__main__":
    main()
