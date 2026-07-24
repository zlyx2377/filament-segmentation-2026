"""GroupKFold by physical image file (no leakage across annotation passes).

The released COCO json lists the same physical file multiple times under
different annotation-pass prefixes (``050101-...``, ``050102-...`` ...). We group
by the on-disk ``file_name`` (unique per physical image) so that ALL passes of
one image always land in the same fold. The dataset yields exactly one sample
per file, so we simply index by file order here.
"""
from __future__ import annotations

import random
from collections import defaultdict

from src.utils.io_utils import load_json


def make_group_folds(cfg: dict, root: str, ann_file: str) -> list:
    """Return a list `fold_of` indexed by *file* position (0..N_files-1).

    Grouping key = file_name (physical image). Each group = one file = one
    dataset sample, so no annotation pass can leak across train/val.
    """
    data = load_json(f"{root}/{ann_file}")
    images = data["images"]
    n_folds = int(cfg["data"]["n_folds"])
    seed = int(cfg["training"]["seed"])

    # unique physical files
    files = sorted({im["file_name"] for im in images})
    groups = files  # one group per file

    g2i: dict = defaultdict(list)
    for i, g in enumerate(groups):
        g2i[g].append(i)

    uniq = sorted(g2i.keys())
    rng = random.Random(seed)
    rng.shuffle(uniq)

    fold_of = [-1] * len(files)
    for fi, grp in enumerate(uniq):
        f = fi % n_folds
        for idx in g2i[grp]:
            fold_of[idx] = f
    return fold_of


def fold_splits(fold_of: list, fold: int):
    train_idx = [i for i, f in enumerate(fold_of) if f != fold]
    val_idx = [i for i, f in enumerate(fold_of) if f == fold]
    return train_idx, val_idx
