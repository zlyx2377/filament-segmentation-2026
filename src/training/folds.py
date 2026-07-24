"""GroupKFold by physical image base (no leakage across annotators)."""
from __future__ import annotations

import random
from collections import defaultdict

from src.utils.io_utils import image_base_name, load_json


def make_group_folds(cfg: dict, root: str, ann_file: str) -> list:
    """Return a list `fold_of` indexed by image position (0..N-1)."""
    data = load_json(f"{root}/{ann_file}")
    images = data["images"]
    n_folds = int(cfg["data"]["n_folds"])
    seed = int(cfg["training"]["seed"])

    groups = []
    for im in images:
        bid = im["id"]
        if isinstance(bid, str) and "-" in bid:
            base = bid.split("-", 1)[1]
        else:
            base = image_base_name(im["file_name"])
        groups.append(base)

    g2i: dict = defaultdict(list)
    for i, g in enumerate(groups):
        g2i[g].append(i)

    uniq = sorted(g2i.keys())
    rng = random.Random(seed)
    rng.shuffle(uniq)

    fold_of = [-1] * len(images)
    for fi, grp in enumerate(uniq):
        f = fi % n_folds
        for idx in g2i[grp]:
            fold_of[idx] = f
    return fold_of


def fold_splits(fold_of: list, fold: int):
    train_idx = [i for i, f in enumerate(fold_of) if f != fold]
    val_idx = [i for i, f in enumerate(fold_of) if f == fold]
    return train_idx, val_idx
