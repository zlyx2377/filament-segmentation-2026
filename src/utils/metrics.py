"""Local evaluation metric — a *faithful approximation* of the official scorer.

The official metric is reported to be non-standard (see ARCHITECTURE.md:
"all-empty masks score > 0.9"). We therefore replicate, as closely as possible:
  * per-instance Dice in the style of torchmetrics DiceScore
  * IoU-based bipartite matching (greedy, descending IoU, one-to-one)
  * counts of:
      - one_to_many   : one GT matched by >1 pred  (fragmentation of a filament)
      - many_to_one   : one pred matched by >1 GT  (over-merge of filaments)
      - unmatched_preds: predicted fragments with NO GT overlap (also fragmentation)
      - unmatched_gts : missed filaments
  * penalized_dice = mean_dice_all - penalty_weight * (sum of the above)

IMPORTANT: This is a proxy. Always validate against the official scorer on Kaggle
before trusting absolute numbers. Tune on `penalized_dice`.
"""
from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Binary Dice. Empty/empty -> 1.0; one empty -> 0.0 (torchmetrics behaviour)."""
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union_sum = pred.sum() + gt.sum()
    if union_sum == 0:
        return 1.0
    return float(2.0 * inter / (union_sum + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + eps))


def _iou_matrix(preds: list[np.ndarray], gts: list[np.ndarray]) -> np.ndarray:
    n, m = len(preds), len(gts)
    mat = np.zeros((n, m), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            mat[i, j] = iou_score(p, g)
    return mat


def match_instances(preds: list[np.ndarray], gts: list[np.ndarray], iou_thr: float = 0.5):
    """Greedy one-to-one matching + multiplicity / unmatched counts."""
    n, m = len(preds), len(gts)
    if n == 0 or m == 0:
        return {
            "matched": [],
            "one_to_many": 0, "many_to_one": 0,
            "unmatched_preds": n, "unmatched_gts": m,
            "n_pred": n, "n_gt": m,
        }
    iou = _iou_matrix(preds, gts)
    pairs = [(iou[i, j], i, j) for i in range(n) for j in range(m) if iou[i, j] >= iou_thr]
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    matched = []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, float(v)))

    one_to_many = int(sum(max(0, int((iou[:, j] >= iou_thr).sum()) - 1) for j in range(m)))
    many_to_one = int(sum(max(0, int((iou[i, :] >= iou_thr).sum()) - 1) for i in range(n)))
    return {
        "matched": matched,
        "one_to_many": one_to_many,
        "many_to_one": many_to_one,
        "unmatched_preds": n - len(used_p),
        "unmatched_gts": m - len(used_g),
        "n_pred": n, "n_gt": m,
    }


def evaluate_image(preds, gts, iou_thr: float = 0.5, penalty_weight: float = 0.05):
    """Evaluate one image. Returns per-image metrics dict."""
    res = match_instances(preds, gts, iou_thr)
    dices = [dice_score(preds[i], gts[j]) for i, j, _ in res["matched"]]
    mean_dice_matched = float(np.mean(dices)) if dices else 0.0

    matched_gt = {j for _, j, _ in res["matched"]}
    gt_dices = []
    for j in range(len(gts)):
        if j in matched_gt:
            i = [pi for pi, gj, _ in res["matched"] if gj == j][0]
            gt_dices.append(dice_score(preds[i], gts[j]))
        else:
            gt_dices.append(0.0)
    mean_dice_all = float(np.mean(gt_dices)) if gt_dices else 0.0

    frag = res["one_to_many"] + res["many_to_one"] + res["unmatched_preds"]
    penalized = max(0.0, mean_dice_all - penalty_weight * frag)

    return {
        "mean_dice_matched": mean_dice_matched,
        "mean_dice_all": mean_dice_all,
        "penalized_dice": penalized,
        "one_to_many": res["one_to_many"],
        "many_to_one": res["many_to_one"],
        "unmatched_preds": res["unmatched_preds"],
        "unmatched_gts": res["unmatched_gts"],
        "fragmentation": frag,
        "n_pred": res["n_pred"],
        "n_gt": res["n_gt"],
    }


def evaluate_dataset(preds_per_image, gts_per_image, iou_thr: float = 0.5, **kw):
    assert len(preds_per_image) == len(gts_per_image)
    agg = [evaluate_image(p, g, iou_thr, **kw) for p, g in zip(preds_per_image, gts_per_image)]
    sum_keys = ["one_to_many", "many_to_one", "unmatched_preds", "unmatched_gts",
                "fragmentation", "n_pred", "n_gt"]
    out = {}
    for k in ["mean_dice_matched", "mean_dice_all", "penalized_dice"]:
        out[k] = float(np.mean([a[k] for a in agg]))
    for k in sum_keys:
        out[k] = int(sum(a[k] for a in agg))
    return out


if __name__ == "__main__":
    a = np.zeros((100, 100), np.uint8); a[10:40, 10:40] = 1
    b = np.zeros((100, 100), np.uint8); b[12:42, 12:42] = 1
    c = np.zeros((100, 100), np.uint8); c[80:82, 80:82] = 1
    print("self-test ->", evaluate_image([b, c], [a]))
    print("dice(a,b)=", dice_score(a, b))
