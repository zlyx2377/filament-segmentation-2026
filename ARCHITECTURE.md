# Architecture Decisions — Filament Segmentation 2026

> Goal: maximize the official score (70% quantitative = Mean Dice + fragmentation /
> over-segmentation penalty distributions; 30% qualitative = pipeline + morphology + code).

## Two findings that shape everything (from the Discussion board, 2026-07-24)

1. **"Possible evaluation issue: all-empty masks score > 0.9"**
   The public leaderboard metric appears non-standard (an empty submission
   should score ~0 under normal Dice). Implications:
   - **Never trust absolute LB numbers.** Build a faithful local replica and tune on it.
   - Our local `src/utils/metrics.py` mirrors `torchmetrics DiceScore` + IoU-based
     bipartite matching + one-to-many / many-to-one counting. Treat it as an
     *approximation* until we can validate against the official scorer on Kaggle.
   - We will NOT submit degenerate (empty) predictions — they are unethical and,
     if the bug is fixed, will score 0.

2. **"Test set overlap with public dataset" (7 upvotes)**
   The test images may be a subset of the public MAGFiLO release
   (Nature Scientific Data, 10.1038/s41597-024-03876-y). If so, their GT masks
   could already be public → a potential "free labels" angle.
   - `src/submission/check_test_overlap.py` (TODO on Kaggle) cross-references
     test image filenames against the public MAGFiLO index. If a match is found,
     we can (a) sanity-check our model and (b) consider using public masks —
     **only after confirming this is permitted** (it would be external data).

## Core design: high-recall semantic segmentation + spine-guided instance separation

Why not Mask R-CNN / Mask2Former directly?
- The dominant metric is **Dice (overlap)**, which a clean semantic mask maximizes.
- The penalties are about *granularity*: one GT ↔ one Pred. A semantic mask +
  **watershed seeded by the predicted spine** gives us clean per-filament instances
  while keeping Dice high and suppressing both fragmentation (one-to-many) and
  over-merge (many-to-one).
- Spine annotations are provided "for free" and carry barb orientation + instance
  boundary cues → using them as an auxiliary head is a cheap, defensible edge.

### Pipeline
```
H-Alpha JPEG (2048², grayscale)
  → physical-prior preprocessing (normalize, CLAHE, solar-disk ROI mask)
  → 2-head model: [filament prob | spine prob]   (U-Net + ConvNeXt/Swin/SegFormer)
  → sliding-window + TTA inference → fused probability maps
  → threshold → binary filament mask
  → instance separation: distance-transform / spine-seeded watershed
  → post-filter: drop tiny fragments (cut one-to-many); NO blind merge (avoid many-to-one)
  → per image: N instances → pycocotools RLE → CSV
```

### Where the points are won
| Lever | Why it matters | Status |
|---|---|---|
| Backbone (ConvNeXt-B / Swin-B / MiT-B5) | stronger features → higher Dice | config-driven |
| Spine auxiliary head | better instance separation, preserves barbs | implemented |
| Solar-disk ROI | kills limb-darkening false positives | implemented |
| TTA + sliding window + ensemble | +1–3% Dice | implemented |
| EMA weights | stabler, better gen | implemented |
| Local faithful metric | tune what's actually scored | implemented |
| Pseudo-labeling test set | self-training boost (advanced) | scaffolded, off by default |
| Code quality + report (30%) | reproducibility wins ties | README + modular code |

## Reproducibility & the 30% qualitative score
- Public Git repo required at close; source + report required for prize.
- Code is modular (`src/data`, `src/models`, `src/training`, `src/inference`,
  `src/postprocess`, `src/submission`), documented, config-driven, deterministic (seed).
- A `notebooks/kaggle_run.py` orchestrates the full pipeline end-to-end.
