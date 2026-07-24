# Solar Filament Segmentation 2026 — Competitive Solution

A modular, reproducible pipeline for the
[Kaggle Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
Targets Top-3 via **high-recall semantic segmentation + spine-guided instance
separation**, multi-backbone ensembling, TTA, and a faithful local replica of the
official (non-standard) metric.

> ⚠️ **Hardware note:** Training requires a GPU. This codebase is designed to run
> inside a **Kaggle Notebook** (free T4/P100). A MacBook Air M2 / 8 GB **cannot**
> train it. Use your Mac only for editing/orchestration; run training on Kaggle.

---

## 1. Critical findings (read before tuning)

1. **The public leaderboard metric is suspected non-standard** ("all-empty masks
   score > 0.9"). → Do **not** trust absolute LB numbers; tune on the local metric
   (`src/utils/metrics.py`). Validate against the official scorer once you can run
   on Kaggle. Never submit empty/degenerate predictions.
2. **Test set may overlap the public MAGFiLO release.** → Run
   `src/submission/check_test_overlap.py` (on Kaggle, after you have the test
   images) to cross-reference filenames. If permitted, public GT can be leveraged.

See `ARCHITECTURE.md` for the full design rationale.

---

## 2. Repository layout

```
filament-segmentation-2026/
├── configs/
│   └── base.yaml                # all hyperparameters (override per model)
├── src/
│   ├── utils/
│   │   ├── rle.py               # pycocotools RLE <-> mask, polygon rasterization
│   │   ├── metrics.py           # local Dice/IoU + matching + one-to-many/many-to-one
│   │   └── io_utils.py          # json/image io, seed, logging
│   ├── data/
│   │   ├── dataset.py           # COCO json parsing, multi-annotator, crop sampling
│   │   ├── transforms.py        # Albumentations pipeline
│   │   └── preprocessing.py     # CLAHE, solar-disk ROI estimation
│   ├── models/
│   │   ├── builder.py           # SMP U-Net (ConvNeXt/Swin/MiT) 2-head
│   │   └── losses.py            # Dice/Focal/Tversky/Lovasz + spine aux
│   ├── training/
│   │   ├── folds.py             # GroupKFold by image base (no leakage)
│   │   └── train.py             # AMP + EMA + cosine training loop + CV
│   ├── inference/
│   │   ├── predictor.py         # sliding-window + TTA
│   │   └── ensemble.py          # multi-checkpoint / multi-backbone fusion
│   ├── postprocess/
│   │   ├── instance.py          # distance-transform / spine-seeded watershed
│   │   └── filter.py            # area filtering, morphology (anti-fragmentation)
│   └── submission/
│       ├── make_submission.py   # instances -> CSV (RLE counts)
│       └── check_test_overlap.py# (TODO on Kaggle) public-MAGFiLO overlap check
├── run_train.py                 # train (single fold / full)
├── run_infer.py                 # inference -> submission
├── run_submit.py                # (optional) re-pack submission
├── notebooks/
│   └── kaggle_run.py            # end-to-end orchestration for Kaggle
├── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

---

## 3. How to run (Kaggle Notebook)

```python
# Cell 1 — install deps
!pip install -q -r requirements.txt
!git clone https://github.com/<you>/filament-segmentation-2026.git
import sys; sys.path.insert(0, "filament-segmentation-2026")

# Cell 2 — train (edit configs/base.yaml paths to /kaggle/input/...)
from run_train import main as train_main
train_main("configs/base.yaml")   # trains fold, saves best checkpoint + EMA

# Cell 3 — infer + submit
from run_infer import main as infer_main
infer_main("configs/base.yaml")    # writes submission.csv
```

Or use the all-in-one orchestrator:

```python
from notebooks.kaggle_run import main
main(config_path="configs/base.yaml", do_train=True, do_infer=True)
```

### Local (Mac) — EDA / metric dev only (no training)
```bash
python -m src.utils.metrics   # self-test of the local metric
python src/submission/check_test_overlap.py --test-dir <dir> --magfilo-index <csv>
```

---

## 4. Tuning checklist (competitive)

- [ ] Local metric correlates with public LB (sanity-check a few submissions).
- [ ] `min_area` grid-searched to minimize **one-to-many** without dropping real barbs.
- [ ] `use_spine_seed=true` reduces **many-to-one** vs plain distance watershed.
- [ ] Ensemble ≥ 2 backbones (ConvNeXt-B + Swin-B) via probability fusion.
- [ ] TTA (flips + rot90) on at 1024 tiles, overlap 0.25.
- [ ] EMA weights enabled; best checkpoint chosen by **local penalized Dice**.
- [ ] (Advanced) Pseudo-label high-confidence test predictions, retrain.
- [ ] Code modular + README reproduces end-to-end; Git repo public at close.
- [ ] Technical report documents preprocessing → architecture → postprocessing.

---

## 5. License & rules
- Accept the competition rules and provide a public repo + source + report to be
  eligible for prizes (see competition "Requirements").
- Respect the Open-Access Policy (repo public at close, kept public until winners announced).
