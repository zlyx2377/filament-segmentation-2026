"""Kaggle script-kernel entrypoint for the Filament Segmentation 2026 solution.

Runs end-to-end:
  1. clone the (public) solution repo
  2. install only deps Kaggle's GPU image lacks
  3. train a holdout fold  ->  sliding-window inference + TTA
     -> spine-guided watershed instance separation  -> build submission.csv
  4. sanity-check the CSV, then auto-submit (Kaggle kernels are pre-authenticated,
     so `kaggle competitions submit` works with no extra credential setup).

The submit step is guarded: we never burn a submission slot on an empty or
degenerate CSV.
"""
import os
import sys
import shutil
import subprocess

REPO_URL = "https://github.com/zlyx2377/filament-segmentation-2026.git"
# Kaggle script kernels start in a READ-ONLY dir (/kaggle/src). All writes
# (git clone, training artifacts, submission) must go under /kaggle/working.
WORK = "/kaggle/working"
REPO_DIR = os.path.join(WORK, "filament-segmentation-2026")
CONFIG = "configs/kaggle.yaml"          # fast validation config (30 epochs)


def run(cmd, **kw):
    print(">>>", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def sanity_ok(path: str) -> bool:
    import pandas as pd
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa
        print("sanity: cannot read csv ->", e)
        return False
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        print("sanity: unexpected columns ->", list(df.columns))
        return False
    if len(df) == 0:
        print("sanity: empty csv")
        return False
    non_empty = df["segmentation"].notna() & (df["segmentation"].astype(str).str.strip() != "")
    frac = non_empty.mean()
    print(f"sanity: {len(df)} rows, {frac:.1%} with masks")
    # block all-empty / near-empty degenerate outputs (frac < 0.1),
    # but allow legitimately sparse predictions.
    return 0.1 <= frac <= 1.0


def main():
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)
    # 1) Clone the public solution repo (no auth needed).
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    os.chdir(REPO_DIR)

    # 2) Install only what Kaggle's base image lacks.
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         "requirements_kaggle.txt"], check=False)

    # 2b) Dump the REAL data layout to the kernel log (so we can adapt if our
    #     guessed paths in base.yaml are wrong). Harmless once data is ready.
    from src.data.explore import print_structure
    print_structure("/kaggle/input/filament-segmentation-2026")

    # 3) End-to-end train + infer (writes submission.csv in repo root).
    from notebooks.kaggle_run import main as kaggle_run_main
    kaggle_run_main(CONFIG, do_train=True, do_infer=True)

    # 4) Copy to /kaggle/working so it is the kernel output.
    csv_src = os.path.join(os.getcwd(), "submission.csv")
    csv_dst = "/kaggle/working/submission.csv"
    if not os.path.exists(csv_src):
        print("ERROR: submission.csv was not produced. Aborting submit.")
        return
    shutil.copy(csv_src, csv_dst)

    # 5) Guard before submitting.
    if not sanity_ok(csv_dst):
        print("WARN: submission failed sanity check; NOT submitting. "
              "Inspect /kaggle/working/submission.csv.")
        return

    # 6) Auto-submit (kernel is pre-authenticated under your account).
    msg = ("auto-submit: convnext_b holdout fold, spine-guided watershed, "
           "30-epoch validation run")
    r = run(["kaggle", "competitions", "submit",
             "-c", "filament-segmentation-2026",
             "-f", csv_dst, "-m", msg],
            capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)
    print("submit exit code:", r.returncode)


if __name__ == "__main__":
    main()
