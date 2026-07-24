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
# Kaggle runs this file as /kaggle/src/script.py, so sys.path[0] is /kaggle/src
# (NOT our cwd). Make the cloned repo importable regardless of cwd.
sys.path.insert(0, REPO_DIR)
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
    # 1) Clone the public solution repo (no auth needed). Always re-clone into a
    #    clean dir so re-runs in the SAME session pick up the latest GitHub code
    #    instead of silently reusing a stale checkout.
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    os.chdir(REPO_DIR)

    # 2) Install only what Kaggle's base image lacks, NEVER replacing torch.
    # Every torch-pulling package is installed with --no-deps so pip cannot
    # swap Kaggle's CUDA torch for a CPU-only build.
    import subprocess as _sp

    def _torch_info():
        r = _sp.run([sys.executable, "-c",
                     "import torch;print(torch.__version__,torch.cuda.is_available(),getattr(torch.version,'cuda',None))"],
                    capture_output=True, text=True)
        return (r.stdout or r.stderr).strip()

    print("=== env BEFORE pip install ===")
    print("torch:", _torch_info())

    run([sys.executable, "-m", "pip", "install", "-q",
         "albumentations>=1.3", "scikit-image>=0.21", "scipy>=1.10",
         "opencv-python-headless>=4.7", "pyyaml>=6.0"], check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "segmentation-models-pytorch>=0.3.3", "timm>=0.9",
         "einops", "pretrainedmodels"], check=False)
    run([sys.executable, "-m", "pip", "install", "-q",
         "huggingface-hub"], check=False)

    print("=== env AFTER pip install ===")
    print("torch:", _torch_info())

    # 2b) Ensure a CUDA torch is present. If the base image only ships a CPU
    # torch (or none), install one from the PyTorch CUDA index. This runs in a
    # subprocess BEFORE this process imports torch, so the cached module is the
    # correct (CUDA) build when training later does `import torch`.
    _ensure = (
        "import subprocess,sys\n"
        "def ok():\n"
        "    try:\n"
        "        import torch\n"
        "        return torch.cuda.is_available()\n"
        "    except Exception:\n"
        "        return False\n"
        "if ok():\n"
        "    print('CUDA torch already present; keeping it.')\n"
        "else:\n"
        "    print('No CUDA torch -> installing from pytorch cu121 index...')\n"
        "    subprocess.run([sys.executable,'-m','pip','install','-q','torch','torchvision','--index-url','https://download.pytorch.org/whl/cu121'],check=False)\n"
        "    print('post-install cuda available:', ok())\n"
    )
    run([sys.executable, "-c", _ensure], check=False)

    # 2c) Diagnostics: what is actually mounted + is GPU available?
    import torch
    print("=== environment diagnostics ===")
    print("torch:", torch.__version__,
          "| cuda available:", torch.cuda.is_available(),
          "| cuda", getattr(torch.version, "cuda", None))
    print("--- /kaggle/input tree (depth<=2) ---")
    if os.path.isdir("/kaggle/input"):
        for dp, dirs, _ in os.walk("/kaggle/input"):
            if dp[len("/kaggle/input"):].count(os.sep) <= 2:
                print(f"  {dp}  (dirs: {dirs[:8]})")
    else:
        print("  /kaggle/input DOES NOT EXIST")

    # 2d) Dump the REAL data layout + auto-discover the COCO json under the
    #     actual mount (we no longer hardcode the competition slug).
    from src.data.explore import print_structure, find_coco_json
    coco = find_coco_json("/kaggle/input")
    print("discovered COCO json:", coco)
    if coco is None:
        print("FATAL: competition data not found under /kaggle/input; aborting.")
        return
    print_structure(os.path.dirname(coco))

    if not torch.cuda.is_available():
        print("WARN: CUDA not available even after ensure step.")
        print("  -> a GPU was NOT allocated (accelerator disabled or weekly "
              "GPU quota exhausted). Enable GPU / wait for quota reset.")
        print("Aborting early to avoid a wasted CPU run.")
        return

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
