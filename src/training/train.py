"""Training loop: AMP + EMA + cosine schedule + GroupKFold CV.

Designed to run on a GPU (Kaggle T4). On CPU it still executes (for code-path
testing) but is far too slow for real training.
"""
from __future__ import annotations

import os
import sys
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import FilamentDataset
from src.data.transforms import get_augmentations
from src.models.builder import build_model, split_output
from src.models.losses import FilamentLoss
from src.training.folds import make_group_folds, fold_splits
from src.utils.io_utils import get_logger, set_seed
from src.utils.metrics import dice_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def state_dict(self):
        return self.shadow


def _val_dice(model, loader, device, use_spine):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            out = model(x)
            fil, _ = split_output(out, use_spine)
            prob = torch.sigmoid(fil)
            pred = (prob > 0.5).float()
            for i in range(pred.shape[0]):
                d = dice_score(pred[i, 0].cpu().numpy(), y[i, 0].cpu().numpy())
                tot += d
                n += 1
    return tot / max(n, 1)


def train_fold(cfg: dict, fold: int, logger):
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training fold {fold} on {device}")

    # Auto-correct guessed data paths against the real Kaggle mount if needed.
    from src.data.explore import resolve_train_paths
    mount = "/kaggle/input/filament-segmentation-2026"
    cfg["data"] = resolve_train_paths(cfg, mount)

    root = cfg["data"]["root"]
    ann = cfg["data"]["train_ann_file"]
    img_dir = cfg["data"]["train_images_dir"]
    use_spine = bool(cfg["data"].get("use_spine_aux", True))

    fold_of = make_group_folds(cfg, root, ann)
    train_idx, val_idx = fold_splits(fold_of, fold)
    logger.info(f"fold {fold}: train={len(train_idx)} val={len(val_idx)}")

    train_ds = FilamentDataset(root, ann, img_dir, cfg, mode="train",
                               transforms=get_augmentations(), fold_indices=train_idx)
    val_ds = FilamentDataset(root, ann, img_dir, cfg, mode="train",
                             transforms=None, fold_indices=val_idx) if val_idx else None

    bs = int(cfg["training"]["batch_size"])
    acc = int(cfg["training"].get("accumulate_grad_batches", 1))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2) if val_ds else None

    model = build_model(cfg).to(device)
    loss_fn = FilamentLoss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                            weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = int(cfg["training"]["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")
    ema = EMA(model, decay=float(cfg["training"].get("ema_decay", 0.9998))) if cfg["training"].get("ema", True) else None

    out_dir = os.path.join(cfg["experiment"]["out_dir"], "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{cfg['experiment']['name']}_fold{fold}_best.pth")
    best_dice = -1.0

    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            sp = batch["spine"].to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x)
                fil, spine = split_output(out, use_spine)
                loss = loss_fn(fil, y, spine, sp if use_spine else None)
                loss = loss / acc
            scaler.scale(loss).backward()
            if (bi + 1) % acc == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run_loss += float(loss) * acc

        sched.step()
        if val_loader is not None:
            vd = _val_dice(model, val_loader, device, use_spine)
            logger.info(f"epoch {ep+1}/{epochs} loss={run_loss/len(train_loader):.4f} val_dice={vd:.4f}")
        else:
            vd = -1.0
            logger.info(f"epoch {ep+1}/{epochs} loss={run_loss/len(train_loader):.4f} (no val fold)")

        save_now = (val_loader is None) or (vd > best_dice)
        if save_now:
            best_dice = max(best_dice, vd)
            state = ema.state_dict() if ema else model.state_dict()
            torch.save({"state_dict": state, "epoch": ep, "val_dice": vd, "cfg": cfg},
                       best_path)
            if val_loader is not None:
                logger.info(f"  -> saved best @ {vd:.4f}")

        # EMA update after epoch
        if ema:
            ema.update(model)

    logger.info(f"fold {fold} done. best val_dice={best_dice:.4f} -> {best_path}")
    return best_path


def main(cfg_path: str = "configs/base.yaml", fold: int | None = None):
    from src.utils.config import load_config
    cfg = load_config(cfg_path)
    logger = get_logger("train")
    if fold is None:
        fold = int(cfg["data"].get("holdout_fold", 0))
    train_fold(cfg, fold, logger)


if __name__ == "__main__":
    main()
