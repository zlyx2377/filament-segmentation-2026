"""Composite loss: Dice + Focal + Tversky + Lovasz (filament head)
+ optional auxiliary spine loss. Weights come from config."""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class FilamentLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lw = cfg["training"]["loss"]
        self.w_dice = float(lw.get("dice", 1.0))
        self.w_focal = float(lw.get("focal", 0.5))
        self.w_tversky = float(lw.get("tversky", 0.5))
        self.w_lovasz = float(lw.get("lovasz", 0.5))
        beta = float(lw.get("tversky_beta", 0.7))  # >0.5 -> favors recall

        self.dice = smp.losses.DiceLoss(mode="binary")
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.tversky = smp.losses.TverskyLoss(mode="binary", alpha=1.0 - beta, beta=beta)
        self.lovasz = smp.losses.LovaszLoss(mode="binary")

        self.use_spine = bool(cfg["data"].get("use_spine_aux", True))
        self.w_spine = float(cfg["training"].get("spine_loss_weight", 0.3))
        if self.use_spine:
            self.spine_dice = smp.losses.DiceLoss(mode="binary")
            self.spine_focal = smp.losses.FocalLoss(mode="binary")

    def forward(self, fil_logits, fil_gt, spine_logits=None, spine_gt=None):
        fil_prob = torch.sigmoid(fil_logits)
        loss = (
            self.w_dice * self.dice(fil_prob, fil_gt)
            + self.w_focal * self.focal(fil_logits, fil_gt)
            + self.w_tversky * self.tversky(fil_prob, fil_gt)
            + self.w_lovasz * self.lovasz(fil_logits, fil_gt)
        )
        if self.use_spine and spine_logits is not None and spine_gt is not None:
            sp_prob = torch.sigmoid(spine_logits)
            loss = loss + self.w_spine * (
                self.spine_dice(sp_prob, spine_gt) + self.spine_focal(spine_logits, spine_gt)
            )
        return loss
