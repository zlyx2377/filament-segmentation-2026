"""Model builder: SMP U-Net with a 2-head output (filament + spine aux).

Encoder is switchable via config (convnext_base / swin_base / mit_b5) so we can
ensemble diverse backbones. Input is 3-channel (see data/preprocessing.py).
"""
from __future__ import annotations

import torch
import segmentation_models_pytorch as smp


def build_model(cfg: dict) -> torch.nn.Module:
    mcfg = cfg["model"]
    enc = mcfg["encoder"]
    use_spine = bool(cfg["data"].get("use_spine_aux", True))
    classes = 1 + (1 if use_spine else 0)
    weights = "imagenet" if mcfg.get("pretrained", True) else None
    model = smp.Unet(
        encoder_name=enc,
        encoder_weights=weights,
        in_channels=3,
        classes=classes,
        decoder_channels=tuple(mcfg.get("decoder_channels", [256, 128, 64, 32, 16])),
    )
    return model


def split_output(out: torch.Tensor, use_spine: bool):
    """Split 2-head output. Returns (filament_logits, spine_logits|None)."""
    fil = out[:, 0:1]
    spi = out[:, 1:2] if use_spine and out.shape[1] > 1 else None
    return fil, spi
