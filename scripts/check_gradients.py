#!/usr/bin/env python3
"""Check gradient flow for all parameters."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch

from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.utils.config import load_config


def main() -> None:
    diff_cfg = load_config(ROOT / "configs" / "diffusion.yaml")

    unet = ConditionalUNet(
        x_channels=diff_cfg["model"]["x_channels"],
        y_channels=diff_cfg["model"]["y_channels"],
        s_channels=diff_cfg["model"]["s_channels"],
        base_channels=diff_cfg["model"].get("base_channels", 64),
        depth=diff_cfg["model"].get("depth", 4),
        time_dim=diff_cfg["model"].get("time_dim", 256),
    )

    # Fake inputs
    b = 2
    y = torch.randn(b, diff_cfg["model"]["y_channels"], 256, 256)
    x_t = torch.randn(b, diff_cfg["model"]["x_channels"], 256, 256)
    s1 = torch.randn(b, diff_cfg["model"]["s_channels"], 256, 256)
    t = torch.randint(0, diff_cfg["schedule"].get("timesteps", 1000), (b,))

    # Forward
    unet_out = unet(x_t, t, y, s1)
    
    # Backward with dummy loss
    loss = unet_out.sum()
    loss.backward()

    # Check gradients
    print("=" * 60)
    print("Gradient Check Results")
    print("=" * 60)
    
    zero_grad_params = []
    for name, param in unet.named_parameters():
        if param.grad is None:
            print(f"[ERROR] {name}: grad is None!")
            zero_grad_params.append(name)
        elif param.grad.abs().sum() == 0:
            print(f"[WARNING] {name}: grad is all zeros!")
            zero_grad_params.append(name)
    
    if zero_grad_params:
        print("\n" + "=" * 60)
        print(f"Found {len(zero_grad_params)} parameters with zero/None gradients:")
        for name in zero_grad_params:
            print(f"  - {name}")
    else:
        print("\n[OK] All parameters have non-zero gradients!")


if __name__ == "__main__":
    main()
