#!/usr/bin/env python3
"""Generate fake data and sanity-check model shapes."""

from __future__ import annotations

import argparse
from typing import Any, List

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch

from sarcloud.models.cond_unet import ConditionalUNet, Downsample, ResBlock, Upsample
from sarcloud.models.rsnet import RSNet
from sarcloud.utils.config import load_config


def shape_repr(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        return str(tuple(obj.shape))
    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(shape_repr(x) for x in obj) + "]"
    return str(type(obj))


def attach_shape_hooks(model: torch.nn.Module, label: str) -> List[str]:
    logs: List[str] = []
    interesting = (
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.BatchNorm2d,
        torch.nn.GroupNorm,
        ResBlock,
        Downsample,
        Upsample,
    )

    def hook(name: str):
        def _fn(module, inputs, output):
            in_shapes = shape_repr(inputs)
            out_shape = shape_repr(output)
            logs.append(f"{label}:{name} in={in_shapes} out={out_shape}")
        return _fn

    for name, module in model.named_modules():
        if isinstance(module, interesting):
            module.register_forward_hook(hook(name))

    return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsnet-config", type=str, default="configs/rsnet.yaml")
    parser.add_argument("--diffusion-config", type=str, default="configs/diffusion.yaml")
    args = parser.parse_args()

    rsnet_cfg = load_config(args.rsnet_config)
    diff_cfg = load_config(args.diffusion_config)

    rsnet = RSNet(
        in_channels=rsnet_cfg["model"]["in_channels"],
        base_channels=rsnet_cfg["model"].get("base_channels", 32),
        depth=rsnet_cfg["model"].get("depth", 4),
        use_batchnorm=rsnet_cfg["model"].get("use_batchnorm", True),
    )

    unet = ConditionalUNet(
        x_channels=diff_cfg["model"]["x_channels"],
        y_channels=diff_cfg["model"]["y_channels"],
        s_channels=diff_cfg["model"]["s_channels"],
        base_channels=diff_cfg["model"].get("base_channels", 64),
        depth=diff_cfg["model"].get("depth", 4),
        time_dim=diff_cfg["model"].get("time_dim", 256),
    )

    rsnet_logs = attach_shape_hooks(rsnet, "RSNet")
    unet_logs = attach_shape_hooks(unet, "CondUNet")

    # Fake inputs
    b = 2
    rsnet_in = torch.randn(b, rsnet_cfg["model"]["in_channels"], 256, 256)
    y = torch.randn(b, diff_cfg["model"]["y_channels"], 256, 256)
    x_t = torch.randn(b, diff_cfg["model"]["x_channels"], 256, 256)
    s1 = torch.randn(b, diff_cfg["model"]["s_channels"], 256, 256)
    t = torch.randint(0, diff_cfg["schedule"].get("timesteps", 1000), (b,))

    with torch.no_grad():
        rsnet_out = rsnet(rsnet_in)
        unet_out = unet(x_t, t, y, s1)

    print("RSNet output:", rsnet_out.shape)
    print("Conditional U-Net output:", unet_out.shape)
    print("\nLayer shape trace:")
    for line in rsnet_logs + unet_logs:
        print(line)


if __name__ == "__main__":
    main()
