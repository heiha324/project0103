#!/usr/bin/env python3
"""Visualize RS-Net predictions on the test split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional for headless envs
    raise RuntimeError("matplotlib is required for visualization") from exc

from sarcloud.data.cloud_dataset import CloudCropDataset, WHUOriCropDataset
from sarcloud.models.rsnet import RSNet
from sarcloud.utils.config import load_config


def resolve_checkpoint(cfg: dict, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)

    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "outputs/rsnet"))
    direct = out_dir / "rsnet_best.pth"
    if direct.exists():
        return direct

    parent = out_dir.parent
    prefix = f"{out_dir.name}_"
    if parent.exists():
        runs = sorted([p for p in parent.glob(f"{prefix}*") if p.is_dir()], reverse=True)
        for run in runs:
            candidate = run / "rsnet_best.pth"
            if candidate.exists():
                return candidate

    raise FileNotFoundError("Could not find rsnet_best.pth; pass --checkpoint explicitly")


def build_dataset(cfg: dict, split: str) -> CloudCropDataset:
    data_cfg = cfg.get("test", cfg.get("val", cfg["data"]))
    dataset_type = data_cfg.get("dataset", "flat")
    common_kwargs = dict(
        image_ext=data_cfg.get("image_ext", ".npy"),
        mask_ext=data_cfg.get("mask_ext", ".npy"),
        crop_size=data_cfg.get("crop_size", 256),
        base_stride=data_cfg.get("base_stride", 128),
        jitter=0,
        cloud_min_ratio=0.0,
        cloud_keep_ratio=1.0,
        bands=data_cfg.get("bands"),
        s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
        s2_clip_max=data_cfg.get("s2_clip_max", 1.0 if dataset_type == "whu_ori" else 10000.0),
        augment=False,
    )
    if dataset_type == "whu_ori":
        return WHUOriCropDataset(
            root=data_cfg["root"],
            split=split,
            level_subdir=data_cfg.get("level_subdir", "level1_10m"),
            **common_kwargs,
        )
    return CloudCropDataset(
        root=data_cfg["root"],
        images_subdir=data_cfg["images_subdir"],
        masks_subdir=data_cfg["masks_subdir"],
        **common_kwargs,
    )


def select_indices(dataset: CloudCropDataset, min_cloud_ratio: float, num_samples: int, seed: int) -> list[int]:
    candidates = [i for i, rec in enumerate(dataset.records) if rec.cloud_ratio >= min_cloud_ratio]
    if len(candidates) < num_samples:
        raise RuntimeError(
            f"Only {len(candidates)} samples meet cloud ratio >= {min_cloud_ratio:.2f}"
        )
    rng = random.Random(seed)
    return rng.sample(candidates, num_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rsnet_whu_ori.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="Test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--min-cloud", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show-prob", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="eval/rsnet_test_vis.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = resolve_checkpoint(cfg, args.checkpoint)

    model = RSNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"].get("base_channels", 32),
        depth=cfg["model"].get("depth", 4),
        use_batchnorm=cfg["model"].get("use_batchnorm", True),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = build_dataset(cfg, args.split)
    indices = select_indices(dataset, args.min_cloud, args.num_samples, args.seed)

    items = [dataset[i] for i in indices]
    images = torch.stack([item[0] for item in items], dim=0)
    masks = [item[1] for item in items]

    with torch.no_grad():
        probs = torch.sigmoid(model(images.to(device))).cpu()

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

    for row, idx in enumerate(indices):
        image = images[row]
        mask = masks[row]
        prob = probs[row, 0]

        rgb = image[:3].permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0.0, 1.0)

        if args.show_prob:
            pred_vis = prob.numpy()
        else:
            pred_vis = (prob >= args.threshold).float().numpy()

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(mask.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row, 2].imshow(pred_vis, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

        ratio = dataset.records[idx].cloud_ratio * 100.0
        axes[row, 0].set_title("RGB")
        axes[row, 1].set_title(f"Mask {ratio:.1f}%")
        axes[row, 2].set_title("Pred")
        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
