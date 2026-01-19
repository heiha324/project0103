#!/usr/bin/env python3
"""Temperature scaling calibration for RS-Net."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from sarcloud.data.cloud_dataset import CloudCropDataset, WHUOriCropDataset
from sarcloud.models.rsnet import RSNet
from sarcloud.utils.config import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rsnet.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/rsnet/rsnet_best.pth")
    parser.add_argument("--output", type=str, default="outputs/rsnet/rsnet_calibration.json")
    parser.add_argument("--max-samples", type=int, default=2000000)
    parser.add_argument("--sample-per-batch", type=int, default=8192)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("val", cfg["data"])

    dataset_type = data_cfg.get("dataset", "flat")
    crop_size = data_cfg.get("crop_size", cfg["data"].get("crop_size", 256))
    base_stride = data_cfg.get("base_stride", cfg["data"].get("base_stride", 128))
    bands = data_cfg.get("bands", cfg["data"].get("bands"))
    s2_clip_min = data_cfg.get("s2_clip_min", cfg["data"].get("s2_clip_min", 0.0))
    s2_clip_max = data_cfg.get("s2_clip_max", cfg["data"].get("s2_clip_max", 10000.0))

    if dataset_type == "whu_ori":
        dataset = WHUOriCropDataset(
            root=data_cfg["root"],
            split=data_cfg.get("split", "Val"),
            level_subdir=data_cfg.get("level_subdir", "level1_10m"),
            image_ext=data_cfg.get("image_ext", ".npy"),
            mask_ext=data_cfg.get("mask_ext", ".npy"),
            crop_size=crop_size,
            base_stride=base_stride,
            jitter=0,
            cloud_min_ratio=0.0,
            cloud_keep_ratio=1.0,
            bands=bands,
            s2_clip_min=s2_clip_min,
            s2_clip_max=s2_clip_max,
            augment=False,
        )
    else:
        dataset = CloudCropDataset(
            root=data_cfg["root"],
            images_subdir=data_cfg["images_subdir"],
            masks_subdir=data_cfg["masks_subdir"],
            image_ext=data_cfg.get("image_ext", ".npy"),
            mask_ext=data_cfg.get("mask_ext", ".npy"),
            crop_size=crop_size,
            base_stride=base_stride,
            jitter=0,
            cloud_min_ratio=0.0,
            cloud_keep_ratio=1.0,
            bands=bands,
            s2_clip_min=s2_clip_min,
            s2_clip_max=s2_clip_max,
            augment=False,
        )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = RSNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"].get("base_channels", 32),
        depth=cfg["model"].get("depth", 4),
        use_batchnorm=cfg["model"].get("use_batchnorm", True),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logits_list = []
    labels_list = []
    max_samples = int(args.max_samples)
    sample_per_batch = int(args.sample_per_batch)
    rng = torch.Generator().manual_seed(args.sample_seed)
    total_samples = 0
    with torch.no_grad():
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc="Calibrate", ncols=80, leave=False)
        for images, masks in iterator:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            logits_flat = logits.detach().flatten().cpu()
            labels_flat = masks.detach().flatten().cpu()
            if sample_per_batch > 0 and logits_flat.numel() > sample_per_batch:
                idx = torch.randperm(logits_flat.numel(), generator=rng)[:sample_per_batch]
                logits_flat = logits_flat[idx]
                labels_flat = labels_flat[idx]
            logits_list.append(logits_flat)
            labels_list.append(labels_flat)
            total_samples += logits_flat.numel()
            if max_samples > 0 and total_samples >= max_samples:
                break

    logits = torch.cat(logits_list).float()
    labels = torch.cat(labels_list).float()

    log_t = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).detach().cpu())

    save_json(args.output, {"temperature": temperature})
    print(f"Saved temperature {temperature:.4f} to {args.output}")


if __name__ == "__main__":
    main()
