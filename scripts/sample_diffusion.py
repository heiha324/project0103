#!/usr/bin/env python3
"""Sample SAR-guided diffusion outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset
from sarcloud.diffusion.gaussian import GaussianDiffusion
from sarcloud.diffusion.sampling import sample_batch
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.utils.config import load_config


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path) -> None:
    """加载模型权重，支持多种 checkpoint 格式。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "ema_state" in ckpt:
        state_dict = ckpt["ema_state"]
    else:
        state_dict = ckpt
    
    # 加载并报告不匹配的 keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/diffusion/diffusion_ema.pth")
    parser.add_argument("--output", type=str, default="outputs/diffusion/samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalUNet(
        x_channels=cfg["model"]["x_channels"],
        y_channels=cfg["model"]["y_channels"],
        s_channels=cfg["model"]["s_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
        time_dim=cfg["model"].get("time_dim", 256),
    ).to(device)
    load_checkpoint(model, args.checkpoint)

    schedule_cfg = cfg["schedule"]
    diffusion = GaussianDiffusion(
        timesteps=schedule_cfg["timesteps"],
        schedule_type=schedule_cfg.get("type", "cosine"),
        beta_start=schedule_cfg.get("beta_start", 1e-4),
        beta_end=schedule_cfg.get("beta_end", 2e-2),
        device=device,
        x0_clip_min=schedule_cfg.get("x0_clip_min", -1.0),
        x0_clip_max=schedule_cfg.get("x0_clip_max", 2.0),
    )

    data_cfg = cfg["sen12ms"]
    dataset_type = data_cfg.get("dataset", "npy")
    if dataset_type == "sen12mscr_raw":
        dataset = Sen12MSCRRawDataset(
            root=data_cfg["root"],
            alpha_root=data_cfg.get("alpha_root"),
            split_csv=data_cfg.get("split_csv"),
            split=data_cfg.get("split"),
            bands=data_cfg.get("bands"),
            s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
            s2_clip_max=data_cfg.get("s2_clip_max", 10000.0),
            s1_db_min=data_cfg.get("s1_db_min", -25.0),
            s1_db_max=data_cfg.get("s1_db_max", 0.0),
            alpha_ext=data_cfg.get("alpha_ext", ".npy"),
            roi_glob=data_cfg.get("roi_glob"),
        )
    else:
        dataset = Sen12MSCRDataset(
            root=data_cfg["root"],
            split=data_cfg["split"],
            s1_subdir=data_cfg["s1_subdir"],
            s2_cloudy_subdir=data_cfg["s2_cloudy_subdir"],
            s2_clear_subdir=data_cfg["s2_clear_subdir"],
            alpha_subdir=data_cfg.get("alpha_subdir"),
            image_ext=data_cfg.get("image_ext", ".npy"),
            bands=data_cfg.get("bands"),
            s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
            s2_clip_max=data_cfg.get("s2_clip_max", 10000.0),
            s1_db_min=data_cfg.get("s1_db_min", -25.0),
            s1_db_max=data_cfg.get("s1_db_max", 0.0),
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = range(len(dataset))
    if tqdm is not None:
        indices = tqdm(indices, desc="Sample", ncols=80)
    
    sampling_cfg = cfg["sampling"]
    init_method = sampling_cfg.get("init_method", "noise")
    noise_ratio = float(sampling_cfg.get("noise_ratio", 1.0))
    
    for idx in indices:
        s1, s2_cloudy, s2_clear, _alpha = dataset[idx]
        s1 = s1.unsqueeze(0).to(device)
        y = s2_cloudy.unsqueeze(0).to(device)
        sample = sample_batch(
            model,
            diffusion,
            y,
            s1,
            steps=sampling_cfg.get("steps", 50),
            schedule_cfg=sampling_cfg,
            init_method=init_method,
            noise_ratio=noise_ratio,
        )
        sample_np = sample.squeeze(0).cpu().numpy().astype(np.float32)
        np.save(output_dir / f"sample_{idx:05d}.npy", sample_np)

    print(f"Saved samples to {output_dir}")


if __name__ == "__main__":
    main()
