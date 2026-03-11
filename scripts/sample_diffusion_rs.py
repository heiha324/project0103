#!/usr/bin/env python3
"""Sample Residual Shifting diffusion outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion.sampling_rs import sample_batch_rs
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.utils.config import load_config
from sarcloud.training.ema import EMA


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, ema: EMA | None = None) -> None:
    """加载模型权重 (支持 EMA)。"""
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # 优先加载 EMA 权重 (如果有)
    if ema is not None and "ema_state" in ckpt:
        print("Loading EMA state...")
        ema.shadow = ckpt["ema_state"]
        # 将 EMA 权重应用到当前模型 (仅用于推理)
        for name, param in model.named_parameters():
            if name in ema.shadow:
                param.data.copy_(ema.shadow[name])
    elif "model_state" in ckpt:
        print("Loading model state...")
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        print("Loading state dict direct...")
        model.load_state_dict(ckpt, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_rs.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/diffusion_rs/samples")
    parser.add_argument("--subset_size", type=int, default=0, help="仅采样前N个样本(调试用)")
    # 覆盖配置文件的参数
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    
    args = parser.parse_args()

    # --- Distributed Init ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo", init_method="env://")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)

    # Model
    model = ConditionalUNet(
        x_channels=cfg["model"]["x_channels"],
        y_channels=cfg["model"]["y_channels"],
        s_channels=cfg["model"]["s_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
        time_dim=cfg["model"].get("time_dim", 256),
    ).to(device)
    
    # EMA wrapper
    ema = EMA(model, decay=0.999) # decay value doesn't matter for loading
    load_checkpoint(model, args.checkpoint, ema=ema)
    model.eval()

    # Diffusion
    diff_cfg = cfg.get("diffusion", {})
    # 允许命令行参数覆盖配置
    kappa = args.kappa if args.kappa is not None else diff_cfg.get("kappa", 1.0)
    
    diffusion = ResidualShiftingDiffusion(
        timesteps=diff_cfg.get("timesteps", 1000),
        kappa=kappa,
        schedule_type=diff_cfg.get("schedule_type", "exponential"),
        min_eta=diff_cfg.get("min_eta", 0.001),
        max_eta=diff_cfg.get("max_eta", 0.99),
        power=diff_cfg.get("power", 2.0),
        x0_clip_min=cfg.get("schedule", {}).get("x0_clip_min", 0.0),
        x0_clip_max=cfg.get("schedule", {}).get("x0_clip_max", 1.0),
    )

    # Dataset
    data_cfg = cfg.get("test") or cfg.get("val") or cfg["sen12ms"]
    dataset_type = data_cfg.get("dataset", "npy")
    
    if dataset_type == "sen12mscr_raw":
        dataset = Sen12MSCRRawDataset(
            root=data_cfg["root"],
            alpha_root=data_cfg.get("alpha_root"),
            split_csv=data_cfg.get("split_csv"),
            split=data_cfg.get("split"), # test
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
            split=data_cfg["split"], # test
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
    if is_distributed:
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Slice Indices ---
    total_len = len(dataset)
    if args.subset_size > 0:
        total_len = min(total_len, args.subset_size)
        
    all_indices = list(range(total_len))
    
    if is_distributed:
        indices = all_indices[rank::world_size]
        if rank == 0:
            print(f"Rank {rank}/{world_size}: Processing {len(indices)} samples")
    else:
        indices = all_indices

    if tqdm is not None and (not is_distributed or rank == 0):
        indices_iter = tqdm(indices, desc="Sample", ncols=80)
    else:
        indices_iter = indices
    
    # Sampling Config
    sampling_cfg = cfg.get("sampling", {}).copy()
    if args.steps is not None:
        sampling_cfg["steps"] = args.steps
    if args.eta is not None:
        sampling_cfg["eta"] = args.eta
        
    steps = sampling_cfg.get("steps", 50)
    
    for idx in indices_iter:
        s1, s2_cloudy, s2_clear, _alpha = dataset[idx]
        s1 = s1.unsqueeze(0).to(device)
        y = s2_cloudy.unsqueeze(0).to(device)
        
        # Sample
        sample = sample_batch_rs(
            model,
            diffusion,
            y,
            s1,
            steps=steps,
            schedule_cfg=sampling_cfg,
        )
        
        # Save
        sample_np = sample.squeeze(0).cpu().numpy().astype(np.float32)
        np.save(output_dir / f"sample_{idx:05d}.npy", sample_np)

    if is_distributed:
        dist.barrier()
        if rank == 0:
            print(f"All ranks finished. Saved samples to {output_dir}")
        dist.destroy_process_group()
    else:
        print(f"Saved samples to {output_dir}")


if __name__ == "__main__":
    main()
