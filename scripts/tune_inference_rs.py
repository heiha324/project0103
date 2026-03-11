#!/usr/bin/env python3
"""Tune inference parameters for Residual Shifting Diffusion (DDP Enabled)."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset, collate_sen12mscr
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion.sampling_rs import sample_batch_rs
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.utils.config import load_config
from sarcloud.training.ema import EMA
from sarcloud.utils.metrics import psnr, ssim

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            dist.init_process_group(backend="gloo", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def load_model(args, cfg, device):
    print("Loading model...")
    model = ConditionalUNet(
        x_channels=cfg["model"]["x_channels"],
        y_channels=cfg["model"]["y_channels"],
        s_channels=cfg["model"]["s_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
        time_dim=cfg["model"].get("time_dim", 256),
    ).to(device)
    
    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # Load EMA
    ema = EMA(model, decay=0.999)
    if "ema_state" in ckpt:
        print("Using EMA weights...")
        ema.shadow = ckpt["ema_state"]
        for name, param in model.named_parameters():
            if name in ema.shadow:
                param.data.copy_(ema.shadow[name])
    else:
        print("WARNING: EMA state not found, using standard weights.")
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        
    model.eval()
    return model

def get_dataloader(args, cfg, rank, world_size, ddp):
    data_cfg = cfg.get("test") or cfg["sen12ms"]
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

    # Subset for tuning
    total_len = len(dataset)
    num_samples = min(args.num_samples, total_len)
    
    # Use deterministic subset for consistency across runs
    np.random.seed(42)
    indices = np.random.choice(total_len, num_samples, replace=False)
    subset = Subset(dataset, indices)
    
    sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
    
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_sen12mscr
    )
    
    return loader

def evaluate_subset(model, loader, diffusion_cls, cfg, device, steps, kappa, eta):
    """Run sampling on loader and return avg metrics."""
    
    diff_cfg = cfg.get("diffusion", {})
    diffusion = diffusion_cls(
        timesteps=diff_cfg.get("timesteps", 1000),
        kappa=kappa, 
        schedule_type=diff_cfg.get("schedule_type", "exponential"),
        min_eta=diff_cfg.get("min_eta", 0.001),
        max_eta=diff_cfg.get("max_eta", 0.99),
        power=diff_cfg.get("power", 2.0),
        x0_clip_min=cfg.get("schedule", {}).get("x0_clip_min", 0.0),
        x0_clip_max=cfg.get("schedule", {}).get("x0_clip_max", 1.0),
    )
    
    schedule_cfg = {"eta": eta, "method": "ddim"}
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    for s1, s2_cloudy, s2_clear, _ in loader:
        s1 = s1.to(device)
        y = s2_cloudy.to(device)
        x0 = s2_clear.to(device)
        
        with torch.no_grad():
            sample = sample_batch_rs(
                model, diffusion, y, s1,
                steps=steps,
                schedule_cfg=schedule_cfg
            )
            sample = torch.clamp(sample, 0.0, 1.0)
            
            # Batch metrics
            for i in range(sample.size(0)):
                total_psnr += psnr(sample[i], x0[i])
                total_ssim += ssim(sample[i], x0[i])
                count += 1
                
    return total_psnr, total_ssim, count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_rs.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples for tuning")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    
    cfg = load_config(args.config)
    model = load_model(args, cfg, device)
    
    # Broadcast model for DDP consistency (not strictly needed for eval but good practice)
    # Actually for inference we don't need DDP wrapper, just load weights on each rank.
    
    loader = get_dataloader(args, cfg, rank, world_size, ddp)
    
    if rank == 0:
        print(f"Tuning on {args.num_samples} samples with {world_size} GPUs (Batch Size {args.batch_size})...")
        print(f"{'Steps':<6} | {'Kappa':<6} | {'Eta':<6} | {'PSNR':<8} | {'SSIM':<8}")
        print("-" * 50)
    
    # Search Space
    step_list = [1, 5, 20, 50]
    kappa_list = [1.0]
    eta_list = [1.0]
    
    best_psnr = -1.0
    best_cfg = None
    
    # Only rank 0 shows progress bar
    combinations = list(itertools.product(step_list, kappa_list, eta_list))
    iterator = tqdm(combinations, desc="Tuning") if rank == 0 and tqdm else combinations
    
    for steps, kappa, eta in iterator:
        local_psnr_sum, local_ssim_sum, local_count = evaluate_subset(
            model, loader, ResidualShiftingDiffusion, cfg, device,
            steps, kappa, eta
        )
        
        # Aggregate results across ranks
        if ddp:
            metrics_tensor = torch.tensor([local_psnr_sum, local_ssim_sum, float(local_count)], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            global_psnr_sum = metrics_tensor[0].item()
            global_ssim_sum = metrics_tensor[1].item()
            global_count = int(metrics_tensor[2].item())
        else:
            global_psnr_sum = local_psnr_sum
            global_ssim_sum = local_ssim_sum
            global_count = local_count
            
        avg_psnr = global_psnr_sum / max(1, global_count)
        avg_ssim = global_ssim_sum / max(1, global_count)
        
        if rank == 0:
            print(f"{steps:<6} | {kappa:<6.1f} | {eta:<6.1f} | {avg_psnr:<8.4f} | {avg_ssim:<8.4f}")
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_cfg = (steps, kappa, eta)
                
    if rank == 0:
        print("\n=== Tuning Completed ===")
        print(f"Best Configuration:")
        print(f"  Steps: {best_cfg[0]}")
        print(f"  Kappa: {best_cfg[1]}")
        print(f"  Eta:   {best_cfg[2]}")
        print(f"  PSNR:  {best_psnr:.4f}")
    
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
