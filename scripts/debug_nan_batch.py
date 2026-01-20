#!/usr/bin/env python3
"""Debug script to locate NaN source with larger batch and more iterations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarcloud.data.sen12ms_cr import Sen12MSCRRawDataset, collate_sen12mscr
from sarcloud.diffusion.gaussian import GaussianDiffusion
from sarcloud.diffusion.losses import grad_l1_loss
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.utils.config import load_config


def main():
    cfg = load_config("configs/diffusion.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    data_cfg = cfg["sen12ms"]
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
    )
    
    batch_size = cfg["train"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_sen12mscr,
    )
    print(f"Dataset size: {len(dataset)}, batch_size: {batch_size}")

    # Build model
    model = ConditionalUNet(
        x_channels=cfg["model"]["x_channels"],
        y_channels=cfg["model"]["y_channels"],
        s_channels=cfg["model"]["s_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
        time_dim=cfg["model"].get("time_dim", 256),
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=cfg["schedule"]["timesteps"],
        schedule_type=cfg["schedule"].get("type", "cosine"),
        beta_start=cfg["schedule"].get("beta_start", 1e-4),
        beta_end=cfg["schedule"].get("beta_end", 2e-2),
        device=device,
        x0_clip_min=cfg["schedule"].get("x0_clip_min", 0.0),
        x0_clip_max=cfg["schedule"].get("x0_clip_max", 1.0),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    
    print("\n=== Training loop debug (100 steps) ===")
    model.train()
    
    nan_found = False
    for step, (s1, s2_cloudy, s2_clear, _alpha) in enumerate(loader):
        if step >= 100:
            break
            
        s1 = s1.to(device)
        y = s2_cloudy.to(device)
        x0 = s2_clear.to(device)
        
        # Check input data
        if torch.isnan(s1).any() or torch.isnan(y).any() or torch.isnan(x0).any():
            print(f"Step {step}: NaN in INPUT DATA!")
            nan_found = True
            break
        
        t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        
        optimizer.zero_grad()
        eps_pred = model(x_t, t, y, s1)
        
        if torch.isnan(eps_pred).any():
            print(f"Step {step}: NaN in eps_pred! t={t.tolist()}")
            nan_found = True
            break
        
        loss_diff = F.mse_loss(eps_pred, noise)
        
        x0_pred = diffusion.predict_x0_from_eps(x_t, t, eps_pred)
        x0_pred = x0_pred.clamp(-1.0, 2.0)
        
        loss_recon = F.l1_loss(x0_pred, x0)
        grad_weight_map = torch.ones_like(x0[:, :1, :, :])
        loss_grad = grad_l1_loss(x0_pred, x0, grad_weight_map)
        
        recon_weight = cfg["loss"].get("recon_weight", cfg["loss"].get("cloud_weight", 1.0))
        grad_weight = cfg["loss"].get("grad_weight", 0.5)
        loss = loss_diff + recon_weight * loss_recon + grad_weight * loss_grad
        
        if torch.isnan(loss):
            print(f"Step {step}: NaN LOSS!")
            print(f"  loss_diff={loss_diff.item():.4f}")
            print(f"  loss_recon={loss_recon.item():.4f}")
            print(f"  loss_grad={loss_grad.item():.4f}")
            nan_found = True
            break
        
        loss.backward()
        
        # Check gradients
        has_nan_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                print(f"Step {step}: NaN gradient in {name}")
                break
        
        if has_nan_grad:
            nan_found = True
            break
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(
                f"Step {step}: loss={loss.item():.4f} (diff={loss_diff.item():.4f}, "
                f"recon={loss_recon.item():.4f}, grad={loss_grad.item():.4f})"
            )
    
    if not nan_found:
        print("\n=== No NaN found in 100 steps! ===")
        print("The issue might be DDP-specific or occur later in training.")
    else:
        print("\n=== NaN found! See details above. ===")


if __name__ == "__main__":
    main()
