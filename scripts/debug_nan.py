#!/usr/bin/env python3
"""Debug script to locate NaN source in diffusion training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
import torch.nn.functional as F

from sarcloud.data.sen12ms_cr import Sen12MSCRRawDataset
from sarcloud.diffusion.gaussian import GaussianDiffusion
from sarcloud.diffusion.losses import grad_l1_loss
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.utils.config import load_config


def check_tensor(name: str, t: torch.Tensor) -> bool:
    """Check tensor for NaN/Inf and print stats."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    t_min = t.min().item()
    t_max = t.max().item()
    t_mean = t.mean().item()
    status = "OK" if not has_nan and not has_inf else "BAD"
    print(f"  {name:20s}: min={t_min:10.4f} max={t_max:10.4f} mean={t_mean:10.4f} nan={has_nan} inf={has_inf} [{status}]")
    return has_nan or has_inf


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
    print(f"Dataset size: {len(dataset)}")

    # Check first few samples for data issues
    print("\n=== Checking data samples ===")
    for i in range(min(5, len(dataset))):
        s1, s2_cloudy, s2_clear, alpha = dataset[i]
        print(f"\nSample {i}:")
        check_tensor("s1", s1)
        check_tensor("s2_cloudy", s2_cloudy)
        check_tensor("s2_clear", s2_clear)
        if alpha is not None:
            check_tensor("alpha", alpha)
            print(f"  alpha coverage: {(alpha > 0.5).float().mean().item():.2%} clear")
        else:
            print("  alpha: None (MISSING!)")

    # Build model
    print("\n=== Building model ===")
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
    )

    # Check diffusion schedule
    print("\n=== Diffusion schedule stats ===")
    print(f"  sqrt_alphas_cumprod[0]:   {diffusion.sqrt_alphas_cumprod[0].item():.6f}")
    print(f"  sqrt_alphas_cumprod[500]: {diffusion.sqrt_alphas_cumprod[500].item():.6f}")
    print(f"  sqrt_alphas_cumprod[999]: {diffusion.sqrt_alphas_cumprod[999].item():.6f}")
    print(f"  Min sqrt_alphas_cumprod:  {diffusion.sqrt_alphas_cumprod.min().item():.6f}")

    # Test forward pass with different timesteps
    print("\n=== Testing forward pass ===")
    s1, s2_cloudy, s2_clear, _alpha = dataset[0]
    s1 = s1.unsqueeze(0).to(device)
    y = s2_cloudy.unsqueeze(0).to(device)
    x0 = s2_clear.unsqueeze(0).to(device)

    test_timesteps = [0, 100, 500, 800, 999]
    
    for t_val in test_timesteps:
        print(f"\n--- Timestep t={t_val} ---")
        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        
        check_tensor("x_t", x_t)
        
        with torch.no_grad():
            eps_pred = model(x_t, t, y, s1)
        check_tensor("eps_pred", eps_pred)
        
        # Predict x0 (this is where explosion happens)
        x0_pred = diffusion.predict_x0_from_eps(x_t, t, eps_pred)
        bad = check_tensor("x0_pred (raw)", x0_pred)
        
        if bad or abs(x0_pred.max().item()) > 10:
            print(f"  WARNING: x0_pred exploded! Max abs value: {x0_pred.abs().max().item():.2f}")
            sqrt_cumprod = diffusion.sqrt_alphas_cumprod[t_val].item()
            print(f"  sqrt_alphas_cumprod[{t_val}] = {sqrt_cumprod:.6f}, 1/sqrt = {1/sqrt_cumprod:.2f}")
        
        # Clamp and check losses
        x0_pred_clamped = x0_pred.clamp(-1.0, 2.0)
        check_tensor("x0_pred (clamped)", x0_pred_clamped)
        
        # Check individual losses
        loss_recon = F.l1_loss(x0_pred_clamped, x0)
        grad_weight_map = torch.ones_like(x0[:, :1, :, :])
        loss_grad = grad_l1_loss(x0_pred_clamped, x0, grad_weight_map)

        check_tensor("loss_recon", loss_recon)
        check_tensor("loss_grad", loss_grad)

    # Test with batch and gradient
    print("\n=== Testing with gradient computation ===")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for step in range(3):
        t = torch.randint(0, 1000, (1,), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        
        optimizer.zero_grad()
        eps_pred = model(x_t, t, y, s1)
        loss_diff = F.mse_loss(eps_pred, noise)
        
        x0_pred = diffusion.predict_x0_from_eps(x_t, t, eps_pred)
        x0_pred = x0_pred.clamp(-1.0, 2.0)
        
        loss_recon = F.l1_loss(x0_pred, x0)
        grad_weight_map = torch.ones_like(x0[:, :1, :, :])
        loss_grad = grad_l1_loss(x0_pred, x0, grad_weight_map)
        recon_weight = cfg["loss"].get("recon_weight", cfg["loss"].get("cloud_weight", 1.0))
        grad_weight = cfg["loss"].get("grad_weight", 0.5)
        loss = loss_diff + recon_weight * loss_recon + grad_weight * loss_grad
        
        print(f"\nStep {step}, t={t.item()}")
        print(
            f"  loss_diff={loss_diff.item():.4f} loss_recon={loss_recon.item():.4f} "
            f"loss_grad={loss_grad.item():.4f}"
        )
        print(f"  total_loss={loss.item():.4f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("  !!! LOSS IS NAN/INF !!!")
            break
        
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0.0
        max_grad = 0.0
        has_nan_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                max_grad = max(max_grad, p.grad.abs().max().item())
                if torch.isnan(p.grad).any():
                    has_nan_grad = True
                    print(f"  NaN gradient in {name}")
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  grad_norm={total_grad_norm:.4f} max_grad={max_grad:.4f} nan_grad={has_nan_grad}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("\n=== Debug complete ===")


if __name__ == "__main__":
    main()
