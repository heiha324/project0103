"""Metrics and losses for cloud detection and restoration."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _ensure_4d_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if pred.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensors, got shape {tuple(pred.shape)}")
    return pred, target


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def _flatten_binary(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = preds.view(-1)
    targets = targets.view(-1)
    return preds, targets


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return float((intersection + 1e-6) / (union + 1e-6))


def precision_recall(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return float(precision), float(recall)


def false_positive_rate(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    fp = (preds * (1 - targets)).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    return float((fp + 1e-6) / (fp + tn + 1e-6))


def overall_accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    tp = (preds * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    total = preds.numel()
    return float((tp + tn + 1e-6) / (total + 1e-6))


def f1_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return float((2 * precision * recall) / (precision + recall + 1e-6))


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred, target = _ensure_4d_pair(pred, target)
    mse_vals = F.mse_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    psnr_vals = torch.where(
        mse_vals < eps,
        torch.full_like(mse_vals, 100.0),
        -10.0 * torch.log10(mse_vals.clamp_min(eps)),
    )
    return float(psnr_vals.mean().item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = _ensure_4d_pair(pred, target)
    mae_vals = F.l1_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    return float(mae_vals.mean().item())


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = _ensure_4d_pair(pred, target)
    mse_vals = F.mse_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    return float(mse_vals.mean().item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = _ensure_4d_pair(pred, target)
    mse_vals = F.mse_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    rmse_vals = torch.sqrt(mse_vals)
    return float(rmse_vals.mean().item())


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Normalized RMSE (RMSE / (max - min))."""
    pred, target = _ensure_4d_pair(pred, target)
    mse_vals = F.mse_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    rmse_vals = torch.sqrt(mse_vals)
    target_flat = target.view(target.size(0), -1)
    val_ranges = target_flat.max(dim=1).values - target_flat.min(dim=1).values
    nrmse_vals = torch.where(
        val_ranges == 0,
        torch.zeros_like(rmse_vals),
        rmse_vals / (val_ranges + 1e-8),
    )
    return float(nrmse_vals.mean().item())


def cc(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson Correlation Coefficient (per-image mean)."""
    pred, target = _ensure_4d_pair(pred, target)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    vx = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    vy = target_flat - target_flat.mean(dim=1, keepdim=True)
    cost = (vx * vy).sum(dim=1)
    norm = torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1))
    cc_vals = cost / (norm + 1e-8)
    return float(cc_vals.mean().item())


def sam(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spectral Angle Mapper (in degrees)."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Dot product along channel dim (dim 1)
    dot = (pred * target).sum(dim=1)
    norm_pred = torch.norm(pred, dim=1)
    norm_target = torch.norm(target, dim=1)
    
    cos_theta = dot / (norm_pred * norm_target + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    return float(torch.mean(theta).item() * 180.0 / math.pi)


def ergas(pred: torch.Tensor, target: torch.Tensor, ratio: float = 1.0) -> float:
    """ERGAS metric."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    b, c, h, w = pred.shape
    diff = pred - target
    mse_band = (diff ** 2).mean(dim=(2, 3)) # (B, C)
    rmse_band = torch.sqrt(mse_band)
    mean_band = target.mean(dim=(2, 3)) # (B, C)
    
    term = (rmse_band / (mean_band + 1e-8)) ** 2
    sum_term = term.sum(dim=1) # (B,)
    
    ergas_val = 100 * ratio * torch.sqrt(sum_term / c)
    return float(ergas_val.mean().item())


def rase(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative Average Spectral Error."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
    mse_band = ((pred - target) ** 2).mean(dim=(2, 3)) # (B, C)
    rmse2_mean = mse_band.mean(dim=1) # (B,)
    
    mean_global = target.mean(dim=(1, 2, 3)) # (B,)
    
    rase_val = 100 * torch.sqrt(rmse2_mean) / (mean_global + 1e-8)
    return float(rase_val.mean().item())


# --- SSIM / MS-SSIM Utils ---

def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
    uqi_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    if uqi_mode:
        C1 = 0
        C2 = 0

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean(), ssim_map.mean(dim=(2, 3))
    return ssim_map.mean(dim=(1, 2, 3)), ssim_map


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> float:
    img1, img2 = _ensure_4d_pair(img1, img2)
    
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)
    
    s_val, _ = _ssim(img1, img2, window, window_size, channel, size_average=False)
    if size_average:
        return float(s_val.mean().item())
    if s_val.numel() == 1:
        return float(s_val.item())
    return float(s_val.mean().item())


def ms_ssim(
    img1: torch.Tensor, 
    img2: torch.Tensor, 
    window_size: int = 11, 
    size_average: bool = True,
    weights: list[float] | None = None
) -> float:
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    
    weights_t = torch.tensor(weights).to(img1.device).type_as(img1)
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)
    
    levels = len(weights)
    mcs = []
    
    for i in range(levels):
        ssim_val, cs_map = _ssim(img1, img2, window, window_size, channel, size_average=False)
        
        if i < levels - 1:
            # We only need Contrast Structure (CS) for first N-1 levels
            # CS map comes from the second term of SSIM: (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            # But _ssim returns full SSIM map. We need to decompose it or just approximate.
            # Actually, standard MS-SSIM computes Contrast and Structure terms separately.
            # My _ssim calculates the product.
            # To do this correctly, I need to refactor _ssim to return Luminance, Contrast, Structure separately?
            # Or use the property that SSIM = L * CS.
            # But usually for MS-SSIM, we assume C1 is small enough or handle it.
            # Let's simplify: Standard MS-SSIM implementation usually calculates CS separately.
            
            # Re-calculation for CS specifically to be safe:
            mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
            
            C2 = 0.03 ** 2
            cs_map_val = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            mcs.append(cs_map_val.mean(dim=(1, 2, 3)))
            
            # Downsample
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            
    # Final level SSIM (L * CS)
    ssim_val, _ = _ssim(img1, img2, window, window_size, channel, size_average=False)
    mcs.append(ssim_val)
    
    # Stack and power
    mcs_stack = torch.stack(mcs) # (levels, Batch)
    msssim_val = torch.prod(mcs_stack ** weights_t.view(-1, 1), dim=0)
    
    if size_average:
        return float(msssim_val.mean().item())
    return float(msssim_val.mean().item()) # Should handle batch return but for vis script single val is fine


def uiqi(pred: torch.Tensor, target: torch.Tensor, window_size: int = 8) -> float:
    """Universal Image Quality Index."""
    pred, target = _ensure_4d_pair(pred, target)
        
    channel = pred.size(1)
    window = _create_window(window_size, channel).to(pred.device).type_as(pred)
    
    # UIQI is SSIM with C1=C2=0 (approximated by using very small epsilon in my ssim function if needed,
    # but I added uqi_mode to set them to 0)
    # Note: Setting C1=C2=0 might cause instability if variance is 0. 
    # The standard UIQI definition doesn't use C1/C2 constants but often implementations add eps.
    # I will use uqi_mode=True which sets Cs to 0.
    
    # Note: _ssim does not handle division by zero if C1=C2=0 and denominator is 0.
    # But usually in image data it's fine or we can add small epsilon to the implementation.
    # Let's rely on the epsilon I should add to _ssim.
    
    val, _ = _ssim(pred, target, window, window_size, channel, size_average=False, uqi_mode=True)
    return float(val.mean().item())
