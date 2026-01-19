"""Alpha construction utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(arr)


def gaussian_kernel_2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = int(3 * sigma + 0.5)
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return tensor
    if tensor.dim() == 2:
        tensor = tensor[None, None, ...]
    elif tensor.dim() == 3:
        tensor = tensor[None, ...]
    if tensor.dim() != 4:
        raise ValueError("Expected tensor with shape (B,C,H,W)")
    kernel = gaussian_kernel_2d(sigma, tensor.device, tensor.dtype)
    kernel = kernel[None, None, ...]
    padding = kernel.shape[-1] // 2
    channels = tensor.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(tensor, kernel, padding=padding, groups=channels)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    if mask.dim() == 2:
        mask = mask[None, None, ...]
    elif mask.dim() == 3:
        mask = mask[:, None, ...]
    if mask.dim() != 4:
        raise ValueError("Expected mask with shape (B,1,H,W) or (H,W)")
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)


def build_alpha(
    p_cloud: np.ndarray | torch.Tensor,
    rgbnir: Optional[np.ndarray | torch.Tensor] = None,
    tau: float = 0.5,
    k: float = 12.0,
    blur_sigma: float = 1.5,
    ring_threshold: float = 0.6,
    ring_radius: int = 8,
    ring_weight: float = 0.5,
    shadow_weight: float = 0.7,
    dark_percentile: float = 10.0,
) -> torch.Tensor:
    """Build alpha from cloud probability (torch output in [0,1])."""
    p_cloud_t = _to_tensor(p_cloud).float()
    if p_cloud_t.dim() == 2:
        p_cloud_t = p_cloud_t[None, None, ...]
    elif p_cloud_t.dim() == 3:
        p_cloud_t = p_cloud_t[:, None, ...]
    if p_cloud_t.dim() != 4:
        raise ValueError("p_cloud must be (H,W), (B,H,W), or (B,1,H,W)")

    p_clear = 1.0 - p_cloud_t
    alpha0 = torch.sigmoid(k * (p_clear - tau))
    alpha = gaussian_blur(alpha0, sigma=blur_sigma).clamp(0.0, 1.0)

    cloud_core = (p_cloud_t > ring_threshold).float()
    cloud_ring = dilate(cloud_core, radius=ring_radius)
    alpha = alpha * (1.0 - ring_weight * cloud_ring)

    if rgbnir is not None:
        rgbnir_t = _to_tensor(rgbnir).float()
        if rgbnir_t.dim() == 3:
            rgbnir_t = rgbnir_t[None, ...]
        if rgbnir_t.dim() != 4:
            raise ValueError("rgbnir must be (C,H,W) or (B,C,H,W)")
        mean_map = rgbnir_t.mean(dim=1, keepdim=True)
        # Compute percentile per-sample
        b, _, h, w = mean_map.shape
        mean_flat = mean_map.view(b, -1)
        thresh = torch.quantile(mean_flat, dark_percentile / 100.0, dim=1)
        thresh = thresh.view(b, 1, 1, 1)
        dark = mean_map < thresh
        shadow_suspect = dark & (cloud_ring > 0)
        alpha = alpha * (1.0 - shadow_weight * shadow_suspect.float())

    return alpha.clamp(0.0, 1.0)


def hard_mask(alpha: torch.Tensor, threshold: float = 0.97) -> torch.Tensor:
    return (alpha > threshold).float()
