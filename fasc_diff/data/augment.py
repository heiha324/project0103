from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _ensure_chw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t
    raise ValueError(f"Expected (H,W) or (C,H,W), got {tuple(t.shape)}")


def _resize_chw(t: torch.Tensor, size: int, *, mode: str) -> torch.Tensor:
    if size <= 0:
        return t
    t4 = t.unsqueeze(0)
    if mode in {"bilinear", "bicubic"}:
        out = F.interpolate(t4, size=(size, size), mode=mode, align_corners=False)
    else:
        out = F.interpolate(t4, size=(size, size), mode=mode)
    return out.squeeze(0)


def apply_emrdm_preprocess(
    tensors: dict[str, torch.Tensor],
    *,
    enabled: bool,
    output_size: int = 256,
    scale_min: float = 0.8,
    scale_max: float = 1.0,
    hflip: float = 0.5,
    vflip: float = 0.5,
    mask_keys: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    if not enabled or not tensors:
        return tensors

    mask_key_set = set(mask_keys or [])
    first = next(iter(tensors.values()))
    ref = _ensure_chw(first)
    _, H, W = ref.shape

    scale_min = float(scale_min)
    scale_max = float(scale_max)
    if scale_min <= 0 or scale_max <= 0 or scale_max < scale_min:
        raise ValueError(f"Invalid scale range: [{scale_min}, {scale_max}]")

    scale = float(torch.empty((), device=ref.device).uniform_(scale_min, scale_max).item())
    crop_h = max(1, int(round(scale * H)))
    crop_w = max(1, int(round(scale * W)))
    crop_h = min(crop_h, H)
    crop_w = min(crop_w, W)
    top = int(torch.randint(0, H - crop_h + 1, (1,), device=ref.device).item()) if H > crop_h else 0
    left = int(torch.randint(0, W - crop_w + 1, (1,), device=ref.device).item()) if W > crop_w else 0

    do_hflip = bool(torch.rand((), device=ref.device).item() < float(hflip))
    do_vflip = bool(torch.rand((), device=ref.device).item() < float(vflip))

    out: dict[str, torch.Tensor] = {}
    for key, t in tensors.items():
        t_chw = _ensure_chw(t)
        t_crop = t_chw[..., top : top + crop_h, left : left + crop_w]
        mode = "nearest" if key in mask_key_set else "bilinear"
        t_resized = _resize_chw(t_crop, output_size, mode=mode)
        if do_hflip:
            t_resized = torch.flip(t_resized, dims=[2])
        if do_vflip:
            t_resized = torch.flip(t_resized, dims=[1])
        out[key] = t_resized

    return out
