#!/usr/bin/env python3
"""Check for problematic alpha values in dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sarcloud.data.sen12ms_cr import Sen12MSCRRawDataset
from sarcloud.utils.config import load_config


def main():
    cfg = load_config("configs/diffusion.yaml")
    
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
    
    print(f"Checking {len(dataset)} samples for problematic alpha values...")
    
    problematic = []
    all_zeros = 0
    all_ones = 0
    has_nan = 0
    
    for i in range(len(dataset)):
        try:
            s1, s2_cloudy, s2_clear, alpha = dataset[i]
        except Exception as e:
            print(f"Sample {i}: ERROR loading - {e}")
            problematic.append((i, "load_error", str(e)))
            continue
        
        if alpha is None:
            print(f"Sample {i}: alpha is None!")
            problematic.append((i, "none", ""))
            continue
        
        if torch.isnan(alpha).any():
            print(f"Sample {i}: NaN in alpha!")
            problematic.append((i, "nan", ""))
            has_nan += 1
            continue
        
        if torch.isnan(s1).any() or torch.isnan(s2_cloudy).any() or torch.isnan(s2_clear).any():
            print(f"Sample {i}: NaN in image data!")
            problematic.append((i, "nan_image", ""))
            continue
        
        alpha_min = alpha.min().item()
        alpha_max = alpha.max().item()
        alpha_mean = alpha.mean().item()
        
        # Check for all zeros or all ones
        if alpha_max < 0.01:
            all_zeros += 1
            if len(problematic) < 20:
                print(f"Sample {i}: alpha nearly all zeros (max={alpha_max:.4f})")
            problematic.append((i, "all_zeros", f"max={alpha_max:.4f}"))
        
        if alpha_min > 0.99:
            all_ones += 1
            if len(problematic) < 20:
                print(f"Sample {i}: alpha nearly all ones (min={alpha_min:.4f})")
            problematic.append((i, "all_ones", f"min={alpha_min:.4f}"))
        
        # Check for extreme boundary (alpha*(1-alpha) = 0 everywhere)
        boundary = alpha * (1 - alpha)
        if boundary.sum().item() < 1e-6:
            if len(problematic) < 20:
                print(f"Sample {i}: boundary weight sum < 1e-6")
            problematic.append((i, "no_boundary", f"sum={boundary.sum().item():.2e}"))
        
        if i % 10000 == 0:
            print(f"Checked {i}/{len(dataset)}...")
    
    print(f"\n=== Summary ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Problematic samples: {len(problematic)}")
    print(f"  - All zeros: {all_zeros}")
    print(f"  - All ones: {all_ones}")
    print(f"  - Has NaN: {has_nan}")
    
    if problematic:
        print(f"\nFirst 10 problematic samples:")
        for idx, issue, detail in problematic[:10]:
            print(f"  Sample {idx}: {issue} {detail}")


if __name__ == "__main__":
    main()
