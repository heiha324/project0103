#!/usr/bin/env python3
"""
Scan the entire dataset to check for NaNs, Infs, or loading errors.
This script iterates through the DataLoader exactly as the training loop does.
If corrupted files are found, they will be reported (via the warnings in __getitem__).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sarcloud.data.sen12ms_cr import Sen12MSCRRawDataset, Sen12MSCRDataset, collate_sen12mscr
from sarcloud.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Check dataset for validity")
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml", help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for scanning")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config)
    
    data_cfg = cfg["sen12ms"]
    dataset_type = data_cfg.get("dataset", "npy")
    print(f"Building dataset type: {dataset_type}")
    
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

    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True,
        collate_fn=collate_sen12mscr,
    )
    
    print("Starting scan... (Warnings about bad files will appear below)")
    start_time = time.time()
    
    total_samples = 0
    batches_with_nan = 0
    
    # We rely on the modified __getitem__ to print "WARN: NaN/Inf..."
    # This loop just drives the loading.
    for i, batch in enumerate(tqdm(loader, unit="batch")):
        s1, s2_cloudy, s2_clear, alpha = batch
        
        # Double check after collation (though __getitem__ should have fixed it)
        batch_has_nan = False
        if torch.isnan(s1).any() or torch.isinf(s1).any(): batch_has_nan = True
        if torch.isnan(s2_cloudy).any() or torch.isinf(s2_cloudy).any(): batch_has_nan = True
        if torch.isnan(s2_clear).any() or torch.isinf(s2_clear).any(): batch_has_nan = True
        if alpha is not None and (torch.isnan(alpha).any() or torch.isinf(alpha).any()): batch_has_nan = True
        
        if batch_has_nan:
            batches_with_nan += 1
            # We can't easily identify WHICH file in the batch caused it here, 
            # but the __getitem__ print should have handled that.
            
        total_samples += s1.shape[0]

    elapsed = time.time() - start_time
    print(f"\nScan complete in {elapsed:.1f}s.")
    print(f"Processed {total_samples} samples.")
    if batches_with_nan > 0:
        print(f"WARNING: Found {batches_with_nan} batches containing residual NaN/Inf values (post-sanitization).")
    else:
        print("No residual NaN/Inf values found in batches (inputs are clean or successfully sanitized).")
    print("Check output above for 'WARN' logs to see if any specific files were fixed on-the-fly.")

if __name__ == "__main__":
    main()
