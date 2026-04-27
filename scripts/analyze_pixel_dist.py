#!/usr/bin/env python3
import sys
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path to import project modules
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from sarcloud.data.sen12ms_cr import Sen12MSCRRawDataset
from sarcloud.utils.image import load_tif

# Config matching your diffusion.yaml
DATA_ROOT = "/home/data/KXShen/SEN12MSCR/SEN12MSCR"
SPLIT_CSV = "/home/ps/KXShen/syncfolder/project0103/supplementary_SEN12MSCR/splits.csv"

def analyze():
    print("Initializing dataset...")
    try:
        ds = Sen12MSCRRawDataset(
            root=DATA_ROOT,
            split_csv=SPLIT_CSV,
            split="train",  # Analyze training data
            bands=[1, 2, 3, 7, 11, 12] # Same bands as config
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_samples = len(ds)
    sample_size = 1000
    indices = random.sample(range(total_samples), min(sample_size, total_samples))
    
    print(f"Dataset size: {total_samples}")
    print(f"Analyzing {len(indices)} random samples (S2 Clear/Ground Truth)...")

    # Bins for distribution (Raw DN values)
    # 0-10000 maps to 0.0-1.0
    bins = [
        0, 500, 1000, 2000, 3000, 
        4000, 5000, 6000, 7000, 8000, 9000, 10000, 
        11000, 12000, 15000, 20000
    ]
    hist_counts = np.zeros(len(bins), dtype=np.int64)
    
    global_min = float('inf')
    global_max = float('-inf')
    total_pixels = 0
    
    # Accumulate stats
    for idx in tqdm(indices):
        sample = ds.samples[idx]
        try:
            # Load raw TIF (skipping the dataset's __getitem__ normalization)
            # We want to see the RAW values
            img = load_tif(sample.s2_clear_path)
            
            # Use only the bands we care about if shape is (C, H, W)
            # But simplistic stats on all bands is also fine for checking brightness
            
            flat = img.flatten()
            if flat.size == 0:
                continue

            # Update min/max
            curr_min = flat.min()
            curr_max = flat.max()
            if curr_min < global_min: global_min = curr_min
            if curr_max > global_max: global_max = curr_max
            
            # Histogram
            # np.digitize returns 1 for 0-500, 2 for 500-1000... 
            # bins[i-1] <= x < bins[i]
            # We treat > last_bin as a separate category
            
            # Using histogram is easier
            # bins: [0, 500, ..., 20000]
            # We add a very large number to catch everything > 20000
            counts, _ = np.histogram(flat, bins=bins + [1_000_000])
            hist_counts += counts
            total_pixels += flat.size
            
        except Exception as e:
            print(f"Error reading {sample.s2_clear_path}: {e}")
            continue

    print("\n" + "="*40)
    print(f" ANALYSIS RESULT ({total_pixels} pixels)")
    print("="*40)
    print(f"Global Min Value: {global_min}")
    print(f"Global Max Value: {global_max}")
    print("-" * 40)
    
    print(f"{'Raw Range':<15} | {'Norm Range':<15} | {'Percentage':<10}")
    print("-" * 45)
    
    for i in range(len(bins)):
        count = hist_counts[i]
        ratio = (count / total_pixels) * 100
        
        lower = bins[i]
        upper = bins[i+1] if i+1 < len(bins) else "Inf"
        
        # Calculate corresponding normalized values (div 10000)
        norm_lower = lower / 10000.0
        norm_upper = upper / 10000.0 if isinstance(upper, (int, float)) else "Inf"
        
        raw_str = f"{lower}-{upper}"
        norm_str = f"{norm_lower:.2f}-{norm_upper if isinstance(norm_upper, str) else f'{norm_upper:.2f}'}"
        
        print(f"{raw_str:<15} | {norm_str:<15} | {ratio:.2f}%")

if __name__ == "__main__":
    analyze()
