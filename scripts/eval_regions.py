#!/usr/bin/env python3
"""Evaluate region metrics for sampled outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset
from sarcloud.diffusion.losses import grad_l1_loss
from sarcloud.utils.config import load_config


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.float()
    err = torch.abs(pred - target)
    denom = mask.sum() * pred.shape[0] + 1e-6
    return float((err * mask).sum() / denom)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.float()
    err = (pred - target) ** 2
    denom = mask.sum() * pred.shape[0] + 1e-6
    return float((err * mask).sum() / denom)


def psnr_from_mse(mse: float) -> float:
    if mse < 1e-12:
        return 100.0
    return float(20 * np.log10(1.0) - 10 * np.log10(mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--samples", type=str, default="outputs/diffusion/samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["sen12ms"]
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

    cloud_thresh = cfg["metrics"].get("cloud_thresh", 0.3)
    clear_thresh = cfg["metrics"].get("clear_thresh", 0.9)

    sample_dir = Path(args.samples)
    results = {
        "cloud_l1": [],
        "cloud_psnr": [],
        "clear_l1": [],
        "clear_psnr": [],
        "boundary_grad": [],
        "global_l1": [],
        "global_psnr": [],
        "global_grad": [],
    }

    indices = range(len(dataset))
    if tqdm is not None:
        indices = tqdm(indices, desc="Eval", ncols=80)
    for idx in indices:
        _, s2_cloudy, s2_clear, alpha = dataset[idx]
        pred_path = sample_dir / f"sample_{idx:05d}.npy"
        if not pred_path.exists():
            continue
        pred = torch.from_numpy(np.load(pred_path)).float()
        target = s2_clear

        if alpha is not None:
            alpha = alpha.squeeze(0)
            cloud_mask = alpha < cloud_thresh
            clear_mask = alpha > clear_thresh
            boundary_weight = alpha * (1 - alpha)

            cloud_mse = masked_mse(pred, target, cloud_mask)
            clear_mse = masked_mse(pred, target, clear_mask)

            results["cloud_l1"].append(masked_l1(pred, target, cloud_mask))
            results["cloud_psnr"].append(psnr_from_mse(cloud_mse))
            results["clear_l1"].append(masked_l1(pred, target, clear_mask))
            results["clear_psnr"].append(psnr_from_mse(clear_mse))
            results["boundary_grad"].append(
                float(
                    grad_l1_loss(
                        pred.unsqueeze(0),
                        s2_cloudy.unsqueeze(0),
                        boundary_weight.unsqueeze(0),
                    )
                )
            )

        mse = float(torch.mean((pred - target) ** 2).item())
        results["global_l1"].append(float(torch.mean(torch.abs(pred - target)).item()))
        results["global_psnr"].append(psnr_from_mse(mse))
        grad_weight_map = torch.ones_like(target[:1, ...])
        results["global_grad"].append(
            float(grad_l1_loss(pred.unsqueeze(0), target.unsqueeze(0), grad_weight_map))
        )

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "region_metrics.json"
    summary = {k: float(np.mean(v)) if v else None for k, v in results.items()}
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json_dumps(summary))

    print("Region metrics:")
    for k, v in summary.items():
        print(f"{k}: {v}")


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    main()
