#!/usr/bin/env python3
"""Evaluate Residual Shifting diffusion with full sampling on a dataset split."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset, collate_sen12mscr
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion.sampling_rs import sample_batch_rs
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.training.ema import EMA
from sarcloud.utils.config import load_config
from sarcloud.utils.metrics import cc, ergas, mae, ms_ssim, mse, nrmse, psnr, rase, rmse, sam, ssim, uiqi


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    """Initialize DDP when launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            device = torch.device("cuda", local_rank)
        else:
            dist.init_process_group(backend="gloo", init_method="env://")
            device = torch.device("cpu")
        return True, rank, world_size, local_rank, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def build_dataset(cfg: dict) -> Sen12MSCRDataset | Sen12MSCRRawDataset:
    """Build evaluation dataset from config."""
    data_cfg = cfg.get("test") or cfg.get("val") or cfg["sen12ms"]
    dataset_type = data_cfg.get("dataset", "npy")
    if dataset_type == "sen12mscr_raw":
        return Sen12MSCRRawDataset(
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
    return Sen12MSCRDataset(
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


def build_model(cfg: dict, checkpoint_path: str | Path, device: torch.device) -> ConditionalUNet:
    """Load model weights and prefer EMA for inference."""
    model = ConditionalUNet(
        x_channels=cfg["model"]["x_channels"],
        y_channels=cfg["model"]["y_channels"],
        s_channels=cfg["model"]["s_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
        time_dim=cfg["model"].get("time_dim", 256),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema = EMA(model, decay=0.999)
    if "ema_state" in ckpt:
        ema.shadow = ckpt["ema_state"]
        for name, param in model.named_parameters():
            if name in ema.shadow:
                param.data.copy_(ema.shadow[name])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


def build_diffusion(cfg: dict, kappa_override: float | None) -> ResidualShiftingDiffusion:
    """Build diffusion schedule from config."""
    diff_cfg = cfg.get("diffusion", {})
    return ResidualShiftingDiffusion(
        timesteps=diff_cfg.get("timesteps", 1000),
        kappa=kappa_override if kappa_override is not None else diff_cfg.get("kappa", 1.0),
        schedule_type=diff_cfg.get("schedule_type", "exponential"),
        min_eta=diff_cfg.get("min_eta", 0.001),
        max_eta=diff_cfg.get("max_eta", 0.99),
        power=diff_cfg.get("power", 2.0),
        x0_clip_min=cfg.get("schedule", {}).get("x0_clip_min", 0.0),
        x0_clip_max=cfg.get("schedule", {}).get("x0_clip_max", 1.0),
    )


def make_json_safe(metrics: dict[str, float]) -> dict[str, float | None]:
    """Convert NaN/Inf to None for JSON output."""
    safe: dict[str, float | None] = {}
    for key, value in metrics.items():
        safe[key] = value if math.isfinite(value) else None
    return safe


def evaluate(
    model: ConditionalUNet,
    loader: DataLoader,
    diffusion: ResidualShiftingDiffusion,
    sampling_cfg: dict,
    device: torch.device,
    ddp: bool,
    rank: int,
) -> dict[str, float]:
    """Run full sampling and aggregate metrics."""
    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    totals = {
        "mae": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "nrmse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "ms_ssim": 0.0,
        "sam": 0.0,
        "ergas": 0.0,
        "cc": 0.0,
        "uiqi": 0.0,
        "rase": 0.0,
    }
    steps = 0

    iterator = loader
    if tqdm is not None and (not ddp or rank == 0):
        iterator = tqdm(loader, desc="Full Eval", ncols=100)

    with torch.no_grad():
        for s1, s2_cloudy, s2_clear, _alpha in iterator:
            s1 = s1.to(device)
            y = s2_cloudy.to(device)
            x0 = s2_clear.to(device)

            with torch.amp.autocast(amp_device, enabled=amp_enabled):
                x0_pred = sample_batch_rs(
                    model,
                    diffusion,
                    y,
                    s1,
                    steps=sampling_cfg.get("steps", 50),
                    schedule_cfg=sampling_cfg,
                )
            x0_pred = x0_pred.clamp(0.0, 1.0).float()
            x0 = x0.float()

            totals["mae"] += mae(x0_pred, x0)
            totals["mse"] += mse(x0_pred, x0)
            totals["rmse"] += rmse(x0_pred, x0)
            totals["nrmse"] += nrmse(x0_pred, x0)
            totals["psnr"] += psnr(x0_pred, x0)
            totals["ssim"] += ssim(x0_pred, x0)
            totals["ms_ssim"] += ms_ssim(x0_pred, x0)
            totals["sam"] += sam(x0_pred, x0)
            totals["ergas"] += ergas(x0_pred, x0)
            totals["cc"] += cc(x0_pred, x0)
            totals["uiqi"] += uiqi(x0_pred, x0)
            totals["rase"] += rase(x0_pred, x0)
            steps += 1

    if ddp:
        values = [totals[k] for k in totals] + [float(steps)]
        metrics_tensor = torch.tensor(values, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_steps = int(metrics_tensor[-1].item())
        if total_steps == 0:
            return {key: float("nan") for key in totals}
        return {
            key: metrics_tensor[idx].item() / total_steps
            for idx, key in enumerate(totals.keys())
        }

    if steps == 0:
        return {key: float("nan") for key in totals}
    return {key: value / steps for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_rs.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=0, help="仅评估前 N 个样本，0 表示全量")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--output-json", type=str, default="eval/diffusion_rs_full_metrics.json")
    args = parser.parse_args()

    ddp, rank, world_size, _local_rank, device = init_distributed()
    cfg = load_config(args.config)

    model = build_model(cfg, args.checkpoint, device)
    diffusion = build_diffusion(cfg, args.kappa)

    dataset = build_dataset(cfg)
    total_len = len(dataset)
    if args.subset_size > 0:
        total_len = min(total_len, args.subset_size)
    all_indices = list(range(total_len))
    rank_indices = all_indices[rank::world_size] if ddp else all_indices
    subset = Subset(dataset, rank_indices)

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sen12mscr,
    )

    sampling_cfg = cfg.get("sampling", {}).copy()
    if args.steps is not None:
        sampling_cfg["steps"] = args.steps
    if args.eta is not None:
        sampling_cfg["eta"] = args.eta

    if rank == 0:
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Config: {args.config}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Eval samples: {total_len}")
        print(
            f"Sampling: method={sampling_cfg.get('method', 'ddim')} "
            f"steps={sampling_cfg.get('steps', 50)} "
            f"eta={sampling_cfg.get('eta', 0.0)} "
            f"kappa={diffusion.kappa}"
        )

    metrics = evaluate(model, loader, diffusion, sampling_cfg, device, ddp, rank)

    if rank == 0:
        print("\nFull-sampling metrics:")
        for key, value in metrics.items():
            display = "nan" if not math.isfinite(value) else f"{value:.6f}"
            print(f"{key}: {display}")

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "num_samples": total_len,
            "sampling": {
                "method": sampling_cfg.get("method", "ddim"),
                "steps": sampling_cfg.get("steps", 50),
                "eta": sampling_cfg.get("eta", 0.0),
                "kappa": diffusion.kappa,
            },
            "metrics": make_json_safe(metrics),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nSaved metrics to {output_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
