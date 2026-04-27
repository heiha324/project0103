#!/usr/bin/env python3
"""Evaluate transformer-v2 diffusion with RSNet-guided adaptive sampling steps.

Usage:
    source /usr/local/anaconda3/etc/profile.d/conda.sh
    conda activate janus_pro
    python scripts/eval_diffusion_rs_transformer_v2_adaptive.py \
      --config configs/diffusion_rs_transformer_v2_13ch_wide.yaml \
      --checkpoint /path/to/diffusion_rs_transformer_best.pth \
      --rsnet-config configs/rsnet_whu_ori.yaml \
      --rsnet-checkpoint /path/to/rsnet_best.pth \
      --rsnet-temperature-json /path/to/rsnet_calibration.json \
      --dataset-section test \
      --skip-cloud-ratio 0.01 \
      --min-steps 10 \
      --max-steps 50 \
      --output-json eval/diffusion_rs_transformer_v2_adaptive_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

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
# NOTE:
# This checkpoint is produced by train_diffusion_rs_transformer_v2.py, which
# defines its own ConditionalTransformer (with additional constructor args).
# Import that exact model definition to keep checkpoint compatibility.
from train_diffusion_rs_transformer_v2 import ConditionalTransformer
from sarcloud.models.rsnet import RSNet
from sarcloud.training.ema import EMA
from sarcloud.utils.config import load_config
from sarcloud.utils.metrics import cc, ergas, mae, ms_ssim, mse, nrmse, psnr, rase, rmse, sam, ssim, uiqi


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    """Initialize DDP if launched by torchrun."""
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


def parse_bands(text: str) -> list[int] | None:
    """Parse band list from comma separated text."""
    normalized = text.strip().lower()
    if normalized in {"", "none", "auto"}:
        return None
    values = [token.strip() for token in text.split(",")]
    return [int(v) for v in values if v]


def load_temperature(path: str | None) -> float:
    """Load RSNet temperature calibration value."""
    if not path:
        return 1.0
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Temperature json not found: {p}")
    data = load_config(p)
    return float(data.get("temperature", 1.0))


def build_dataset_from_section(cfg: dict, section: str) -> Sen12MSCRDataset | Sen12MSCRRawDataset:
    """Build dataset from config section."""
    if section not in cfg:
        raise KeyError(f"Config section '{section}' not found")
    data_cfg = cfg[section]
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


def build_transformer_model(cfg: dict, checkpoint_path: str | Path, device: torch.device) -> ConditionalTransformer:
    """Build transformer-v2 model and load checkpoint (prefer EMA)."""
    model_cfg = cfg["model"]
    model = ConditionalTransformer(
        x_channels=model_cfg["x_channels"],
        y_channels=model_cfg["y_channels"],
        s_channels=model_cfg["s_channels"],
        embed_dim=model_cfg.get("embed_dim", 512),
        depth=model_cfg.get("depth", 8),
        num_heads=model_cfg.get("num_heads", 16),
        patch_size=model_cfg.get("patch_size", 16),
        time_dim=model_cfg.get("time_dim", 256),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        refine_channels=model_cfg.get("refine_channels", 256),
        detail_min_channels=model_cfg.get("detail_min_channels", 128),
        cond_se_reduction=model_cfg.get("cond_se_reduction", 16),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema = EMA(model, decay=0.999)
    if "ema_state" in ckpt:
        ema.shadow = ckpt["ema_state"]
        ema.apply_to(model)
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def build_rsnet_model(rsnet_cfg: dict, checkpoint_path: str | Path, device: torch.device) -> tuple[RSNet, int]:
    """Build cloud detector and load weights."""
    model_cfg = rsnet_cfg["model"]
    in_channels = int(model_cfg["in_channels"])
    model = RSNet(
        in_channels=in_channels,
        base_channels=model_cfg.get("base_channels", 32),
        depth=model_cfg.get("depth", 4),
        use_batchnorm=model_cfg.get("use_batchnorm", True),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, in_channels


def build_diffusion(cfg: dict, kappa_override: float | None) -> ResidualShiftingDiffusion:
    """Build diffusion schedule from config."""
    diff_cfg = cfg.get("diffusion", {})
    schedule_cfg = cfg.get("schedule", {})
    return ResidualShiftingDiffusion(
        timesteps=diff_cfg.get("timesteps", 1000),
        kappa=kappa_override if kappa_override is not None else diff_cfg.get("kappa", 1.0),
        schedule_type=diff_cfg.get("schedule_type", "linear"),
        min_eta=diff_cfg.get("min_eta", 0.001),
        max_eta=diff_cfg.get("max_eta", 0.99),
        power=diff_cfg.get("power", 2.0),
        x0_clip_min=schedule_cfg.get("x0_clip_min", 0.0),
        x0_clip_max=schedule_cfg.get("x0_clip_max", 1.0),
    )


def select_rsnet_input(
    y: torch.Tensor,
    expected_channels: int,
    bands: Sequence[int] | None,
) -> torch.Tensor:
    """Select cloudy-image channels for RSNet inference."""
    if y.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for y, got {tuple(y.shape)}")
    channels = int(y.size(1))

    if bands is not None:
        if len(bands) != expected_channels:
            raise ValueError(
                f"rsnet bands length {len(bands)} != model in_channels {expected_channels}"
            )
        if max(bands) >= channels or min(bands) < 0:
            raise ValueError(
                f"rsnet bands {list(bands)} out of range for cloudy channels {channels}"
            )
        index = torch.tensor(list(bands), device=y.device, dtype=torch.long)
        return torch.index_select(y, dim=1, index=index)

    if channels == expected_channels:
        return y

    if expected_channels == 4 and channels >= 8:
        # SEN12MS 常用 RGB+NIR 通道索引
        default_bands = torch.tensor([1, 2, 3, 7], device=y.device, dtype=torch.long)
        return torch.index_select(y, dim=1, index=default_bands)

    raise ValueError(
        "Cannot infer RSNet input bands automatically. "
        f"Cloudy channels={channels}, rsnet in_channels={expected_channels}. "
        "Please set --rsnet-bands."
    )


def estimate_cloud_ratio(
    y: torch.Tensor,
    rsnet: RSNet,
    temperature: float,
    expected_channels: int,
    bands: Sequence[int] | None,
    cloud_threshold: float,
) -> torch.Tensor:
    """Estimate cloud coverage ratio for each image in batch."""
    rs_input = select_rsnet_input(y, expected_channels=expected_channels, bands=bands)
    logits = rsnet(rs_input)
    if temperature > 0:
        logits = logits / temperature
    probs = torch.sigmoid(logits)
    mask = (probs >= cloud_threshold).float()
    return mask.mean(dim=(1, 2, 3))


def map_ratio_to_steps(
    cloud_ratio: torch.Tensor,
    skip_ratio: float,
    min_steps: int,
    max_steps: int,
) -> torch.Tensor:
    """Map cloud ratio to adaptive sampling steps."""
    steps = torch.zeros_like(cloud_ratio, dtype=torch.long)
    active = cloud_ratio >= skip_ratio
    if active.any():
        ratio_active = cloud_ratio[active]
        denom = max(1.0 - skip_ratio, 1e-6)
        alpha = ((ratio_active - skip_ratio) / denom).clamp(0.0, 1.0)
        mapped = min_steps + alpha * float(max_steps - min_steps)
        steps[active] = torch.round(mapped).to(torch.long)
    return steps


def safe_json_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Convert NaN/Inf to None for JSON output."""
    out: dict[str, float | None] = {}
    for key, value in metrics.items():
        out[key] = value if math.isfinite(value) else None
    return out


def evaluate_adaptive(
    model: ConditionalTransformer,
    rsnet: RSNet,
    loader: DataLoader,
    diffusion: ResidualShiftingDiffusion,
    sampling_cfg: dict,
    device: torch.device,
    ddp: bool,
    rank: int,
    rsnet_temperature: float,
    rsnet_in_channels: int,
    rsnet_bands: Sequence[int] | None,
    cloud_threshold: float,
    skip_ratio: float,
    min_steps: int,
    max_steps: int,
) -> tuple[dict[str, float], dict[str, int | float | dict[str, int]]]:
    """Run adaptive inference and aggregate image quality metrics."""
    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    metric_keys = (
        "mae",
        "mse",
        "rmse",
        "nrmse",
        "psnr",
        "ssim",
        "ms_ssim",
        "sam",
        "ergas",
        "cc",
        "uiqi",
        "rase",
    )
    totals = {k: 0.0 for k in metric_keys}
    sample_count = 0
    no_diff_count = 0
    cloud_ratio_sum = 0.0
    total_step_sum = 0.0
    diff_step_sum = 0.0
    step_hist = torch.zeros(max_steps + 1, dtype=torch.long, device=device)

    iterator = loader
    if tqdm is not None and (not ddp or rank == 0):
        iterator = tqdm(loader, desc="Adaptive Eval", ncols=100)

    model.eval()
    rsnet.eval()
    with torch.no_grad():
        for s1, s2_cloudy, s2_clear, _alpha in iterator:
            s1 = s1.to(device)
            y = s2_cloudy.to(device)
            x0 = s2_clear.to(device)

            cloud_ratio = estimate_cloud_ratio(
                y=y,
                rsnet=rsnet,
                temperature=rsnet_temperature,
                expected_channels=rsnet_in_channels,
                bands=rsnet_bands,
                cloud_threshold=cloud_threshold,
            )
            steps_per_sample = map_ratio_to_steps(
                cloud_ratio=cloud_ratio,
                skip_ratio=skip_ratio,
                min_steps=min_steps,
                max_steps=max_steps,
            )

            x0_pred = y.clone()
            unique_steps = torch.unique(steps_per_sample)
            for step_count in unique_steps.tolist():
                if step_count <= 0:
                    continue
                index = (steps_per_sample == step_count).nonzero(as_tuple=False).squeeze(1)
                if index.numel() == 0:
                    continue
                y_sub = y.index_select(0, index)
                s1_sub = s1.index_select(0, index)
                with torch.amp.autocast(amp_device, enabled=amp_enabled):
                    pred_sub = sample_batch_rs(
                        model,
                        diffusion,
                        y_sub,
                        s1_sub,
                        steps=int(step_count),
                        schedule_cfg=sampling_cfg,
                    )
                x0_pred.index_copy_(0, index, pred_sub)

            x0_pred = x0_pred.clamp(0.0, 1.0).float()
            x0 = x0.float()
            batch_size = int(x0.size(0))

            totals["mae"] += mae(x0_pred, x0) * batch_size
            totals["mse"] += mse(x0_pred, x0) * batch_size
            totals["rmse"] += rmse(x0_pred, x0) * batch_size
            totals["nrmse"] += nrmse(x0_pred, x0) * batch_size
            totals["psnr"] += psnr(x0_pred, x0) * batch_size
            totals["ssim"] += ssim(x0_pred, x0) * batch_size
            totals["ms_ssim"] += ms_ssim(x0_pred, x0) * batch_size
            totals["sam"] += sam(x0_pred, x0) * batch_size
            totals["ergas"] += ergas(x0_pred, x0) * batch_size
            totals["cc"] += cc(x0_pred, x0) * batch_size
            totals["uiqi"] += uiqi(x0_pred, x0) * batch_size
            totals["rase"] += rase(x0_pred, x0) * batch_size

            step_values = steps_per_sample.to(torch.long)
            hist_add = torch.bincount(step_values, minlength=max_steps + 1)
            step_hist += hist_add.to(device=device, dtype=torch.long)

            sample_count += batch_size
            no_diff_count += int((step_values == 0).sum().item())
            cloud_ratio_sum += float(cloud_ratio.sum().item())
            total_step_sum += float(step_values.float().sum().item())
            diff_step_sum += float(step_values[step_values > 0].float().sum().item())

    aggregate_vector = torch.tensor(
        [totals[k] for k in metric_keys]
        + [
            float(sample_count),
            float(no_diff_count),
            cloud_ratio_sum,
            total_step_sum,
            diff_step_sum,
        ],
        device=device,
        dtype=torch.float64,
    )

    if ddp:
        dist.all_reduce(aggregate_vector, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_hist, op=dist.ReduceOp.SUM)

    total_samples = int(round(aggregate_vector[len(metric_keys)].item()))
    if total_samples <= 0:
        empty_metrics = {k: float("nan") for k in metric_keys}
        empty_stats: dict[str, int | float | dict[str, int]] = {
            "num_samples": 0,
            "skip_count": 0,
            "skip_ratio": 0.0,
            "mean_cloud_ratio": float("nan"),
            "mean_steps_all": float("nan"),
            "mean_steps_diffused_only": float("nan"),
            "step_histogram": {},
        }
        return empty_metrics, empty_stats

    metrics = {
        key: aggregate_vector[idx].item() / total_samples
        for idx, key in enumerate(metric_keys)
    }

    global_no_diff = int(round(aggregate_vector[len(metric_keys) + 1].item()))
    global_cloud_ratio_sum = aggregate_vector[len(metric_keys) + 2].item()
    global_total_step_sum = aggregate_vector[len(metric_keys) + 3].item()
    global_diff_step_sum = aggregate_vector[len(metric_keys) + 4].item()
    global_diff_count = max(total_samples - global_no_diff, 0)

    step_hist_cpu = step_hist.detach().cpu().tolist()
    step_histogram = {
        str(step): int(count)
        for step, count in enumerate(step_hist_cpu)
        if count > 0
    }

    stats = {
        "num_samples": total_samples,
        "skip_count": global_no_diff,
        "skip_ratio": float(global_no_diff / total_samples),
        "mean_cloud_ratio": float(global_cloud_ratio_sum / total_samples),
        "mean_steps_all": float(global_total_step_sum / total_samples),
        "mean_steps_diffused_only": (
            float(global_diff_step_sum / global_diff_count) if global_diff_count > 0 else 0.0
        ),
        "step_histogram": step_histogram,
    }
    return metrics, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_rs_transformer_v2_13ch_wide.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="transformer v2 diffusion checkpoint")
    parser.add_argument("--rsnet-config", type=str, default="configs/rsnet_whu_ori.yaml")
    parser.add_argument("--rsnet-checkpoint", type=str, required=True)
    parser.add_argument("--rsnet-temperature-json", type=str, default="")
    parser.add_argument("--rsnet-bands", type=str, default="1,2,3,7")
    parser.add_argument("--rsnet-cloud-threshold", type=float, default=0.5)
    parser.add_argument("--dataset-section", type=str, default="test")
    parser.add_argument("--subset-size", type=int, default=0, help="0 表示全量测试集")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-cloud-ratio", type=float, default=0.01, help="< 此阈值直接输出有云图像")
    parser.add_argument("--min-steps", type=int, default=10, help="云量>=0.01 的最小采样步数")
    parser.add_argument("--max-steps", type=int, default=50, help="云量接近1.0 时的最大采样步数")
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--output-json", type=str, default="eval/diffusion_rs_transformer_v2_adaptive_metrics.json")
    args = parser.parse_args()

    if args.min_steps <= 0 or args.max_steps <= 0:
        raise ValueError("min_steps and max_steps must be positive")
    if args.max_steps < args.min_steps:
        raise ValueError("max_steps must be >= min_steps")
    if args.skip_cloud_ratio < 0.0 or args.skip_cloud_ratio >= 1.0:
        raise ValueError("skip_cloud_ratio must be in [0, 1)")

    rsnet_bands = parse_bands(args.rsnet_bands)
    rsnet_temperature = load_temperature(args.rsnet_temperature_json)

    ddp, rank, world_size, _local_rank, device = init_distributed()
    cfg = load_config(args.config)
    rsnet_cfg = load_config(args.rsnet_config)

    diffusion_model = build_transformer_model(cfg, args.checkpoint, device)
    rsnet_model, rsnet_in_channels = build_rsnet_model(rsnet_cfg, args.rsnet_checkpoint, device)
    diffusion = build_diffusion(cfg, args.kappa)

    dataset = build_dataset_from_section(cfg, args.dataset_section)
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
    if args.eta is not None:
        sampling_cfg["eta"] = args.eta

    if rank == 0:
        print(f"Diffusion checkpoint: {args.checkpoint}")
        print(f"RSNet checkpoint: {args.rsnet_checkpoint}")
        print(f"Config: {args.config}")
        print(f"Dataset section: {args.dataset_section}")
        print(f"Eval samples: {total_len}")
        print(f"World size: {world_size}")
        print(
            "Adaptive policy: "
            f"cloud<{args.skip_cloud_ratio:.4f} -> skip, "
            f"cloud>= {args.skip_cloud_ratio:.4f} -> linear {args.min_steps}-{args.max_steps} steps"
        )
        print(
            "Sampling cfg: "
            f"method={sampling_cfg.get('method', 'ddim')} "
            f"eta={sampling_cfg.get('eta', 0.0)} "
            f"kappa={diffusion.kappa}"
        )
        print(
            "RSNet cfg: "
            f"threshold={args.rsnet_cloud_threshold} "
            f"temperature={rsnet_temperature:.4f} "
            f"bands={rsnet_bands if rsnet_bands is not None else 'auto'}"
        )

    metrics, stats = evaluate_adaptive(
        model=diffusion_model,
        rsnet=rsnet_model,
        loader=loader,
        diffusion=diffusion,
        sampling_cfg=sampling_cfg,
        device=device,
        ddp=ddp,
        rank=rank,
        rsnet_temperature=rsnet_temperature,
        rsnet_in_channels=rsnet_in_channels,
        rsnet_bands=rsnet_bands,
        cloud_threshold=args.rsnet_cloud_threshold,
        skip_ratio=args.skip_cloud_ratio,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )

    if rank == 0:
        print("\nAdaptive-eval metrics:")
        for key, value in metrics.items():
            display = "nan" if not math.isfinite(value) else f"{value:.6f}"
            print(f"{key}: {display}")

        print("\nAdaptive policy stats:")
        print(f"num_samples: {stats['num_samples']}")
        print(f"skip_count: {stats['skip_count']}")
        print(f"skip_ratio: {stats['skip_ratio']:.6f}")
        print(f"mean_cloud_ratio: {stats['mean_cloud_ratio']:.6f}")
        print(f"mean_steps_all: {stats['mean_steps_all']:.6f}")
        print(f"mean_steps_diffused_only: {stats['mean_steps_diffused_only']:.6f}")

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "diffusion_checkpoint": str(args.checkpoint),
            "rsnet_checkpoint": str(args.rsnet_checkpoint),
            "config": str(args.config),
            "rsnet_config": str(args.rsnet_config),
            "dataset_section": args.dataset_section,
            "sampling": {
                "method": sampling_cfg.get("method", "ddim"),
                "eta": sampling_cfg.get("eta", 0.0),
                "kappa": diffusion.kappa,
            },
            "adaptive_policy": {
                "skip_cloud_ratio": args.skip_cloud_ratio,
                "min_steps": args.min_steps,
                "max_steps": args.max_steps,
                "rsnet_cloud_threshold": args.rsnet_cloud_threshold,
                "rsnet_temperature": rsnet_temperature,
                "rsnet_bands": rsnet_bands,
            },
            "metrics": safe_json_metrics(metrics),
            "stats": stats,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nSaved metrics to {output_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
