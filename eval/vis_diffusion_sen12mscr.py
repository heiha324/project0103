#!/usr/bin/env python3
"""Visualize before/after diffusion outputs on SEN12MS-CR."""

from __future__ import annotations

import argparse
import os
import random
import re
import site
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
plt = None  # optional; imported lazily in main

torch = None  # type: ignore[assignment]
GaussianDiffusion = None  # type: ignore[assignment]
ConditionalUNet = None  # type: ignore[assignment]
Sen12MSCRDataset = None  # type: ignore[assignment]
Sen12MSCRRawDataset = None  # type: ignore[assignment]
load_config = None  # type: ignore[assignment]


def _ensure_torch_importable() -> None:
    if os.environ.get("SAR_CLOUD_LD_FIXED") == "1":
        return
    try:
        import torch as _torch  # noqa: F401
        return
    except Exception:
        pass

    user_site = Path(site.getusersitepackages())
    nvjitlink = user_site / "nvidia/nvjitlink/lib"
    cusparse = user_site / "nvidia/cusparse/lib"
    if not nvjitlink.exists() or not cusparse.exists():
        return

    required = [str(nvjitlink), str(cusparse)]
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [p for p in ld_path.split(":") if p and p not in required]
    os.environ["LD_LIBRARY_PATH"] = ":".join(required + paths)
    os.environ["SAR_CLOUD_LD_FIXED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


def _import_deps() -> None:
    global torch
    global GaussianDiffusion
    global ConditionalUNet
    global Sen12MSCRDataset
    global Sen12MSCRRawDataset
    global load_config

    import torch as _torch

    from sarcloud.data.sen12ms_cr import Sen12MSCRDataset as _Sen12MSCRDataset
    from sarcloud.data.sen12ms_cr import Sen12MSCRRawDataset as _Sen12MSCRRawDataset
    from sarcloud.diffusion.gaussian import GaussianDiffusion as _GaussianDiffusion
    from sarcloud.models.cond_unet import ConditionalUNet as _ConditionalUNet
    from sarcloud.utils.config import load_config as _load_config

    torch = _torch
    GaussianDiffusion = _GaussianDiffusion
    ConditionalUNet = _ConditionalUNet
    Sen12MSCRDataset = _Sen12MSCRDataset
    Sen12MSCRRawDataset = _Sen12MSCRRawDataset
    load_config = _load_config


def parse_indices(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def list_samples(sample_dir: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in sorted(sample_dir.glob("sample_*.npy")):
        match = re.search(r"sample_(\d+)", path.stem)
        if match:
            mapping[int(match.group(1))] = path
    return mapping


def _resolve_checkpoint_from_run(run_dir: Path, ckpt_name: str) -> Path:
    if run_dir.is_file():
        return run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    candidates = {
        "ema": run_dir / "diffusion_ema.pth",
        "best": run_dir / "diffusion_best.pth",
        "last": run_dir / "diffusion_last.pth",
    }
    if ckpt_name not in candidates:
        raise ValueError(f"Unknown ckpt name: {ckpt_name}")
    checkpoint_path = candidates[ckpt_name]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_checkpoint(cfg: dict, checkpoint_arg: str | None, run_dir: str | None, ckpt_name: str) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)
    if run_dir:
        return _resolve_checkpoint_from_run(Path(run_dir), ckpt_name)
    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "outputs/diffusion"))
    for name in ("diffusion_ema.pth", "diffusion_best.pth", "diffusion_last.pth"):
        candidate = out_dir / name
        if candidate.exists():
            return candidate
    parent = out_dir.parent
    prefix = f"{out_dir.name}_"
    if parent.exists():
        runs = sorted([p for p in parent.glob(f"{prefix}*") if p.is_dir()], reverse=True)
        for run in runs:
            for name in ("diffusion_ema.pth", "diffusion_best.pth", "diffusion_last.pth"):
                candidate = run / name
                if candidate.exists():
                    return candidate
    raise FileNotFoundError("Could not find diffusion checkpoint; pass --checkpoint explicitly")


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "ema_state" in ckpt:
        model.load_state_dict(ckpt["ema_state"], strict=False)
    else:
        model.load_state_dict(ckpt)


def sample_batch(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    y: torch.Tensor,
    s: torch.Tensor,
    steps: int,
    schedule_cfg: dict,
) -> torch.Tensor:
    model.eval()
    device = y.device
    t_seq = diffusion.sample_timesteps(steps)
    x = torch.randn_like(y)
    method = schedule_cfg.get("method", "ddim")
    eta = float(schedule_cfg.get("eta", 0.0))
    use_ddim = method != "ddpm" or steps < diffusion.timesteps

    for step, t in enumerate(t_seq):
        t_batch = torch.full((y.size(0),), t, device=device, dtype=torch.long)
        t_prev = t_seq[step + 1] if step + 1 < len(t_seq) else None
        t_prev_batch = (
            torch.full((y.size(0),), t_prev, device=device, dtype=torch.long) if t_prev is not None else None
        )
        with torch.no_grad():
            eps = model(x, t_batch, y, s)
            if use_ddim:
                x = diffusion.ddim_step(x, t_batch, t_prev_batch, eps, eta=eta)
            else:
                x = diffusion.p_sample(x, t_batch, eps)

    return x


def to_rgb(chw: np.ndarray, rgb_indices: list[int]) -> np.ndarray:
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {chw.shape}")
    if max(rgb_indices) >= chw.shape[0]:
        raise ValueError(f"RGB indices {rgb_indices} out of range for shape {chw.shape}")
    rgb = chw[rgb_indices, ...]
    rgb = np.transpose(rgb, (1, 2, 0))
    return np.clip(rgb, 0.0, 1.0)


def _save_grid_pil(rows: list[list[np.ndarray]], output_path: Path, pad: int = 2) -> None:
    from PIL import Image

    if not rows:
        raise ValueError("No rows to save")
    cols = len(rows[0])
    if any(len(row) != cols for row in rows):
        raise ValueError("All rows must have the same number of columns")

    def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.ndim != 3 or img.shape[2] not in (1, 3):
            raise ValueError(f"Expected HWC image, got shape {img.shape}")
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = np.clip(img, 0.0, 1.0)
        return np.round(img * 255.0).astype(np.uint8)

    row_imgs: list[np.ndarray] = []
    for row in rows:
        imgs = [to_uint8_rgb(img) for img in row]
        h = imgs[0].shape[0]
        if any(im.shape[0] != h for im in imgs):
            raise ValueError("Row images must have same height")
        pad_v = np.zeros((h, pad, 3), dtype=np.uint8)
        merged = imgs[0]
        for im in imgs[1:]:
            merged = np.concatenate([merged, pad_v, im], axis=1)
        row_imgs.append(merged)

    w = row_imgs[0].shape[1]
    if any(im.shape[1] != w for im in row_imgs):
        raise ValueError("All rows must have same width after merge")
    pad_h = np.zeros((pad, w, 3), dtype=np.uint8)
    grid = row_imgs[0]
    for im in row_imgs[1:]:
        grid = np.concatenate([grid, pad_h, im], axis=0)

    Image.fromarray(grid).save(output_path)


def build_dataset(data_cfg: dict):
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


def select_indices(
    args: argparse.Namespace,
    dataset,
    pool: list[int],
    sample_map: dict[int, Path] | None,
) -> list[int]:
    if args.indices:
        indices = parse_indices(args.indices)
        if sample_map is not None:
            missing = [idx for idx in indices if idx not in sample_map]
            if missing:
                raise RuntimeError(f"Missing samples for indices: {missing}")
        return indices

    if not pool:
        raise RuntimeError("No candidate samples found")
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    selected: list[int] = []
    scanned = 0
    for idx in pool:
        scanned += 1
        if args.max_scan > 0 and scanned > args.max_scan:
            break
        if args.min_cloud > 0:
            _, _, _, alpha = dataset[idx]
            if alpha is None:
                raise RuntimeError("Alpha cache missing; cannot filter by cloud ratio")
            alpha_np = alpha.squeeze(0).numpy()
            cloud_ratio = float((alpha_np < args.cloud_thresh).mean())
            if cloud_ratio < args.min_cloud:
                continue
        selected.append(idx)
        if len(selected) >= args.num_samples:
            break

    if len(selected) < args.num_samples:
        raise RuntimeError(
            f"Only {len(selected)} samples found after scanning {scanned}. "
            "Lower --min-cloud or increase --max-scan."
        )
    return selected


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml")
    parser.add_argument("--samples", type=str, default="outputs/diffusion/samples")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="ema", choices=("ema", "best", "last"))
    parser.add_argument("--data", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cloud", type=float, default=0.0)
    parser.add_argument("--cloud-thresh", type=float, default=0.3)
    parser.add_argument("--max-scan", type=int, default=0)
    parser.add_argument("--rgb", type=str, default="0,1,2")
    parser.add_argument("--show-alpha", action="store_true")
    parser.add_argument("--use-matplotlib", action="store_true")
    parser.add_argument("--steps", type=str, default=None, help="Comma-separated steps, e.g. '50,100,500'")
    parser.add_argument("--output", type=str, default="eval/diffusion_vis.png")
    args = parser.parse_args()

    _ensure_torch_importable()
    _import_deps()
    
    # Import metrics
    from sarcloud.utils.metrics import (
        mae, mse, rmse, nrmse, psnr, ssim, ms_ssim, sam, ergas, cc, uiqi, rase
    )

    cfg = load_config(args.config)
    data_cfg = cfg["sen12ms"]
    if args.data == "test":
        data_cfg = cfg.get("test") or cfg.get("val") or data_cfg
    dataset = build_dataset(data_cfg)

    # Parse steps
    sampling_cfg = cfg.get("sampling", {})
    default_steps = sampling_cfg.get("steps", 50)
    if args.steps:
        steps_list = parse_int_list(args.steps)
    else:
        steps_list = [default_steps]

    sample_dir = Path(args.samples)
    sample_map = list_samples(sample_dir) if sample_dir.exists() else {}
    
    # Check if we can use existing samples
    # We can only use existing samples if steps is not specified (or matches default) AND checkpoint is not specified
    use_samples = bool(sample_map) and args.checkpoint is None and args.run_dir is None and args.steps is None

    model = None
    diffusion = None
    device = None
    
    if not use_samples:
        try:
            checkpoint_path = resolve_checkpoint(cfg, args.checkpoint, args.run_dir, args.ckpt)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"No samples found under {sample_dir} and no checkpoint available. "
                "Provide --checkpoint/--run-dir or run scripts/sample_diffusion.py first."
            ) from exc
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = ConditionalUNet(
            x_channels=cfg["model"]["x_channels"],
            y_channels=cfg["model"]["y_channels"],
            s_channels=cfg["model"]["s_channels"],
            base_channels=cfg["model"].get("base_channels", 64),
            depth=cfg["model"].get("depth", 4),
            time_dim=cfg["model"].get("time_dim", 256),
        ).to(device)
        load_checkpoint(model, checkpoint_path)
        model.eval()
        diffusion = GaussianDiffusion(
            timesteps=cfg["schedule"]["timesteps"],
            schedule_type=cfg["schedule"].get("type", "cosine"),
            beta_start=cfg["schedule"].get("beta_start", 1e-4),
            beta_end=cfg["schedule"].get("beta_end", 2e-2),
            device=device,
            x0_clip_min=cfg["schedule"].get("x0_clip_min", 0.0),
            x0_clip_max=cfg["schedule"].get("x0_clip_max", 1.0),
        )

    rgb_indices = parse_indices(args.rgb)
    if len(rgb_indices) != 3:
        raise ValueError("--rgb must have exactly three indices")

    pool = list(sample_map.keys()) if use_samples else list(range(len(dataset)))
    indices = select_indices(args, dataset, pool, sample_map if use_samples else None)

    rows: list[list[np.ndarray]] = []
    row_labels: list[str] = []

    # Metrics storage: {step: {metric: [val, val, ...]}}
    metrics_stats = {s: {} for s in steps_list}
    metric_names = [
        "MAE", "MSE", "RMSE", "nRMSE", "PSNR", "SSIM", "MS-SSIM", 
        "SAM", "ERGAS", "CC", "UIQI", "RASE"
    ]
    for s in steps_list:
        for m in metric_names:
            metrics_stats[s][m] = []

    for row, idx in enumerate(indices):
        s1, s2_cloudy, s2_clear, alpha = dataset[idx]
        
        # Prepare targets for metrics (ensure tensor on device if needed, or CPU)
        # Metrics are computed on CPU to save GPU memory and avoid synchronization issues in loop
        target_t = s2_clear.float() # (C, H, W)
        
        preds_rgb = []
        
        if use_samples:
            # Single prediction from file
            # Assuming the file corresponds to the single step in steps_list[0]
            # Since use_samples is True only if steps is None (default 50), steps_list has 1 element.
            pred_np = np.load(sample_map[idx]).astype(np.float32)
            preds_rgb.append(to_rgb(pred_np, rgb_indices))
            
            # Compute metrics
            pred_t = torch.from_numpy(pred_np)
            current_step = steps_list[0]
            
            # Compute all metrics
            metrics_stats[current_step]["MAE"].append(mae(pred_t, target_t))
            metrics_stats[current_step]["MSE"].append(mse(pred_t, target_t))
            metrics_stats[current_step]["RMSE"].append(rmse(pred_t, target_t))
            metrics_stats[current_step]["nRMSE"].append(nrmse(pred_t, target_t))
            metrics_stats[current_step]["PSNR"].append(psnr(pred_t, target_t))
            metrics_stats[current_step]["SSIM"].append(ssim(pred_t, target_t))
            metrics_stats[current_step]["MS-SSIM"].append(ms_ssim(pred_t, target_t))
            metrics_stats[current_step]["SAM"].append(sam(pred_t, target_t))
            metrics_stats[current_step]["ERGAS"].append(ergas(pred_t, target_t))
            metrics_stats[current_step]["CC"].append(cc(pred_t, target_t))
            metrics_stats[current_step]["UIQI"].append(uiqi(pred_t, target_t))
            metrics_stats[current_step]["RASE"].append(rase(pred_t, target_t))

        else:
            # Online inference for each step count
            s1_t = s1.unsqueeze(0).to(device)
            y_t = s2_cloudy.unsqueeze(0).to(device)
            
            for step_cnt in steps_list:
                with torch.inference_mode():
                    pred_t_gpu = sample_batch(
                        model,
                        diffusion,
                        y_t,
                        s1_t,
                        steps=step_cnt,
                        schedule_cfg=sampling_cfg,
                    )
                # Move to CPU for metrics and visualization
                pred_t = pred_t_gpu.squeeze(0).cpu() # (C, H, W)
                preds_rgb.append(to_rgb(pred_t.numpy(), rgb_indices))
                
                # Compute all metrics
                metrics_stats[step_cnt]["MAE"].append(mae(pred_t, target_t))
                metrics_stats[step_cnt]["MSE"].append(mse(pred_t, target_t))
                metrics_stats[step_cnt]["RMSE"].append(rmse(pred_t, target_t))
                metrics_stats[step_cnt]["nRMSE"].append(nrmse(pred_t, target_t))
                metrics_stats[step_cnt]["PSNR"].append(psnr(pred_t, target_t))
                metrics_stats[step_cnt]["SSIM"].append(ssim(pred_t, target_t))
                metrics_stats[step_cnt]["MS-SSIM"].append(ms_ssim(pred_t, target_t))
                metrics_stats[step_cnt]["SAM"].append(sam(pred_t, target_t))
                metrics_stats[step_cnt]["ERGAS"].append(ergas(pred_t, target_t))
                metrics_stats[step_cnt]["CC"].append(cc(pred_t, target_t))
                metrics_stats[step_cnt]["UIQI"].append(uiqi(pred_t, target_t))
                metrics_stats[step_cnt]["RASE"].append(rase(pred_t, target_t))

        cloudy_rgb = to_rgb(s2_cloudy.numpy(), rgb_indices)
        clear_rgb = to_rgb(s2_clear.numpy(), rgb_indices)

        cloud_ratio = None
        if alpha is not None:
            alpha_np = alpha.squeeze(0).numpy()
            cloud_ratio = float((alpha_np < args.cloud_thresh).mean())

        label = f"idx {idx}"
        if cloud_ratio is not None:
            label = f"{label} cloud {cloud_ratio * 100:.1f}%"
        row_labels.append(label)

        # Order: [Cloudy, Pred_1, Pred_2..., Clear, (Alpha)]
        row_imgs: list[np.ndarray] = [cloudy_rgb] + preds_rgb + [clear_rgb]
        
        if args.show_alpha:
            alpha_vis = alpha.squeeze(0).numpy() if alpha is not None else np.zeros_like(s2_clear[0].numpy())
            row_imgs.append(alpha_vis)
        rows.append(row_imgs)

    # Print Metrics Summary
    print("\n" + "="*60)
    print(f"METRICS SUMMARY (Average over {len(indices)} samples)")
    print("="*60)
    
    for s in steps_list:
        print(f"Step {s}:")
        stats = metrics_stats[s]
        for m_name in metric_names:
            vals = stats[m_name]
            if not vals:
                print(f"  {m_name:<8}: N/A")
                continue
            avg = np.mean(vals)
            std = np.std(vals)
            print(f"  {m_name:<8}: {avg:.4f} ± {std:.4f}")
        print("-" * 40)
    print("="*60 + "\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    global plt
    if args.use_matplotlib and plt is None:
        try:  # pragma: no cover - optional dependency
            import matplotlib.pyplot as _plt

            plt = _plt
        except Exception:
            plt = None

    if args.use_matplotlib and plt is not None:
        cols = len(rows[0])
        n = len(rows)
        fig, axes = plt.subplots(n, cols, figsize=(cols * 3, n * 3), squeeze=False)
        
        # Determine column titles
        titles = ["Cloudy"]
        if use_samples:
            titles.append("Pred")
        else:
            for s in steps_list:
                titles.append(f"Step {s}")
        titles.append("Clear")
        if args.show_alpha:
            titles.append("Alpha")
            
        for row_idx, (label, row_imgs) in enumerate(zip(row_labels, rows)):
            for col_idx, img in enumerate(row_imgs):
                if img.ndim == 2:
                    axes[row_idx, col_idx].imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                else:
                    axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].axis("off")
                
                # Set titles only on first row
                if row_idx == 0 and col_idx < len(titles):
                    axes[row_idx, col_idx].set_title(titles[col_idx], fontsize=10)
            
            # Add row label to the left of the first image
            axes[row_idx, 0].text(-0.1, 0.5, label, transform=axes[row_idx, 0].transAxes, 
                                  va='center', ha='right', fontsize=8, rotation=90)
                                  
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        _save_grid_pil(rows, output_path)

    print(f"Saved visualization to {output_path}")
    if row_labels:
        print("Rows:", "; ".join(row_labels))


if __name__ == "__main__":
    main()
