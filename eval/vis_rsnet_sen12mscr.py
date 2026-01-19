#!/usr/bin/env python3
"""Visualize RS-Net predictions on SEN12MS-CR cloudy TIFFs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch

try:
    import tifffile as tiff
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("tifffile is required to read .tif inputs") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional for headless envs
    raise RuntimeError("matplotlib is required for visualization") from exc

from sarcloud.models.rsnet import RSNet
from sarcloud.utils.config import load_config
from sarcloud.utils.image import normalize_s2, select_bands


def parse_indices(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    # If channel dimension is first and small, assume CHW.
    if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
        return arr
    # If last dimension is small, assume HWC.
    if arr.shape[-1] <= 16:
        return np.transpose(arr, (2, 0, 1))
    # Fallback: pick the layout with smaller channel dimension.
    return arr if arr.shape[0] <= arr.shape[-1] else np.transpose(arr, (2, 0, 1))


def load_tif(path: Path) -> np.ndarray:
    return tiff.imread(str(path))


def list_tifs(root: Path) -> list[Path]:
    paths = list(root.rglob("*.tif"))
    paths += list(root.rglob("*.tiff"))
    return sorted(paths)


def resolve_checkpoint(cfg: dict, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)

    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "outputs/rsnet"))
    direct = out_dir / "rsnet_best.pth"
    if direct.exists():
        return direct

    parent = out_dir.parent
    prefix = f"{out_dir.name}_"
    if parent.exists():
        runs = sorted([p for p in parent.glob(f"{prefix}*") if p.is_dir()], reverse=True)
        for run in runs:
            candidate = run / "rsnet_best.pth"
            if candidate.exists():
                return candidate

    raise FileNotFoundError("Could not find rsnet_best.pth; pass --checkpoint explicitly")


def prepare_rgb(chw: np.ndarray, rgb_bands: list[int], clip_min: float, clip_max: float) -> np.ndarray:
    rgb = select_bands(chw, rgb_bands)
    rgb = normalize_s2(rgb, clip_min, clip_max)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.transpose(rgb, (1, 2, 0))
    return np.round(rgb * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rsnet_whu_ori.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--input-root",
        type=str,
        default="/home/data/KXShen/SEN12MSCR/ROIs1868_summer_s2_cloudy",
    )
    parser.add_argument("--bands", type=str, default="3,2,1,7")
    parser.add_argument("--rgb-bands", type=str, default="3,2,1")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=10000.0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--min-cloud", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cloud-thresh", type=float, default=None)
    parser.add_argument("--max-scan", type=int, default=500)
    parser.add_argument("--show-prob", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="eval/rsnet_sen12mscr_vis.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = resolve_checkpoint(cfg, args.checkpoint)

    bands = parse_indices(args.bands)
    rgb_bands = parse_indices(args.rgb_bands)
    if len(bands) != cfg["model"]["in_channels"]:
        raise ValueError(
            f"--bands has {len(bands)} entries, but model expects {cfg['model']['in_channels']} channels"
        )
    if len(rgb_bands) != 3:
        raise ValueError("--rgb-bands must have exactly 3 indices")

    model = RSNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"].get("base_channels", 32),
        depth=cfg["model"].get("depth", 4),
        use_batchnorm=cfg["model"].get("use_batchnorm", True),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    root = Path(args.input_root)
    paths = list_tifs(root)
    if not paths:
        raise RuntimeError(f"No .tif files found under {root}")
    rng = random.SystemRandom() if args.seed is None else random.Random(args.seed)
    if args.max_scan > 0 and len(paths) > args.max_scan:
        paths = rng.sample(paths, args.max_scan)
    else:
        rng.shuffle(paths)

    cloud_thresh = args.threshold if args.cloud_thresh is None else args.cloud_thresh
    candidates = []

    with torch.inference_mode():
        scanned = 0
        for path in paths:
            scanned += 1
            arr = ensure_chw(load_tif(path))
            if max(bands + rgb_bands) >= arr.shape[0]:
                continue
            inp = select_bands(arr, bands)
            inp = normalize_s2(inp, args.clip_min, args.clip_max)
            image_t = torch.from_numpy(inp).float().unsqueeze(0).to(device)
            prob = torch.sigmoid(model(image_t))[0, 0].cpu().numpy()
            cloud_ratio = float((prob >= cloud_thresh).mean())
            if cloud_ratio < args.min_cloud:
                continue

            rgb_uint8 = prepare_rgb(arr, rgb_bands, args.clip_min, args.clip_max)
            candidates.append((path, rgb_uint8, prob, cloud_ratio))

    if len(candidates) < args.num_samples:
        raise RuntimeError(
            f"Only {len(candidates)} samples met min-cloud >= {args.min_cloud:.2f} "
            f"after scanning {scanned} files. Increase --max-scan or lower --min-cloud."
        )

    selected = rng.sample(candidates, args.num_samples) if len(candidates) > args.num_samples else candidates

    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n), squeeze=False)

    for row, (path, rgb_uint8, prob, cloud_ratio) in enumerate(selected):
        if args.show_prob:
            pred_vis = prob
        else:
            pred_vis = (prob >= args.threshold).astype(np.float32)

        axes[row, 0].imshow(rgb_uint8)
        axes[row, 1].imshow(pred_vis, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row, 0].set_title(f"RGB ({path.name})", fontsize=8)
        axes[row, 1].set_title(f"Pred {cloud_ratio * 100:.1f}%", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
