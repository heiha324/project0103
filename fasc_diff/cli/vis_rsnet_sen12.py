from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from fasc_diff.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset
from fasc_diff.models.rsnet import RSNet
from fasc_diff.utils.checkpoint import load_checkpoint
from fasc_diff.utils.seed import set_seed


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


def _to_rgb(x_chw: torch.Tensor, rgb_indices: tuple[int, int, int]) -> torch.Tensor:
    if x_chw.ndim != 3:
        raise ValueError(f"Expected CHW, got {tuple(x_chw.shape)}")
    if x_chw.shape[0] == 3:
        return x_chw
    return x_chw[list(rgb_indices), ...]


def _colorize_mask(mask_hw: torch.Tensor) -> torch.Tensor:
    """
    mask_hw: (H,W) int64 with classes:
      0 clear, 1 thick, 2 thin, 3 shadow
    returns: (3,H,W) float in [0,1]
    """
    if mask_hw.ndim != 2:
        raise ValueError(f"Expected HW mask, got {tuple(mask_hw.shape)}")
    palette = torch.tensor(
        [
            [0, 0, 0],        # 0 clear
            [255, 0, 0],      # 1 thick
            [0, 255, 0],      # 2 thin
            [0, 0, 255],      # 3 shadow
        ],
        dtype=torch.float32,
        device=mask_hw.device,
    ) / 255.0
    m = mask_hw.clamp(0, 3).to(torch.long)
    return palette[m].permute(2, 0, 1).contiguous()


def _parse_int_list(v) -> list[int] | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    raise TypeError(f"Expected list[int] or None, got {type(v)}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Trained RSNet checkpoint (.pt)")
    ap.add_argument("--dataset", type=str, default="sen12mscr_raw", help="sen12mscr | sen12mscr_raw")
    ap.add_argument("--sen12-root", type=str, required=True, help="SEN12MS-CR root")
    ap.add_argument("--split", type=str, default="train", help="train/val/test")
    ap.add_argument("--split-csv", type=str, default=None, help="Split CSV (recommended for sen12mscr_raw)")
    ap.add_argument("--roi-glob", type=str, default=None, help="Optional ROI glob for sen12mscr_raw")
    ap.add_argument("--alpha-root", type=str, default=None, help="Optional alpha root for sen12mscr_raw")
    ap.add_argument("--alpha-subdir", type=str, default=None, help="Optional alpha subdir for sen12mscr")
    ap.add_argument("--image-ext", type=str, default=None, help="Override image extension (default: .tif for raw, .npy for preprocessed)")
    ap.add_argument("--bands", type=int, nargs="*", default=None, help="Optional band indices (0-based) for S2")

    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--rgb-indices", type=int, nargs=3, default=[3, 2, 1], help="RGB indices for S2 bands (0-based)")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder (default: output/rsnet_sen12_YYYYmmdd_HHMMSS)")
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    device = _pick_device(args.device)

    outdir = Path(args.outdir) if args.outdir else Path("output") / f"rsnet_sen12_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    model = RSNet(in_channels=13, num_classes=4).to(device).eval()
    load_checkpoint(args.ckpt, model=model, optimizer=None, map_location=device)

    dataset_type = str(args.dataset).lower()
    bands = _parse_int_list(args.bands)
    if dataset_type == "sen12mscr":
        image_ext = args.image_ext or ".npy"
        ds = Sen12MSCRDataset(
            root=args.sen12_root,
            split=args.split,
            image_ext=image_ext,
            bands=bands,
            alpha_subdir=args.alpha_subdir,
        )
    elif dataset_type == "sen12mscr_raw":
        image_ext = args.image_ext or ".tif"
        ds = Sen12MSCRRawDataset(
            root=args.sen12_root,
            split=args.split,
            split_csv=args.split_csv,
            roi_glob=args.roi_glob,
            bands=bands,
            alpha_root=args.alpha_root,
            alpha_ext=".npy",
        )
    else:
        raise ValueError("dataset must be sen12mscr or sen12mscr_raw")

    n = len(ds)
    k = min(int(args.num_samples), n)
    indices = random.sample(range(n), k=k)
    rgb_indices = tuple(int(x) for x in args.rgb_indices)

    tiles: list[torch.Tensor] = []
    summary_path = outdir / "samples.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"ckpt={args.ckpt}\n")
        f.write(f"dataset={dataset_type} root={args.sen12_root} split={args.split}\n")
        f.write(f"num_samples={k} seed={args.seed}\n")
        f.write("indices:\n")
        for idx in indices:
            f.write(f"  - {idx}\n")

    for j, idx in enumerate(indices):
        sample = ds[idx]
        x = sample["opt_cloudy"].to(device)  # (13,H,W) in [-1,1]
        rel = sample.get("relpath", sample.get("id", str(idx)))

        logits = model(x.unsqueeze(0))
        pred = torch.argmax(logits, dim=1)[0]  # (H,W)

        rgb = _to_01(_to_rgb(x, rgb_indices))
        mask_rgb = _colorize_mask(pred)

        pair = make_grid(torch.stack([rgb, mask_rgb], dim=0), nrow=2)
        out_path = outdir / f"{j:02d}_{Path(str(rel)).stem}.png"
        save_image(pair, out_path)

        tiles.append(rgb)
        tiles.append(mask_rgb)

    grid = make_grid(torch.stack(tiles, dim=0), nrow=2)
    save_image(grid, outdir / "grid.png")

    print(f"[OK] Saved {k} visualizations to: {outdir}")
    print(f"[OK] Saved grid to: {outdir / 'grid.png'}")
    print(f"[OK] Saved index list to: {summary_path}")


if __name__ == "__main__":
    main()

