from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fasc_diff.data.masks import normalized_entropy, prob_to_m_soft, prob_to_p_thick
from fasc_diff.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset
from fasc_diff.models.rsnet import RSNet
from fasc_diff.utils.checkpoint import load_checkpoint
from fasc_diff.utils.logger import log_message, setup_logger
from fasc_diff.utils.seed import set_seed


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _init_distributed(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0, _pick_device(device_arg)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() and device_arg != "cpu" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return True, rank, world_size, local_rank, device


def _is_main_process(distributed: bool, rank: int) -> bool:
    return (not distributed) or rank == 0


def _resolve_outdir(*, outdir: str | None, outroot: str, prefix: str, distributed: bool, rank: int) -> Path:
    if outdir is not None and str(outdir).strip() != "":
        return Path(outdir)

    run_name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    if distributed:
        obj_list = [run_name] if rank == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        run_name = str(obj_list[0])

    return Path(outroot) / run_name


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
    ap.add_argument("--split-csv", type=str, default=None, help="Required for sen12mscr_raw if you want CSV split")
    ap.add_argument("--roi-glob", type=str, default=None, help="Optional ROI glob for sen12mscr_raw")
    ap.add_argument("--alpha-root", type=str, default=None, help="Optional alpha root for sen12mscr_raw")
    ap.add_argument("--alpha-subdir", type=str, default=None, help="Optional alpha subdir for sen12mscr (preprocessed)")
    ap.add_argument("--image-ext", type=str, default=None, help="Override image extension (default: .tif for raw, .npy for preprocessed)")
    ap.add_argument("--bands", type=int, nargs="*", default=None, help="Optional band indices (0-based) for S2")

    ap.add_argument("--outroot", type=str, default="/home/data/KXShen/model/1_25", help="Base output folder")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder (if set, disables auto time subfolder)")
    ap.add_argument("--ext", type=str, default=".npz", help="Output file extension (default .npz)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    distributed, rank, world_size, local_rank, device = _init_distributed(args.device)
    is_main = _is_main_process(distributed, rank)
    set_seed(args.seed + rank)

    outdir = _resolve_outdir(outdir=args.outdir, outroot=args.outroot, prefix=f"teacher_rsnet_{args.split}", distributed=distributed, rank=rank)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("cache_rsnet_teacher", outdir / "cache.log") if is_main else None
    log_message(f"[TeacherCache] start_time={time.strftime('%Y-%m-%d %H:%M:%S')}", logger, console=is_main, use_tqdm=False)
    log_message(f"outdir={outdir}", logger, console=is_main, use_tqdm=False)
    log_message(f"ckpt={args.ckpt}", logger, console=is_main, use_tqdm=False)
    log_message(f"split={args.split} dataset={args.dataset}", logger, console=is_main, use_tqdm=False)
    if distributed:
        log_message(f"ddp=1 world_size={world_size} rank={rank} local_rank={local_rank}", logger, console=is_main, use_tqdm=False)

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

    sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if distributed else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = RSNet(in_channels=13, num_classes=4).to(device).eval()
    load_checkpoint(args.ckpt, model=model, optimizer=None, map_location=device)

    iterator = dl
    if is_main:
        iterator = tqdm(dl, desc=f"cache {args.split}", ncols=70)

    saved = 0
    skipped = 0
    for batch in iterator:
        x = batch["opt_cloudy"].to(device)  # (B,13,H,W) in [-1,1]
        relpaths = batch.get("relpath")
        if not isinstance(relpaths, list):
            # Fallback: build relpath from ids (best-effort)
            relpaths = [str(i) for i in batch.get("id", list(range(x.shape[0])))]

        logits = model(x)
        prob = torch.softmax(logits, dim=1).clamp(1e-8, 1.0)
        prob_np = prob.detach().cpu().numpy()

        for i in range(prob_np.shape[0]):
            rel = str(relpaths[i])
            out_path = (outdir / rel).with_suffix(args.ext)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and (not args.overwrite):
                skipped += 1
                continue

            p = prob_np[i].astype(np.float16, copy=False)  # (4,H,W)
            m_soft = prob_to_m_soft(p).astype(np.float16, copy=False)
            u = normalized_entropy(p).astype(np.float16, copy=False)
            p_thick = prob_to_p_thick(p).astype(np.float16, copy=False)
            np.savez_compressed(out_path, prob=p, m_soft=m_soft, uncertainty=u, p_thick=p_thick)
            saved += 1

    total_saved = saved
    total_skipped = skipped
    if distributed:
        saved_t = torch.tensor([saved], device=device)
        skipped_t = torch.tensor([skipped], device=device)
        dist.all_reduce(saved_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(skipped_t, op=dist.ReduceOp.SUM)
        total_saved = int(saved_t.item())
        total_skipped = int(skipped_t.item())
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        log_message(f"Done. saved={total_saved} skipped={total_skipped} outdir={outdir}", logger, console=True, use_tqdm=False)


if __name__ == "__main__":
    main()
