from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml
from torchvision.utils import make_grid, save_image

from fasc_diff.data.npz_seg_dataset import NPZSegmentationDataset
from fasc_diff.data.cloudsen12plus_mlstac import CloudSEN12PlusMLSTACDataset, CloudSEN12PlusMLSTACConfig
from fasc_diff.models.rsnet import RSNet
from fasc_diff.utils.checkpoint import load_checkpoint, save_checkpoint
from fasc_diff.utils.jsonl import append_jsonl
from fasc_diff.utils.logger import log_message, setup_logger
from fasc_diff.utils.seed import set_seed


def _init_distributed(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if device_arg == "auto":
            return False, 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, torch.device(device_arg)

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


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _resolve_outdir(*, outdir: str | None, outroot: str, prefix: str, distributed: bool, rank: int) -> Path:
    if outdir is not None and str(outdir).strip() != "":
        return Path(outdir)

    run_name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    if distributed:
        obj_list = [run_name] if rank == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        run_name = str(obj_list[0])

    return Path(outroot) / run_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="Train data: folder of .npz or a .mlstac file")
    ap.add_argument("--val-root", type=str, default=None, help="Val data: folder of .npz or a .mlstac file")
    ap.add_argument("--outroot", type=str, default="/home/data/KXShen/model/1_25", help="Base output folder (default: /home/data/KXShen/model/1_25)")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder (if set, disables auto time subfolder)")
    ap.add_argument("--in-channels", type=int, default=13, help="Input channels (Sentinel-2 bands)")
    ap.add_argument("--num-classes", type=int, default=4, help="Number of classes")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    distributed, rank, world_size, local_rank, device = _init_distributed(args.device)
    is_main = _is_main_process(distributed, rank)
    set_seed(args.seed + rank)

    outdir = _resolve_outdir(outdir=args.outdir, outroot=args.outroot, prefix="rsnet", distributed=distributed, rank=rank)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train_rsnet", outdir / "train.log") if is_main else None
    log_message(f"[RSNet] start_time={time.strftime('%Y-%m-%d %H:%M:%S')}", logger, console=is_main, use_tqdm=False)
    log_message(f"outdir={outdir}", logger, console=is_main, use_tqdm=False)
    if distributed:
        log_message(f"ddp=1 world_size={world_size} rank={rank} local_rank={local_rank}", logger, console=is_main, use_tqdm=False)
    if is_main:
        with (outdir / "config_merged.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False, allow_unicode=True)

    def build_dataset(path: str, *, train: bool) -> torch.utils.data.Dataset:
        p = Path(path)
        if p.is_file() and p.suffix.lower() == ".mlstac":
            # CloudSEN12+ p509/p2000 MLSTAC file.
            cfg = CloudSEN12PlusMLSTACConfig(
                band_indices=tuple(range(1, args.in_channels + 1)),
                label_band=args.in_channels + 1,  # CM1
                clip_min=0.0,
                clip_max=10000.0,
                to_minus1_1=True,
            )
            return CloudSEN12PlusMLSTACDataset(p, cfg=cfg, random_flip=train)
        # Fallback: our simple npz segmentation format.
        return NPZSegmentationDataset(p, random_flip=train)

    ds = build_dataset(args.data_root, train=True)
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if is_main and args.val_root is not None:
        val_ds = build_dataset(args.val_root, train=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = RSNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    start_epoch = 0
    best_val_loss = math.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, model=model, optimizer=opt, map_location=device)
        start_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_loss = float((ckpt.get("extra") or {}).get("best_val_loss", best_val_loss))
        log_message(f"Resumed from {args.resume} (epoch={start_epoch}, step={start_step})", logger, console=is_main, use_tqdm=False)

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    global_step = start_step
    model.train()
    metrics_path = outdir / "metrics.jsonl"

    def evaluate(*, epoch: int) -> dict[str, float]:
        if val_loader is None:
            return {}
        model.eval()
        total = 0.0
        steps = 0
        with torch.no_grad():
            iterator = tqdm(val_loader, desc=f"val_rsnet {epoch+1}/{args.epochs}", ncols=70, leave=False)
            for batch in iterator:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                total += float(loss.item())
                steps += 1
        model.train()
        return {"loss": total / max(1, steps)}

    def save_vis(epoch: int) -> None:
        if val_loader is None:
            return
        batch = next(iter(val_loader), None)
        if batch is None:
            return
        x = batch["image"][:4].to(device)
        y = batch["label"][:4].to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        model.train()

        # Use B04,B03,B02 (red,green,blue) = channels 3,2,1 in 0-based.
        rgb = x[:, [3, 2, 1], ...].clamp(-1, 1)
        rgb = (rgb + 1.0) * 0.5

        def colorize(mask: torch.Tensor) -> torch.Tensor:
            palette = torch.tensor(
                [
                    [0, 0, 0],        # 0 clear
                    [255, 0, 0],      # 1 thick
                    [0, 255, 0],      # 2 thin
                    [0, 0, 255],      # 3 shadow
                ],
                dtype=torch.float32,
                device=mask.device,
            ) / 255.0
            m = mask.clamp(0, 3).to(torch.long)
            return palette[m].permute(0, 3, 1, 2).contiguous()

        y_rgb = colorize(y)
        p_rgb = colorize(pred)

        tiles = []
        for i in range(rgb.shape[0]):
            tiles.extend([rgb[i], p_rgb[i], y_rgb[i]])
        grid = make_grid(torch.stack(tiles, dim=0), nrow=3)
        vis_dir = outdir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, vis_dir / f"epoch_{epoch:04d}.png")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(total=len(dl), desc=f"train_rsnet {epoch+1}/{args.epochs}", ncols=70) if is_main else None
        epoch_loss = 0.0
        steps = 0
        for batch in dl:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item())
            steps += 1
            global_step += 1
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        train_loss = epoch_loss / max(1, steps)
        val_metrics = evaluate(epoch=epoch) if is_main else {}
        record = {"epoch": epoch, "step": global_step, "train_loss": train_loss, "val": val_metrics}
        if is_main:
            append_jsonl(metrics_path, record)

        msg = f"Epoch {epoch+1}/{args.epochs} - train_loss {train_loss:.4f} (steps={steps})"
        if val_metrics:
            msg += f" val_loss {float(val_metrics.get('loss', float('nan'))):.4f}"
        log_message(msg, logger, console=is_main, use_tqdm=True)

        extra = {"best_val_loss": best_val_loss}
        if is_main:
            save_checkpoint(outdir / "rsnet_last.pt", model=_unwrap_ddp(model), optimizer=opt, epoch=epoch, step=global_step, extra=extra)
            if val_metrics:
                v = float(val_metrics.get("loss", float("nan")))
                if math.isfinite(v) and v < best_val_loss:
                    best_val_loss = v
                    extra["best_val_loss"] = best_val_loss
                    save_checkpoint(outdir / "rsnet_best.pt", model=_unwrap_ddp(model), optimizer=opt, epoch=epoch, step=global_step, extra=extra)
                    log_message(f"Saved best checkpoint (val_loss={best_val_loss:.4f})", logger, console=True, use_tqdm=True)
                save_vis(epoch)

    if distributed:
        dist.destroy_process_group()

    log_message(f"Done. Saved to {outdir}", logger, console=is_main, use_tqdm=False)


if __name__ == "__main__":
    main()
