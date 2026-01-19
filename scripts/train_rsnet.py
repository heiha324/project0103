#!/usr/bin/env python3
"""Train RS-Net cloud detector."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import logging
from pathlib import Path
from typing import Dict
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from sarcloud.data.cloud_dataset import (
    CloudBucketSampler,
    CloudCropDataset,
    DistributedCloudBucketSampler,
    WHUOriCropDataset,
)
from sarcloud.models.rsnet import RSNet
from sarcloud.utils.config import load_config
from sarcloud.utils.metrics import (
    compute_iou,
    dice_loss,
    false_positive_rate,
    f1_score,
    overall_accuracy,
    precision_recall,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def progress(loader, desc: str, enabled: bool, leave: bool, total: int | None = None):
    if not enabled or tqdm is None:
        return loader
    return tqdm(loader, desc=desc, ncols=80, leave=leave, total=total)


def _chain_first(first_batch, iterator):
    yield first_batch
    yield from iterator


def debug_print(msg: str, use_tqdm: bool) -> None:
    if tqdm is not None and use_tqdm:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def log_message(msg: str, logger: logging.Logger | None, console: bool, use_tqdm: bool) -> None:
    if logger is not None:
        logger.info(msg)
    if console:
        debug_print(msg, use_tqdm=use_tqdm)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Val",
    enabled: bool = True,
) -> Dict[str, float]:
    model.eval()
    ious = []
    precisions = []
    recalls = []
    fprs = []
    f1s = []
    oas = []
    with torch.no_grad():
        for images, masks in progress(loader, desc=desc, enabled=enabled, leave=False, total=len(loader)):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            ious.append(compute_iou(probs, masks))
            precision, recall = precision_recall(probs, masks)
            precisions.append(precision)
            recalls.append(recall)
            fprs.append(false_positive_rate(probs, masks))
            f1s.append(f1_score(probs, masks))
            oas.append(overall_accuracy(probs, masks))
    return {
        "iou": float(sum(ious) / max(1, len(ious))),
        "precision": float(sum(precisions) / max(1, len(precisions))),
        "recall": float(sum(recalls) / max(1, len(recalls))),
        "fpr": float(sum(fprs) / max(1, len(fprs))),
        "far": float(sum(fprs) / max(1, len(fprs))),
        "f1": float(sum(f1s) / max(1, len(f1s))),
        "oa": float(sum(oas) / max(1, len(oas))),
    }


def build_dataset(
    data_cfg: Dict,
    augment: bool,
    jitter: int | None = None,
    cloud_min_ratio: float | None = None,
    cloud_keep_ratio: float | None = None,
) -> CloudCropDataset:
    dataset_type = data_cfg.get("dataset", "flat")
    jitter = data_cfg.get("jitter", 16) if jitter is None else jitter
    cloud_min_ratio = data_cfg.get("cloud_min_ratio", 0.01) if cloud_min_ratio is None else cloud_min_ratio
    cloud_keep_ratio = data_cfg.get("cloud_keep_ratio", 0.2) if cloud_keep_ratio is None else cloud_keep_ratio
    if dataset_type == "whu_ori":
        return WHUOriCropDataset(
            root=data_cfg["root"],
            split=data_cfg.get("split", "Train"),
            level_subdir=data_cfg.get("level_subdir", "level1_10m"),
            image_ext=data_cfg.get("image_ext", ".npy"),
            mask_ext=data_cfg.get("mask_ext", ".npy"),
            crop_size=data_cfg.get("crop_size", 256),
            base_stride=data_cfg.get("base_stride", 128),
            jitter=jitter,
            cloud_min_ratio=cloud_min_ratio,
            cloud_keep_ratio=cloud_keep_ratio,
            bands=data_cfg.get("bands"),
            s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
            s2_clip_max=data_cfg.get("s2_clip_max", 1.0),
            augment=augment,
        )
    return CloudCropDataset(
        root=data_cfg["root"],
        images_subdir=data_cfg["images_subdir"],
        masks_subdir=data_cfg["masks_subdir"],
        image_ext=data_cfg.get("image_ext", ".npy"),
        mask_ext=data_cfg.get("mask_ext", ".npy"),
        crop_size=data_cfg.get("crop_size", 256),
        base_stride=data_cfg.get("base_stride", 128),
        jitter=jitter,
        cloud_min_ratio=cloud_min_ratio,
        cloud_keep_ratio=cloud_keep_ratio,
        bands=data_cfg.get("bands"),
        s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
        s2_clip_max=data_cfg.get("s2_clip_max", 10000.0),
        augment=augment,
    )


def build_loader(
    dataset: torch.utils.data.Dataset,
    data_cfg: Dict,
    batch_sampler=None,
    batch_size: int | None = None,
    shuffle: bool = False,
):
    loader_cfg = data_cfg.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", data_cfg.get("num_workers", 4)))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))
    timeout = int(loader_cfg.get("timeout", 0))

    kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
        if timeout > 0:
            kwargs["timeout"] = timeout

    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rsnet.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ddp, rank, world_size, local_rank = init_distributed()
    is_main = rank == 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    base_seed = cfg.get("seed", 42)
    set_seed(base_seed + rank)

    train_cfg = cfg["train"]
    quiet = bool(train_cfg.get("quiet", False))
    data_cfg = cfg["data"]
    val_cfg = cfg.get("val", data_cfg)
    output_cfg = cfg.get("output", {})
    out_dir = Path(output_cfg["dir"])
    fmt = output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S")
    run_ts = time.strftime(fmt) if is_main else ""
    if ddp:
        ts_holder = [run_ts]
        dist.broadcast_object_list(ts_holder, src=0)
        run_ts = ts_holder[0]
    if output_cfg.get("auto_timestamp", False):
        out_dir = out_dir.parent / f"{out_dir.name}_{run_ts}"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_dir_cfg = output_cfg.get("log_dir", "logs")
    log_dir = Path(log_dir_cfg)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_path = None
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
        if output_cfg.get("auto_timestamp", False):
            log_path = log_dir / f"{out_dir.name}.log"
        else:
            log_path = log_dir / f"{out_dir.name}_{run_ts}.log"
    logger = None
    if is_main and log_path is not None:
        logger = logging.getLogger("train")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
        log_message(f"Run timestamp: {run_ts}", logger, console=False, use_tqdm=False)
        log_message(f"Output dir: {out_dir}", logger, console=False, use_tqdm=False)
        log_message(f"World size: {world_size}, rank: {rank}", logger, console=False, use_tqdm=False)
        log_message(f"Config: {args.config}", logger, console=False, use_tqdm=False)

    train_dataset = build_dataset(data_cfg, augment=True)

    if ddp:
        sampler = DistributedCloudBucketSampler(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            num_replicas=world_size,
            rank=rank,
            seed=base_seed,
        )
    else:
        sampler = CloudBucketSampler(train_dataset, batch_size=train_cfg["batch_size"])
    train_loader = build_loader(train_dataset, data_cfg, batch_sampler=sampler)

    val_dataset = build_dataset(
        val_cfg,
        augment=False,
        jitter=0,
        cloud_min_ratio=0.0,
        cloud_keep_ratio=1.0,
    )

    val_loader = build_loader(
        val_dataset,
        val_cfg,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )

    model = RSNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"].get("base_channels", 32),
        depth=cfg["model"].get("depth", 4),
        use_batchnorm=cfg["model"].get("use_batchnorm", True),
    ).to(device)
    if ddp:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)

    pos_weight = torch.tensor(train_cfg.get("pos_weight", 5.0), device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    amp_enabled = bool(train_cfg.get("amp", False))
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled) if amp_enabled else None
    best_score = -math.inf

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        train_desc = f"Train {epoch+1}/{train_cfg['num_epochs']}"
        if ddp and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        debug_cfg = train_cfg.get("debug", {})
        debug_interval = int(debug_cfg.get("interval", 0))
        debug_all_ranks = bool(debug_cfg.get("all_ranks", False))
        debug_sync = bool(debug_cfg.get("sync_cuda", False))
        debug_first_batch = bool(debug_cfg.get("first_batch", False))
        trace_steps = int(debug_cfg.get("trace_steps", 0))
        if quiet:
            debug_interval = 0
            debug_all_ranks = False
            debug_first_batch = False
            trace_steps = 0
        last_end = time.perf_counter()

        train_iter = iter(train_loader)
        if debug_first_batch:
            if debug_all_ranks or is_main:
                msg = f"[rank {rank}] epoch {epoch+1} waiting for first batch..."
                if not quiet:
                    debug_print(msg, use_tqdm=is_main)
            t_fetch_start = time.perf_counter()
            first_batch = next(train_iter)
            t_fetch = time.perf_counter() - t_fetch_start
            if debug_all_ranks or is_main:
                msg = f"[rank {rank}] epoch {epoch+1} first batch fetched in {t_fetch:.3f}s"
                if not quiet:
                    debug_print(msg, use_tqdm=is_main)
            train_iter = _chain_first(first_batch, train_iter)
            last_end = time.perf_counter()

        for step_idx, (images, masks) in enumerate(
            progress(
                train_iter,
                desc=train_desc,
                enabled=is_main and not quiet,
                leave=True,
                total=len(train_loader),
            ),
            start=1,
        ):
            data_time = time.perf_counter() - last_end
            step_start = time.perf_counter()
            trace_this = trace_steps > 0 and step_idx <= trace_steps
            should_print = debug_all_ranks or is_main
            if trace_this and should_print:
                if not quiet:
                    debug_print(
                        f"[rank {rank}] step {step_idx} start data={data_time:.3f}s",
                        use_tqdm=is_main,
                    )
            images = images.to(device)
            masks = masks.to(device)
            if trace_this and debug_sync and device.type == "cuda":
                torch.cuda.synchronize()
            if trace_this and should_print:
                if not quiet:
                    debug_print(f"[rank {rank}] step {step_idx} to_device done", use_tqdm=is_main)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.amp.autocast(amp_device, enabled=True):
                    logits = model(images)
                    loss_bce = bce(logits, masks)
                    loss_dice = dice_loss(torch.sigmoid(logits), masks)
                    loss = loss_bce + train_cfg.get("dice_weight", 1.0) * loss_dice
                if trace_this and debug_sync and device.type == "cuda":
                    torch.cuda.synchronize()
                if trace_this and should_print:
                    if not quiet:
                        debug_print(f"[rank {rank}] step {step_idx} forward done", use_tqdm=is_main)
                assert scaler is not None
                scaler.scale(loss).backward()
                if trace_this and debug_sync and device.type == "cuda":
                    torch.cuda.synchronize()
                if trace_this and should_print:
                    if not quiet:
                        debug_print(f"[rank {rank}] step {step_idx} backward done", use_tqdm=is_main)
                if train_cfg.get("grad_clip", 0.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                if trace_this and debug_sync and device.type == "cuda":
                    torch.cuda.synchronize()
                if trace_this and should_print:
                    if not quiet:
                        debug_print(f"[rank {rank}] step {step_idx} optim done", use_tqdm=is_main)
            else:
                logits = model(images)
                loss_bce = bce(logits, masks)
                loss_dice = dice_loss(torch.sigmoid(logits), masks)
                loss = loss_bce + train_cfg.get("dice_weight", 1.0) * loss_dice
                if trace_this and debug_sync and device.type == "cuda":
                    torch.cuda.synchronize()
                if trace_this and should_print:
                    if not quiet:
                        debug_print(f"[rank {rank}] step {step_idx} forward done", use_tqdm=is_main)
                loss.backward()
                if trace_this and debug_sync and device.type == "cuda":
                    torch.cuda.synchronize()
                if trace_this and should_print:
                    if not quiet:
                        debug_print(f"[rank {rank}] step {step_idx} backward done", use_tqdm=is_main)
                if train_cfg.get("grad_clip", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                optimizer.step()
                if trace_this and debug_sync and device.type == "cuda":
                    torch.cuda.synchronize()
                if trace_this and should_print:
                    if not quiet:
                        debug_print(f"[rank {rank}] step {step_idx} optim done", use_tqdm=is_main)

            if debug_sync and device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start
            last_end = time.perf_counter()
            if debug_interval > 0 and (step_idx == 1 or step_idx % debug_interval == 0):
                if debug_all_ranks or is_main:
                    mem = ""
                    if device.type == "cuda":
                        mem = f", mem={torch.cuda.memory_allocated(device) / 1024**2:.1f}MiB"
                    msg = (
                        f"[rank {rank}] step {step_idx}/{len(train_loader)} "
                        f"data={data_time:.3f}s step={step_time:.3f}s loss={loss.item():.4f}{mem}"
                    )
                    if tqdm is not None and is_main:
                        tqdm.write(msg)
                    else:
                        print(msg, flush=True)

        metrics = None
        if is_main:
            eval_model = model.module if ddp else model
            metrics = evaluate(eval_model, val_loader, device, desc="Val", enabled=not quiet)
        if ddp:
            dist.barrier()
        if is_main and metrics is not None:
            score = metrics["iou"] + 0.5 * metrics["recall"] - 0.25 * metrics["fpr"]
            state_dict = model.module.state_dict() if ddp else model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state": state_dict,
                "metrics": metrics,
                "score": score,
                "model_cfg": cfg["model"],
            }
            torch.save(checkpoint, out_dir / "rsnet_last.pth")
            log_message(f"Epoch {epoch+1}: saved rsnet_last.pth", logger, console=False, use_tqdm=False)
            if score > best_score:
                best_score = score
                torch.save(checkpoint, out_dir / "rsnet_best.pth")
                log_message(f"Epoch {epoch+1}: saved rsnet_best.pth", logger, console=False, use_tqdm=False)

            summary = (
                f"Epoch {epoch+1}/{train_cfg['num_epochs']} - OA {metrics['oa']:.3f} "
                f"Pre {metrics['precision']:.3f} FAR {metrics['far']:.3f} "
                f"Recall {metrics['recall']:.3f} F1 {metrics['f1']:.3f} "
                f"IoU {metrics['iou']:.3f} Score {score:.3f}"
            )
            log_message(summary, logger, console=not quiet, use_tqdm=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
