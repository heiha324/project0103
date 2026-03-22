#!/usr/bin/env python3
"""Train Residual Shifting diffusion with a transformer denoiser."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import train_diffusion_rs as base
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion.losses import grad_l1_loss
from sarcloud.models.cond_transformer import ConditionalTransformer
from sarcloud.training.ema import EMA
from sarcloud.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Residual Shifting Transformer Model")
    parser.add_argument("--config", type=str, default="configs/diffusion_rs_transformer.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ddp, rank, world_size, local_rank = base.init_distributed()
    is_main = rank == 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    base_seed = cfg.get("seed", 42)
    base.set_seed(base_seed + rank)

    data_cfg = cfg["sen12ms"]
    dataset = base.build_dataset(data_cfg)

    sampler = None
    if ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=base_seed,
        )

    loader = base.build_loader(
        dataset,
        data_cfg,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=ddp,
        collate_fn=base.collate_sen12mscr,
    )

    eval_cfg = cfg.get("test") or cfg.get("val") or data_cfg
    eval_dataset = base.build_dataset(eval_cfg)

    eval_sampler = None
    if ddp:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    eval_loader = base.build_loader(
        eval_dataset,
        eval_cfg,
        batch_size=eval_cfg.get("batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        sampler=eval_sampler,
        drop_last=False,
        collate_fn=base.collate_sen12mscr,
    )

    eval_max_batches = int(eval_cfg.get("max_batches", 0))
    if ddp and eval_max_batches > 0:
        eval_max_batches = max(1, eval_max_batches // world_size)

    model_cfg = cfg["model"]
    model = ConditionalTransformer(
        x_channels=model_cfg["x_channels"],
        y_channels=model_cfg["y_channels"],
        s_channels=model_cfg["s_channels"],
        embed_dim=model_cfg.get("embed_dim", 256),
        depth=model_cfg.get("depth", 8),
        num_heads=model_cfg.get("num_heads", 8),
        patch_size=model_cfg.get("patch_size", 16),
        time_dim=model_cfg.get("time_dim", 256),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        refine_channels=model_cfg.get("refine_channels", 128),
    ).to(device)

    if ddp:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=False, gradient_as_bucket_view=True)

    diff_cfg = cfg.get("diffusion", {})
    schedule_cfg = cfg.get("schedule", {})
    diffusion = ResidualShiftingDiffusion(
        timesteps=diff_cfg.get("timesteps", 1000),
        kappa=diff_cfg.get("kappa", 1.0),
        schedule_type=diff_cfg.get("schedule_type", "linear"),
        min_eta=diff_cfg.get("min_eta", 0.001),
        max_eta=diff_cfg.get("max_eta", 0.99),
        power=diff_cfg.get("power", 2.0),
        x0_clip_min=schedule_cfg.get("x0_clip_min", 0.0),
        x0_clip_max=schedule_cfg.get("x0_clip_max", 1.0),
    )

    base_lr = cfg["train"].get("lr", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    num_epochs = cfg["train"]["num_epochs"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 5)
    use_scheduler = cfg["train"].get("use_scheduler", True)
    scheduler = None
    if use_scheduler:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=base_lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    amp_enabled = bool(cfg["train"].get("amp", False))
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled) if amp_enabled else None

    ema_model = model.module if ddp else model
    ema = EMA(ema_model, decay=cfg["train"].get("ema_decay", 0.999))

    start_epoch = 0
    best_loss = math.inf
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank} if ddp else device
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

        if "model_state" in checkpoint:
            load_target = model.module if ddp else model
            missing, unexpected = load_target.load_state_dict(checkpoint["model_state"], strict=False)
            if is_main:
                if missing:
                    print(f"WARNING: Missing keys: {missing}")
                if unexpected:
                    print(f"WARNING: Unexpected keys: {unexpected}")
                print(f"Loaded model from {ckpt_path}")

        if "ema_state" in checkpoint:
            ema.shadow = checkpoint["ema_state"]
            if is_main:
                print(f"Loaded EMA state from {ckpt_path}")

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if is_main:
                print(f"Loaded optimizer state from {ckpt_path}")

        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if is_main:
                print(f"Loaded scheduler state from {ckpt_path}")

        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
            if is_main:
                print(f"Loaded GradScaler state from {ckpt_path}")

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            if is_main:
                print(f"Resuming from epoch {start_epoch}")

        resume_best_loss = checkpoint.get("best_loss")
        if resume_best_loss is None:
            test_metrics = checkpoint.get("test_metrics") or {}
            resume_best_loss = test_metrics.get("loss")
        if resume_best_loss is not None and math.isfinite(float(resume_best_loss)):
            best_loss = float(resume_best_loss)
            if is_main:
                print(f"Loaded best loss {best_loss:.4f} from {ckpt_path}")

        if (
            start_epoch > 0
            and "optimizer_state" not in checkpoint
            and scheduler is not None
            and checkpoint.get("scheduler_state") is None
        ):
            for _ in range(start_epoch):
                scheduler.step()
            if is_main:
                print(f"Advanced scheduler by {start_epoch} epochs to match resumed training")

    output_cfg = cfg.get("output", {})
    out_dir = Path(output_cfg["dir"])
    fmt = output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S")
    run_ts = time.strftime(fmt) if is_main else ""

    if ddp:
        ts_holder = [run_ts]
        dist.broadcast_object_list(ts_holder, src=0)
        run_ts = ts_holder[0]

    if output_cfg.get("auto_timestamp", True):
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
        logger = logging.getLogger("train_diffusion_rs_transformer")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
        base.log_message(f"[Residual Shifting Transformer] 运行时间戳: {run_ts}", logger, console=False, use_tqdm=False)
        base.log_message(f"输出目录: {out_dir}", logger, console=False, use_tqdm=False)
        base.log_message(f"World size: {world_size}, Rank: {rank}", logger, console=False, use_tqdm=False)
        base.log_message(f"配置文件: {args.config}", logger, console=False, use_tqdm=False)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)

        iterator = loader
        if base.tqdm is not None and is_main:
            iterator = base.tqdm(loader, desc=f"Train {epoch+1}/{num_epochs}", ncols=80)

        epoch_loss = 0.0
        steps = 0
        for s1, s2_cloudy, s2_clear, _alpha in iterator:
            s1 = s1.to(device)
            y = s2_cloudy.to(device)
            x0 = s2_clear.to(device)

            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, y, t, noise)

            aux_time_weight = base.compute_time_weight(
                t,
                diffusion.timesteps,
                min_weight=cfg["loss"].get("aux_min_weight", 0.1),
                max_weight=cfg["loss"].get("aux_max_weight", 1.0),
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=amp_enabled):
                eps_pred = model(x_t, t, y, s1)
                loss_diff = F.mse_loss(eps_pred, noise)

                x0_pred = diffusion.predict_x0_from_eps(x_t, y, t, eps_pred, clip=False)
                x0_pred = x0_pred.clamp(diffusion.x0_clip_min, diffusion.x0_clip_max)

                loss_recon_raw = F.l1_loss(x0_pred, x0, reduction="none")
                grad_weight_map = torch.ones_like(x0[:, :1, :, :])
                weight_map = grad_weight_map * aux_time_weight
                loss_recon = base.weighted_mean(loss_recon_raw, aux_time_weight)
                loss_grad = grad_l1_loss(x0_pred, x0, weight_map)

                recon_weight = cfg["loss"].get("recon_weight", 1.0)
                grad_weight = cfg["loss"].get("grad_weight", 0.5)
                loss = loss_diff + recon_weight * loss_recon + grad_weight * loss_grad

            loss_is_finite = torch.isfinite(loss.detach())
            if ddp:
                flag = torch.tensor(float(loss_is_finite.item()), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                loss_is_finite = flag.item() == 1.0
            else:
                loss_is_finite = bool(loss_is_finite.item())

            if not loss_is_finite:
                if is_main:
                    base.log_message(
                        f"ERROR: NaN/Inf loss detected at epoch {epoch+1} step {steps} BEFORE optimizer.step()",
                        logger,
                        console=True,
                        use_tqdm=True,
                    )
                    base.log_message(f"  loss_diff: {loss_diff.item()}", logger, console=True, use_tqdm=True)
                    base.log_message(f"  loss_recon: {loss_recon.item()}", logger, console=True, use_tqdm=True)
                    base.log_message(f"  loss_grad: {loss_grad.item()}", logger, console=True, use_tqdm=True)
                    base.log_message(
                        f"  t[min,max,mean]: {int(t.min().item())}, {int(t.max().item())}, {float(t.float().mean().item()):.2f}",
                        logger,
                        console=True,
                        use_tqdm=True,
                    )
                    base.log_message(
                        f"  finite flags: x0={bool(torch.isfinite(x0).all().item())} "
                        f"y={bool(torch.isfinite(y).all().item())} s1={bool(torch.isfinite(s1).all().item())} "
                        f"x_t={bool(torch.isfinite(x_t).all().item())} eps_pred={bool(torch.isfinite(eps_pred).all().item())}",
                        logger,
                        console=True,
                        use_tqdm=True,
                    )
                    base.log_message(
                        f"  x_t[min,max]: {float(x_t.detach().min().item()):.4f}, {float(x_t.detach().max().item()):.4f}",
                        logger,
                        console=True,
                        use_tqdm=True,
                    )
                    base.log_message(
                        f"  eps_pred[min,max]: {float(eps_pred.detach().nan_to_num().min().item()):.4f}, "
                        f"{float(eps_pred.detach().nan_to_num().max().item()):.4f}",
                        logger,
                        console=True,
                        use_tqdm=True,
                    )
                if ddp and dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
                raise RuntimeError(f"NaN/Inf loss at epoch {epoch+1} step {steps}")

            if amp_enabled:
                assert scaler is not None
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if cfg["train"].get("grad_clip", 0.0) > 0:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])

            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            ema.update(ema_model)

            epoch_loss += float(loss.item())
            steps += 1
            if base.tqdm is not None and is_main and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(f"loss={loss.item():.4f}")

        if use_scheduler and scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = epoch_loss / max(1, steps)
        if is_main:
            base.log_message(
                f"Epoch {epoch+1}/{num_epochs} - train_loss {train_loss:.4f} lr {current_lr:.2e}",
                logger,
                console=True,
                use_tqdm=True,
            )

        eval_metrics = None
        eval_model = model.module if ddp else model
        eval_metrics = base.evaluate(
            eval_model,
            eval_loader,
            diffusion,
            cfg,
            device,
            amp_device,
            amp_enabled,
            desc="Test",
            max_batches=eval_max_batches,
            use_tqdm=True,
            ema=ema,
            use_ema=True,
            ddp=ddp,
        )

        if is_main:
            if eval_metrics is not None:
                base.log_message(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"test_loss {eval_metrics['loss']:.4f} diff {eval_metrics['diff']:.4f} "
                    f"recon {eval_metrics['recon']:.4f} grad {eval_metrics['grad']:.4f}\n"
                    f"  MAE {eval_metrics.get('mae', 0):.4f} MSE {eval_metrics.get('mse', 0):.4f} "
                    f"RMSE {eval_metrics.get('rmse', 0):.4f} PSNR {eval_metrics.get('psnr', 0):.2f}\n"
                    f"  SSIM {eval_metrics.get('ssim', 0):.4f} MS-SSIM {eval_metrics.get('ms_ssim', 0):.4f} "
                    f"SAM {eval_metrics.get('sam', 0):.2f} ERGAS {eval_metrics.get('ergas', 0):.2f}\n"
                    f"  CC {eval_metrics.get('cc', 0):.4f} UIQI {eval_metrics.get('uiqi', 0):.4f} "
                    f"RASE {eval_metrics.get('rase', 0):.2f}",
                    logger,
                    console=True,
                    use_tqdm=True,
                )

            base.save_vis_samples(
                eval_model,
                diffusion,
                eval_dataset,
                cfg,
                device,
                epoch,
                base_seed,
                out_dir,
                logger,
                use_tqdm=True,
                ema=ema,
                use_ema=True,
            )

        if ddp:
            dist.barrier()

        if is_main:
            model_state = model.module.state_dict() if ddp else model.state_dict()
            current_best_loss = best_loss
            current_test_loss = float("nan")
            if eval_metrics is not None:
                current_test_loss = float(eval_metrics.get("loss", float("nan")))
                if math.isfinite(current_test_loss) and current_test_loss < current_best_loss:
                    current_best_loss = current_test_loss
            checkpoint = {
                "epoch": epoch,
                "model_state": model_state,
                "ema_state": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_loss": current_best_loss,
                "config": cfg,
                "train_loss": train_loss,
                "test_metrics": eval_metrics,
            }
            torch.save(checkpoint, out_dir / "diffusion_rs_transformer_last.pth")
            torch.save({"ema_state": ema.shadow}, out_dir / "diffusion_rs_transformer_ema.pth")
            base.log_message(
                f"Epoch {epoch+1}/{num_epochs} - checkpoints saved",
                logger,
                console=True,
                use_tqdm=True,
            )

            if eval_metrics is not None:
                if math.isfinite(current_test_loss) and current_test_loss < best_loss:
                    best_loss = current_test_loss
                    checkpoint["best_loss"] = best_loss
                    torch.save(checkpoint, out_dir / "diffusion_rs_transformer_best.pth")
                    base.log_message(
                        f"Epoch {epoch+1}/{num_epochs} - saved diffusion_rs_transformer_best.pth (test_loss {current_test_loss:.4f})",
                        logger,
                        console=True,
                        use_tqdm=True,
                    )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
