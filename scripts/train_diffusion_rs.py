#!/usr/bin/env python3
"""残差偏移扩散模型训练脚本 (Residual Shifting Diffusion Training)。

本脚本实现了基于 Residual Shifting 扩散的去云模型训练，与标准 DDPM 的区别：
1. 扩散过程从有云图像 y 向清晰图像 x0 转换
2. 采样起点是 y + noise，而不是纯噪声
3. 前向过程: x_t = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*noise

支持：
- 多卡并行训练 (DDP) 与混合精度加速 (AMP)
- 动态学习率调度 (Warmup + Cosine Decay)
- 指数移动平均 (EMA) 模型权重
- 训练过程中的实时采样与可视化
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

# 将 src 目录添加到 Python 路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

# 可选依赖
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# 训练阶段不直接渲染 PNG，可视化采样先保存为 NPY，再由离线脚本渲染。
plt = None

# 项目模块 - 使用 Residual Shifting 扩散
from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset, collate_sen12mscr
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion.losses import grad_l1_loss
from sarcloud.diffusion.sampling_rs import sample_batch_rs
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.training.ema import EMA
from sarcloud.training.samplers import DistributedEvalSamplerNoPad
from sarcloud.utils.config import load_config
from sarcloud.utils.metrics import (
    mae, mse, rmse, nrmse, psnr, ssim, ms_ssim, sam, ergas, cc, uiqi, rase
)


METRIC_PROTOCOL = "image_mean_v1"
LOSS_KEYS = ("loss", "diff", "recon", "grad")
IMAGE_METRIC_FNS = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "nrmse": nrmse,
    "psnr": psnr,
    "ssim": ssim,
    "ms_ssim": ms_ssim,
    "sam": sam,
    "ergas": ergas,
    "cc": cc,
    "uiqi": uiqi,
    "rase": rase,
}
IMAGE_METRIC_KEYS = tuple(IMAGE_METRIC_FNS.keys())


def set_seed(seed: int) -> None:
    """设置全局随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, int]:
    """初始化分布式训练环境。"""
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


def log_message(msg: str, logger: logging.Logger | None, console: bool, use_tqdm: bool) -> None:
    """统一日志打印函数。"""
    if logger is not None:
        logger.info(msg)
    if console:
        if tqdm is not None and use_tqdm:
            tqdm.write(msg)
        else:
            print(msg, flush=True)


def apply_ema_weights(model: torch.nn.Module, ema: EMA) -> Dict[str, torch.Tensor]:
    """将 EMA 权重应用到模型，并备份原始权重。"""
    backup: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            backup[name] = param.detach().clone()
            param.data.copy_(ema.shadow[name])
    return backup


def restore_weights(model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
    """恢复模型的原始权重。"""
    for name, param in model.named_parameters():
        if name in backup:
            param.data.copy_(backup[name])


def build_distributed_eval_sampler(dataset, ddp: bool, rank: int, world_size: int) -> Sampler | None:
    """构建不重复样本的 DDP 评估采样器。"""
    if not ddp:
        return None
    return DistributedEvalSamplerNoPad(dataset, num_replicas=world_size, rank=rank)


def accumulate_image_metric_totals(
    totals: Dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    metric_keys: tuple[str, ...] = IMAGE_METRIC_KEYS,
) -> int:
    """按单图指标均值累加，返回本批样本数。"""
    batch_size = int(pred.shape[0]) if pred.ndim == 4 else 1
    for key in metric_keys:
        totals[key] += IMAGE_METRIC_FNS[key](pred, target) * batch_size
    return batch_size


def mean_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric_keys: tuple[str, ...],
) -> Dict[str, float]:
    """计算一批样本的单图平均指标。"""
    totals = {key: 0.0 for key in metric_keys}
    sample_count = accumulate_image_metric_totals(totals, pred, target, metric_keys=metric_keys)
    if sample_count <= 0:
        return {key: float("nan") for key in metric_keys}
    return {key: totals[key] / sample_count for key in metric_keys}


def compute_time_weight(
    t: torch.Tensor,
    timesteps: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> torch.Tensor:
    """计算时间相关的辅助 Loss 权重。
    
    在 Residual Shifting 中，t 接近 T 时图像接近 y（有云），此时权重较小；
    t 接近 0 时图像接近 x0（清晰），此时权重较大。
    """
    denom = max(timesteps - 1, 1)
    t_ratio = t.float() / denom
    weight = max_weight - (max_weight - min_weight) * t_ratio
    return weight.view(-1, 1, 1, 1)


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算带权平均，支持广播权重。"""
    if loss.shape != weight.shape:
        weight = weight.expand_as(loss)
    denom = weight.sum().clamp_min(eps)
    return (loss * weight).sum() / denom


def build_dataset(data_cfg: Dict):
    """根据配置构建数据集。"""
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


def build_loader(
    dataset,
    data_cfg: Dict,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler | None = None,
    drop_last: bool = False,
    collate_fn=None,
) -> DataLoader:
    """构建 DataLoader。"""
    num_workers = int(data_cfg.get("num_workers", 4))
    if sampler is not None:
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    diffusion: ResidualShiftingDiffusion,
    cfg: Dict,
    device: torch.device,
    amp_device: str,
    amp_enabled: bool,
    desc: str,
    max_batches: int,
    use_tqdm: bool,
    ema: EMA | None = None,
    use_ema: bool = False,
    ddp: bool = False,
) -> Dict[str, float]:
    """执行模型评估 (使用 Residual Shifting)。"""
    backup = None
    if use_ema and ema is not None:
        backup = apply_ema_weights(model, ema)
    
    eval_seed = cfg.get("seed", 42) + 10000
    if ddp:
        eval_seed += dist.get_rank()
    eval_rng = torch.Generator(device=device)
    eval_rng.manual_seed(eval_seed)
    
    model.eval()
    totals = {key: 0.0 for key in LOSS_KEYS + IMAGE_METRIC_KEYS}
    steps = 0
    sample_count = 0
    
    show_progress = use_tqdm and (not ddp or dist.get_rank() == 0)
    iterator = loader
    if tqdm is not None and show_progress:
        iterator = tqdm(loader, desc=desc, ncols=80, leave=False)
        
    with torch.no_grad():
        for step_idx, (s1, s2_cloudy, s2_clear, _alpha) in enumerate(iterator, start=1):
            if max_batches > 0 and step_idx > max_batches:
                break
            s1 = s1.to(device)
            y = s2_cloudy.to(device)  # 有云图像
            x0 = s2_clear.to(device)  # 清晰图像

            # 随机采样时间步
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device, generator=eval_rng)
            noise = torch.randn(x0.shape, device=device, generator=eval_rng)
            
            # ===== 关键修改: Residual Shifting 前向过程需要 y =====
            x_t = diffusion.q_sample(x0, y, t, noise)

            with torch.amp.autocast(amp_device, enabled=amp_enabled):
                eps_pred = model(x_t, t, y, s1)
                loss_diff = F.mse_loss(eps_pred, noise)
                
                # ===== 关键修改: predict_x0 需要 y =====
                x0_pred = diffusion.predict_x0_from_eps(x_t, y, t, eps_pred, clip=False)
                x0_pred = x0_pred.clamp(diffusion.x0_clip_min, diffusion.x0_clip_max)
                
                aux_time_weight = compute_time_weight(
                    t, diffusion.timesteps,
                    min_weight=cfg["loss"].get("aux_min_weight", 0.1),
                    max_weight=cfg["loss"].get("aux_max_weight", 1.0),
                )
                # 按样本时间步加权 (E[w * loss])，避免先均值化权重
                loss_recon_raw = F.l1_loss(x0_pred, x0, reduction="none")
                grad_weight_map = torch.ones_like(x0[:, :1, :, :])
                weight_map = grad_weight_map * aux_time_weight
                loss_recon = weighted_mean(loss_recon_raw, aux_time_weight)
                loss_grad = grad_l1_loss(x0_pred, x0, weight_map)
                
                recon_weight = cfg["loss"].get("recon_weight", 1.0)
                grad_weight = cfg["loss"].get("grad_weight", 0.5)
                loss = loss_diff + recon_weight * loss_recon + grad_weight * loss_grad
                
                x0_p = x0_pred.detach().float()
                x0_t = x0.detach().float()
                batch_size = accumulate_image_metric_totals(totals, x0_p, x0_t)

            totals["loss"] += float(loss.item())
            totals["diff"] += float(loss_diff.item())
            totals["recon"] += float(loss_recon.item())
            totals["grad"] += float(loss_grad.item())
            steps += 1
            sample_count += batch_size
    
    if backup is not None:
        restore_weights(model, backup)
    
    # DDP 聚合
    if ddp:
        vals = [totals[key] for key in LOSS_KEYS + IMAGE_METRIC_KEYS] + [float(steps), float(sample_count)]
        metrics_tensor = torch.tensor(vals, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_steps = metrics_tensor[-2].item()
        total_samples = metrics_tensor[-1].item()
        if total_steps == 0 or total_samples == 0:
            return {k: float("nan") for k in totals}
        res = {}
        for i, key in enumerate(LOSS_KEYS):
            res[key] = metrics_tensor[i].item() / total_steps
        offset = len(LOSS_KEYS)
        for j, key in enumerate(IMAGE_METRIC_KEYS):
            res[key] = metrics_tensor[offset + j].item() / total_samples
        return res

    if steps == 0 or sample_count == 0:
        return {k: float("nan") for k in totals}
    out = {key: totals[key] / steps for key in LOSS_KEYS}
    out.update({key: totals[key] / sample_count for key in IMAGE_METRIC_KEYS})
    return out


def _auto_scale_rgb(rgb: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    """自动拉伸 RGB 对比度。"""
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if high_p <= low_p:
        return np.clip(rgb, 0.0, 1.0)
    lo, hi = np.percentile(rgb, [low_p, high_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.clip(rgb, 0.0, 1.0)
    rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0)


def _compute_channel_percentiles(rgb: np.ndarray, low_p: float, high_p: float):
    """计算每个通道的百分位数。"""
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC rgb array, got shape {rgb.shape}")
    if high_p <= low_p:
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    flat = rgb.reshape(-1, 3)
    lo, hi = np.percentile(flat, [low_p, high_p], axis=0)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    return lo.astype(np.float32), hi.astype(np.float32)


def _apply_scale_rgb(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """应用归一化。"""
    lo = np.asarray(lo, dtype=np.float32).reshape(1, 1, -1)
    hi = np.asarray(hi, dtype=np.float32).reshape(1, 1, -1)
    denom = hi - lo
    denom = np.where(denom < 1e-6, 1.0, denom)
    rgb = (rgb - lo) / denom
    return np.clip(rgb, 0.0, 1.0)


def to_rgb(
    chw: np.ndarray,
    rgb_indices: list[int],
    auto_scale: bool = False,
    scale_percentiles: tuple[float, float] = (1.0, 99.0),
    scale_params=None,
    per_channel: bool = False,
) -> np.ndarray:
    """将多波段图像转为 RGB。"""
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {chw.shape}")
    if max(rgb_indices) >= chw.shape[0]:
        raise ValueError(f"RGB indices {rgb_indices} out of range for shape {chw.shape}")
    
    rgb = chw[rgb_indices, ...]
    rgb = np.transpose(rgb, (1, 2, 0))
    
    if scale_params is not None:
        lo, hi = scale_params
        rgb = _apply_scale_rgb(rgb, lo, hi)
    elif auto_scale:
        if per_channel:
            lo, hi = _compute_channel_percentiles(rgb, scale_percentiles[0], scale_percentiles[1])
            rgb = _apply_scale_rgb(rgb, lo, hi)
        else:
            rgb = _auto_scale_rgb(rgb, scale_percentiles[0], scale_percentiles[1])
            
    return np.clip(rgb, 0.0, 1.0)



def save_vis_samples(
    model: torch.nn.Module,
    diffusion: ResidualShiftingDiffusion,
    dataset,
    cfg: Dict,
    device: torch.device,
    epoch: int,
    base_seed: int,
    out_dir: Path,
    logger: logging.Logger | None,
    use_tqdm: bool,
    ema: EMA | None = None,
    use_ema: bool = False,
) -> None:
    """生成可视化采样缓存 (保存为 NPY，PNG 由离线脚本渲染)。"""
    backup = None
    try:
        if len(dataset) == 0:
            return

        vis_cfg = cfg.get("vis", {})
        num_samples = int(vis_cfg.get("num_samples", 10))
        rgb_indices = vis_cfg.get("rgb_indices", [0, 1, 2])
        vis_steps = vis_cfg.get("steps", [50])
        if isinstance(vis_steps, int):
            vis_steps = [vis_steps]

        auto_scale = bool(vis_cfg.get("auto_scale", True))
        scale_percentiles = vis_cfg.get("scale_percentiles", [1.0, 99.0])
        auto_scale_ref = str(vis_cfg.get("auto_scale_ref", "clear")).lower()
        per_channel = bool(vis_cfg.get("per_channel", True))
        if not isinstance(scale_percentiles, (list, tuple)) or len(scale_percentiles) != 2:
            scale_percentiles = [1.0, 99.0]
        scale_percentiles = (float(scale_percentiles[0]), float(scale_percentiles[1]))

        rng = random.SystemRandom()
        sample_count = min(num_samples, len(dataset))
        indices = rng.sample(range(len(dataset)), k=sample_count)

        batch = [dataset[i] for i in indices]
        s1_list, cloudy_list, clear_list, _alpha_list = zip(*batch)

        s1 = torch.stack(s1_list, dim=0).to(device)
        y = torch.stack(cloudy_list, dim=0).to(device)
        x0 = torch.stack(clear_list, dim=0).to(device)

        if use_ema and ema is not None:
            backup = apply_ema_weights(model, ema)

        sampling_cfg = cfg.get("sampling", {})
        vis_root = out_dir / "vis_npy"
        vis_dir = vis_root / f"epoch_{epoch+1:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)

        s1_cpu = s1.cpu()
        y_cpu = y.cpu()
        x0_cpu = x0.cpu()

        np.save(vis_dir / "sample_indices.npy", np.asarray(indices, dtype=np.int64))
        np.save(vis_dir / "s1.npy", s1_cpu.numpy().astype(np.float32))
        np.save(vis_dir / "cloudy.npy", y_cpu.numpy().astype(np.float32))
        np.save(vis_dir / "clear.npy", x0_cpu.numpy().astype(np.float32))

        metrics_map: dict[str, dict[str, float]] = {}
        vis_meta = {
            "epoch": epoch + 1,
            "sample_count": sample_count,
            "sample_indices": indices,
            "rgb_indices": list(rgb_indices),
            "vis_steps": [int(step) for step in vis_steps],
            "auto_scale": auto_scale,
            "scale_percentiles": [float(scale_percentiles[0]), float(scale_percentiles[1])],
            "auto_scale_ref": auto_scale_ref,
            "per_channel": per_channel,
            "vis_gain": 5.0,
            "sampling": {
                "method": sampling_cfg.get("method", "ddim"),
                "eta": float(sampling_cfg.get("eta", 0.0)),
            },
        }

        with torch.inference_mode():
            for step_cnt in vis_steps:
                pred = sample_batch_rs(
                    model,
                    diffusion,
                    y,
                    s1,
                    steps=step_cnt,
                    schedule_cfg=sampling_cfg,
                )
                pred_cpu = torch.clamp(pred.cpu(), 0.0, 1.0)

                log_message(
                    f"[Debug] Vis Prediction Stats (Step {step_cnt}): "
                    f"min={pred_cpu.min():.4f} max={pred_cpu.max():.4f} mean={pred_cpu.mean():.4f}",
                    logger,
                    console=True,
                    use_tqdm=use_tqdm,
                )

                np.save(vis_dir / f"step_{int(step_cnt):04d}.npy", pred_cpu.numpy().astype(np.float32))

                vis_metrics = mean_image_metrics(pred_cpu, x0_cpu, ("psnr", "ssim", "mae"))
                val_psnr = vis_metrics["psnr"]
                val_ssim = vis_metrics["ssim"]
                val_mae = vis_metrics["mae"]
                metrics_map[str(int(step_cnt))] = {
                    "psnr": float(val_psnr),
                    "ssim": float(val_ssim),
                    "mae": float(val_mae),
                }

                log_message(
                    f"[Vis] Step {step_cnt}: PSNR {val_psnr:.2f} SSIM {val_ssim:.4f} MAE {val_mae:.4f}",
                    logger,
                    console=True,
                    use_tqdm=use_tqdm,
                )

        (vis_dir / "meta.json").write_text(
            json.dumps(vis_meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (vis_dir / "metrics.json").write_text(
            json.dumps(metrics_map, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (vis_root / "latest_epoch.txt").write_text(f"{epoch+1:03d}\n", encoding="utf-8")

        log_message(
            f"Saved Residual Shifting visualization arrays to {vis_dir}",
            logger,
            console=True,
            use_tqdm=use_tqdm,
        )

        try:
            from render_vis_samples import render_epoch

            vis_png_dir = out_dir / "vis"
            png_path = render_epoch(
                vis_dir,
                vis_png_dir,
                output_name=f"diffusion_rs_vis_epoch_{epoch+1:03d}.png",
                latest_name="diffusion_rs_vis.png",
            )
            log_message(
                f"Saved Residual Shifting visualization to {png_path}",
                logger,
                console=True,
                use_tqdm=use_tqdm,
            )
        except Exception as exc:
            log_message(
                f"Visualization render failed: {exc}",
                logger,
                console=True,
                use_tqdm=use_tqdm,
            )
    except Exception as exc:
        log_message(f"Visualization cache export failed: {exc}", logger, console=True, use_tqdm=use_tqdm)
        import traceback
        traceback.print_exc()
    finally:
        if backup is not None:
            restore_weights(model, backup)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Residual Shifting Diffusion Model")
    parser.add_argument("--config", type=str, default="configs/diffusion_rs.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # 1. 初始化
    cfg = load_config(args.config)
    ddp, rank, world_size, local_rank = init_distributed()
    is_main = rank == 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    base_seed = cfg.get("seed", 42)
    set_seed(base_seed + rank)

    # 2. 构建数据集
    data_cfg = cfg["sen12ms"]
    dataset = build_dataset(data_cfg)

    # 3. 构建 DataLoader
    sampler = None
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=base_seed)
    
    loader = build_loader(
        dataset, data_cfg,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=ddp,
        collate_fn=collate_sen12mscr,
    )

    eval_cfg = cfg.get("val") or cfg.get("test") or data_cfg
    eval_dataset = build_dataset(eval_cfg)
    
    eval_sampler = None
    if ddp:
        eval_sampler = build_distributed_eval_sampler(eval_dataset, ddp=True, rank=rank, world_size=world_size)
        
    eval_loader = build_loader(
        eval_dataset, eval_cfg,
        batch_size=eval_cfg.get("batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        sampler=eval_sampler,
        drop_last=False,
        collate_fn=collate_sen12mscr,
    )
    
    eval_max_batches = int(eval_cfg.get("max_batches", 0))
    if ddp and eval_max_batches > 0:
        eval_max_batches = max(1, eval_max_batches // world_size)

    # 4. 构建模型
    model = ConditionalUNet(
        x_channels=cfg["model"]["x_channels"],
        y_channels=cfg["model"]["y_channels"],
        s_channels=cfg["model"]["s_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
        time_dim=cfg["model"].get("time_dim", 256),
    ).to(device)
    
    if ddp:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=False, gradient_as_bucket_view=True)

    # ===== 5. 初始化 Residual Shifting 扩散 (关键区别) =====
    diff_cfg = cfg.get("diffusion", {})
    schedule_cfg = cfg.get("schedule", {})
    
    diffusion = ResidualShiftingDiffusion(
        timesteps=diff_cfg.get("timesteps", 1000),
        kappa=diff_cfg.get("kappa", 1.0),
        schedule_type=diff_cfg.get("schedule_type", "exponential"),
        min_eta=diff_cfg.get("min_eta", 0.001),
        max_eta=diff_cfg.get("max_eta", 0.99),
        power=diff_cfg.get("power", 2.0),
        x0_clip_min=schedule_cfg.get("x0_clip_min", 0.0),
        x0_clip_max=schedule_cfg.get("x0_clip_max", 1.0),
    )

    # 6. 优化器与调度器
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
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=base_lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
        )

    # 7. 混合精度与 EMA
    amp_enabled = bool(cfg["train"].get("amp", False))
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled) if amp_enabled else None
    
    ema_model = model.module if ddp else model
    ema = EMA(ema_model, decay=cfg["train"].get("ema_decay", 0.999))

    # Resume
    start_epoch = 0
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
                
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            if is_main:
                print(f"Resuming from epoch {start_epoch}")

    # 8. 输出目录
    output_cfg = cfg.get("output", {})
    out_dir = Path(output_cfg["dir"])
    fmt = output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S")
    run_ts = time.strftime(fmt) if is_main else ""
    
    if ddp:
        ts_holder = [run_ts]
        dist.broadcast_object_list(ts_holder, src=0)
        run_ts = ts_holder[0]
        
    auto_timestamp = output_cfg.get("auto_timestamp", True)
    if auto_timestamp:
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
        logger = logging.getLogger("train_diffusion_rs")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
        log_message(f"[Residual Shifting Diffusion] 运行时间戳: {run_ts}", logger, console=False, use_tqdm=False)
        log_message(f"输出目录: {out_dir}", logger, console=False, use_tqdm=False)
        log_message(f"World size: {world_size}, Rank: {rank}", logger, console=False, use_tqdm=False)
        log_message(f"配置文件: {args.config}", logger, console=False, use_tqdm=False)

    # 9. 训练主循环
    best_loss = math.inf
    for epoch in range(start_epoch, num_epochs):
        model.train()
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)
            
        iterator = loader
        if tqdm is not None and is_main:
            iterator = tqdm(loader, desc=f"Train {epoch+1}/{num_epochs}", ncols=80)
            
        epoch_loss = 0.0
        steps = 0
        for s1, s2_cloudy, s2_clear, _alpha in iterator:
            s1 = s1.to(device)
            y = s2_cloudy.to(device)   # 有云图像
            x0 = s2_clear.to(device)   # 清晰图像

            # 随机采样 t
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            
            # ===== 关键修改: Residual Shifting 前向过程 =====
            # x_t = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*noise
            x_t = diffusion.q_sample(x0, y, t, noise)

            aux_time_weight = compute_time_weight(
                t, diffusion.timesteps,
                min_weight=cfg["loss"].get("aux_min_weight", 0.1),
                max_weight=cfg["loss"].get("aux_max_weight", 1.0),
            )

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(amp_device, enabled=amp_enabled):
                eps_pred = model(x_t, t, y, s1)
                
                # 主 Loss: 预测噪声
                loss_diff = F.mse_loss(eps_pred, noise)
                
                # ===== 关键修改: predict_x0 需要 y =====
                x0_pred = diffusion.predict_x0_from_eps(x_t, y, t, eps_pred, clip=False)
                x0_pred = x0_pred.clamp(diffusion.x0_clip_min, diffusion.x0_clip_max)
                
                # 按样本时间步加权 (E[w * loss])
                loss_recon_raw = F.l1_loss(x0_pred, x0, reduction="none")
                grad_weight_map = torch.ones_like(x0[:, :1, :, :])
                weight_map = grad_weight_map * aux_time_weight
                loss_recon = weighted_mean(loss_recon_raw, aux_time_weight)
                loss_grad = grad_l1_loss(x0_pred, x0, weight_map)

                recon_weight = cfg["loss"].get("recon_weight", 1.0)
                grad_weight = cfg["loss"].get("grad_weight", 0.5)
                loss = loss_diff + recon_weight * loss_recon + grad_weight * loss_grad

            # NaN/Inf 检测提前到 optimizer.step() 之前，避免污染权重/EMA
            loss_is_finite = torch.isfinite(loss.detach())
            if ddp:
                flag = torch.tensor(float(loss_is_finite.item()), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                loss_is_finite = flag.item() == 1.0
            else:
                loss_is_finite = bool(loss_is_finite.item())

            if not loss_is_finite:
                if is_main:
                    log_message(
                        f"ERROR: NaN/Inf loss detected at epoch {epoch+1} step {steps} BEFORE optimizer.step()",
                        logger, console=True, use_tqdm=True
                    )
                    log_message(f"  loss_diff: {loss_diff.item()}", logger, console=True, use_tqdm=True)
                    log_message(f"  loss_recon: {loss_recon.item()}", logger, console=True, use_tqdm=True)
                    log_message(f"  loss_grad: {loss_grad.item()}", logger, console=True, use_tqdm=True)
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
            if tqdm is not None and is_main and hasattr(iterator, 'set_postfix_str'):
                iterator.set_postfix_str(f"loss={loss.item():.4f}")

        if use_scheduler and scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = epoch_loss / max(1, steps)
        if is_main:
            log_message(
                f"Epoch {epoch+1}/{num_epochs} - train_loss {train_loss:.4f} lr {current_lr:.2e}",
                logger, console=True, use_tqdm=True
            )

        # 10. 评估与保存
        eval_metrics = None
        if True:  # 所有进程都参与评估
            eval_model = model.module if ddp else model
            eval_metrics = evaluate(
                eval_model, eval_loader, diffusion, cfg, device,
                amp_device, amp_enabled, desc="Eval", max_batches=eval_max_batches,
                use_tqdm=True, ema=ema, use_ema=True, ddp=ddp,
            )
            
        if is_main:
            if eval_metrics is not None:
                log_message(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"eval_loss {eval_metrics['loss']:.4f} diff {eval_metrics['diff']:.4f} "
                    f"recon {eval_metrics['recon']:.4f} grad {eval_metrics['grad']:.4f}\n"
                    f"  MAE {eval_metrics.get('mae', 0):.4f} MSE {eval_metrics.get('mse', 0):.4f} "
                    f"RMSE {eval_metrics.get('rmse', 0):.4f} PSNR {eval_metrics.get('psnr', 0):.2f}\n"
                    f"  SSIM {eval_metrics.get('ssim', 0):.4f} MS-SSIM {eval_metrics.get('ms_ssim', 0):.4f} "
                    f"SAM {eval_metrics.get('sam', 0):.2f} ERGAS {eval_metrics.get('ergas', 0):.2f}\n"
                    f"  CC {eval_metrics.get('cc', 0):.4f} UIQI {eval_metrics.get('uiqi', 0):.4f} "
                    f"RASE {eval_metrics.get('rase', 0):.2f}",
                    logger, console=True, use_tqdm=True
                )
            
            save_vis_samples(
                eval_model, diffusion, eval_dataset, cfg, device,
                epoch, base_seed, out_dir, logger, use_tqdm=True,
                ema=ema, use_ema=True,
            )

        if ddp:
            dist.barrier()

        if is_main:
            model_state = model.module.state_dict() if ddp else model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state": model_state,
                "ema_state": ema.shadow,
                "config": cfg,
                "train_loss": train_loss,
                "eval_metrics": eval_metrics,
                "eval_metrics_protocol": METRIC_PROTOCOL,
                "test_metrics": eval_metrics,
                "test_metrics_protocol": METRIC_PROTOCOL,
            }
            torch.save(checkpoint, out_dir / "diffusion_rs_last.pth")
            torch.save({"ema_state": ema.shadow}, out_dir / "diffusion_rs_ema.pth")
            log_message(f"Epoch {epoch+1}/{num_epochs} - checkpoints saved", logger, console=True, use_tqdm=True)
            
            if eval_metrics is not None:
                eval_loss = float(eval_metrics.get("loss", float("nan")))
                if math.isfinite(eval_loss) and eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save(checkpoint, out_dir / "diffusion_rs_best.pth")
                    log_message(
                        f"Epoch {epoch+1}/{num_epochs} - saved diffusion_rs_best.pth (eval_loss {eval_loss:.4f})",
                        logger, console=True, use_tqdm=True
                    )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
