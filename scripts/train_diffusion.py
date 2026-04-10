#!/usr/bin/env python3
"""SAR 引导的条件扩散模型训练脚本 (SAR-guided Conditional Diffusion Training)。

本脚本实现了基于 PyTorch DDP (DistributedDataParallel) 的分布式训练流程，支持：
1. 多卡并行训练与混合精度加速 (AMP)。
2. SAR 和 光学图像的多模态数据加载。
3. 动态学习率调度 (Warmup + Cosine Decay)。
4. 指数移动平均 (EMA) 模型权重更新。
5. 训练过程中的实时采样与可视化。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

# 将 src 目录添加到 Python 路径，确保能导入项目模块
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

# 尝试导入 tqdm 用于显示进度条 (可选)
try:  # pragma: no cover
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# 尝试导入 matplotlib 用于可视化 (无头模式 headless safe)
plt = None
try:
    import matplotlib
    matplotlib.use("Agg")  # 强制使用非交互式后端，防止在服务器上报错
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset, collate_sen12mscr
from sarcloud.diffusion.gaussian import GaussianDiffusion
from sarcloud.diffusion.losses import grad_l1_loss
from sarcloud.diffusion.sampling import sample_batch
from sarcloud.models.cond_unet import ConditionalUNet
from sarcloud.training.ema import EMA
from sarcloud.training.samplers import DistributedEvalSamplerNoPad
from sarcloud.utils.config import load_config
from sarcloud.utils.metrics import (
    mae, mse, rmse, nrmse, psnr, ssim, ms_ssim, sam, ergas, cc, uiqi, rase
)


def set_seed(seed: int) -> None:
    """设置全局随机种子以确保可复现性 (Reproducibility)。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, int]:
    """初始化分布式训练环境 (Distributed Training Environment)。

    尝试从环境变量中读取 DDP 配置 (通常由 torchrun 自动设置)。
    
    Returns:
        tuple: (是否启用DDP, 全局Rank, World Size, 本地Rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # 选择通信后端: GPU 使用 nccl (最快), CPU 使用 gloo
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def log_message(msg: str, logger: logging.Logger | None, console: bool, use_tqdm: bool) -> None:
    """统一日志打印函数，兼容 tqdm 进度条和文件日志。"""
    if logger is not None:
        logger.info(msg)
    if console:
        if tqdm is not None and use_tqdm:
            tqdm.write(msg) # 避免打断进度条显示
        else:
            print(msg, flush=True)


def apply_ema_weights(model: torch.nn.Module, ema: EMA) -> Dict[str, torch.Tensor]:
    """将 EMA (影子权重) 应用到模型上，并备份原始权重。
    
    用于在评估 (Evaluation) 阶段临时使用 EMA 权重，评估完后再恢复。
    """
    backup: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            backup[name] = param.detach().clone()
            param.data.copy_(ema.shadow[name])
    return backup


def restore_weights(model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
    """从备份中恢复模型的原始权重。"""
    for name, param in model.named_parameters():
        if name in backup:
            param.data.copy_(backup[name])


def compute_time_weight(
    t: torch.Tensor,
    timesteps: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> torch.Tensor:
    """计算时间相关的辅助 Loss 权重。
    
    原理：在扩散过程早期 (t 接近 T)，图像主要是噪声，预测 x0 非常困难且不准确。
    此时应降低重建 Loss 的权重，避免模型被错误的梯度误导。
    随着 t 减小 (接近 0)，图像变得清晰，此时应增加权重以微调细节。
    
    Args:
        t: 当前时间步张量 (B,)。
        timesteps: 总步数。
        min_weight: t=T 时的权重 (通常较小，如 0.1)。
        max_weight: t=0 时的权重 (通常为 1.0)。
    
    Returns:
        权重张量 (B, 1, 1, 1)。
    """
    denom = max(timesteps - 1, 1)
    t_ratio = t.float() / denom  # 归一化时间 [0, 1]
    # 线性插值: 当 t=0 时 weight=max; 当 t=T 时 weight=min
    weight = max_weight - (max_weight - min_weight) * t_ratio
    return weight.view(-1, 1, 1, 1)


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算带权平均，支持广播权重。"""
    if loss.shape != weight.shape:
        weight = weight.expand_as(loss)
    denom = weight.sum().clamp_min(eps)
    return (loss * weight).sum() / denom


def build_dataset(data_cfg: Dict):
    """根据配置构建数据集实例。"""
    dataset_type = data_cfg.get("dataset", "npy")
    
    if dataset_type == "sen12mscr_raw":
        # 使用原始 .tif 文件的数据集读取器
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
        
    # 默认使用预处理过的 .npy 数据集读取器 (更快)
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
    """构建 DataLoader。支持 DDP 采样器。"""
    num_workers = int(data_cfg.get("num_workers", 4))
    if sampler is not None:
        shuffle = False # 使用采样器时必须关闭 shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True, # 锁页内存，加速数据传输到 GPU
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    diffusion: GaussianDiffusion,
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
    """执行模型评估循环。
    
    计算测试集上的各主要 Loss 指标。
    """
    backup = None
    if use_ema and ema is not None:
        backup = apply_ema_weights(model, ema)
    
    # 设置固定种子以确保评估结果在不同 Epoch 间可比
    # 在 DDP 模式下，每个 Rank 需要不同的种子以生成不同的噪声
    eval_seed = cfg.get("seed", 42) + 10000
    if ddp:
        eval_seed += dist.get_rank()
        
    eval_rng = torch.Generator(device=device)
    eval_rng.manual_seed(eval_seed)
    
    model.eval()
    totals = {
        "loss": 0.0, "diff": 0.0, "recon": 0.0, "grad": 0.0,
        "mae": 0.0, "mse": 0.0, "rmse": 0.0, "nrmse": 0.0,
        "psnr": 0.0, "ssim": 0.0, "ms_ssim": 0.0, "sam": 0.0,
        "ergas": 0.0, "cc": 0.0, "uiqi": 0.0, "rase": 0.0,
    }
    steps = 0
    sample_count = 0
    
    # 仅在主进程显示进度条
    show_progress = use_tqdm and (not ddp or dist.get_rank() == 0)
    iterator = loader
    if tqdm is not None and show_progress:
        iterator = tqdm(loader, desc=desc, ncols=80, leave=False)
        
    with torch.no_grad():
        for step_idx, (s1, s2_cloudy, s2_clear, _alpha) in enumerate(iterator, start=1):
            if max_batches > 0 and step_idx > max_batches:
                break
            s1 = s1.to(device)
            y = s2_cloudy.to(device)
            x0 = s2_clear.to(device)

            # 采样随机时间步 t
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device, generator=eval_rng)
            noise = torch.randn(x0.shape, device=device, generator=eval_rng)
            # 前向加噪得到 x_t
            x_t = diffusion.q_sample(x0, t, noise)

            with torch.amp.autocast(amp_device, enabled=amp_enabled):
                # 模型预测噪声
                eps_pred = model(x_t, t, y, s1)
                
                # 1. 基础扩散损失 (MSE)
                loss_diff = F.mse_loss(eps_pred, noise)
                
                # 预测 x0 用于计算辅助损失
                x0_pred = diffusion.predict_x0_from_eps(x_t, t, eps_pred, clip=False)
                # 使用 diffusion 对象的 clip 范围
                x0_pred = x0_pred.clamp(diffusion.x0_clip_min, diffusion.x0_clip_max)
                
                # 计算时间权重
                aux_time_weight = compute_time_weight(
                    t,
                    diffusion.timesteps,
                    min_weight=cfg["loss"].get("aux_min_weight", 0.1),
                    max_weight=cfg["loss"].get("aux_max_weight", 1.0),
                )
                # 按样本时间步加权 (E[w * loss])
                loss_recon_raw = F.l1_loss(x0_pred, x0, reduction="none")
                grad_weight_map = torch.ones_like(x0[:, :1, :, :])
                weight_map = grad_weight_map * aux_time_weight
                # 2. 重建损失 (L1)
                loss_recon = weighted_mean(loss_recon_raw, aux_time_weight)
                # 3. 梯度损失 (Edge/Texture)
                loss_grad = grad_l1_loss(x0_pred, x0, weight_map)
                
                # 总损失加权求和
                recon_weight = cfg["loss"].get("recon_weight", 1.0)
                grad_weight = cfg["loss"].get("grad_weight", 0.5)
                loss = loss_diff + recon_weight * loss_recon + grad_weight * loss_grad
                
                # 计算详细指标
                x0_p = x0_pred.detach().float()
                x0_t = x0.detach().float()
                batch_size = int(x0_p.size(0))
                m_mae = mae(x0_p, x0_t)
                m_mse = mse(x0_p, x0_t)
                m_rmse = rmse(x0_p, x0_t)
                m_nrmse = nrmse(x0_p, x0_t)
                m_psnr = psnr(x0_p, x0_t)
                m_ssim = ssim(x0_p, x0_t)
                m_ms_ssim = ms_ssim(x0_p, x0_t)
                m_sam = sam(x0_p, x0_t)
                m_ergas = ergas(x0_p, x0_t)
                m_cc = cc(x0_p, x0_t)
                m_uiqi = uiqi(x0_p, x0_t)
                m_rase = rase(x0_p, x0_t)

            totals["loss"] += float(loss.item())
            totals["diff"] += float(loss_diff.item())
            totals["recon"] += float(loss_recon.item())
            totals["grad"] += float(loss_grad.item())
            totals["mae"] += m_mae * batch_size
            totals["mse"] += m_mse * batch_size
            totals["rmse"] += m_rmse * batch_size
            totals["nrmse"] += m_nrmse * batch_size
            totals["psnr"] += m_psnr * batch_size
            totals["ssim"] += m_ssim * batch_size
            totals["ms_ssim"] += m_ms_ssim * batch_size
            totals["sam"] += m_sam * batch_size
            totals["ergas"] += m_ergas * batch_size
            totals["cc"] += m_cc * batch_size
            totals["uiqi"] += m_uiqi * batch_size
            totals["rase"] += m_rase * batch_size
            
            steps += 1
            sample_count += batch_size
    
    # 评估结束，恢复训练权重
    if backup is not None:
        restore_weights(model, backup)
    
    # DDP 模式下聚合所有 GPU 的评估结果
    if ddp:
        # 将统计量打包为张量
        # keys: loss, diff, recon, grad, mae, mse, rmse, nrmse, psnr, ssim, ms_ssim, sam, ergas, cc, uiqi, rase
        vals = [
            totals["loss"], totals["diff"], totals["recon"], totals["grad"],
            totals["mae"], totals["mse"], totals["rmse"], totals["nrmse"],
            totals["psnr"], totals["ssim"], totals["ms_ssim"], totals["sam"],
            totals["ergas"], totals["cc"], totals["uiqi"], totals["rase"],
            float(steps), float(sample_count)
        ]
        metrics_tensor = torch.tensor(vals, device=device)
        
        # 全局求和
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_steps = metrics_tensor[-2].item()
        total_samples = metrics_tensor[-1].item()
        if total_steps == 0 or total_samples == 0:
             return {k: float("nan") for k in totals}
             
        res = {}
        keys = list(totals.keys()) # keys are in insertion order (Py3.7+)
        # totals insertion order: loss, diff, recon, grad, mae... rase
        # The first 16 elements of metrics_tensor correspond to these keys
        for i, k in enumerate(keys[:4]):
            res[k] = metrics_tensor[i].item() / total_steps
        for i, k in enumerate(keys[4:], start=4):
            res[k] = metrics_tensor[i].item() / total_samples
        return res

    if steps == 0 or sample_count == 0:
        return {k: float("nan") for k in totals}
    out = {k: totals[k] / steps for k in ("loss", "diff", "recon", "grad")}
    out.update({k: totals[k] / sample_count for k in list(totals.keys())[4:]})
    return out


def summarize_array(name: str, arr: np.ndarray) -> str:
    """生成数组的统计摘要字符串 (Min, Max, Mean, Percentiles)。"""
    flat = arr.reshape(-1)
    if flat.size == 0:
        return f"{name} empty"
    flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)
    p1, p99 = np.percentile(flat, [1, 99])
    neg_ratio = float((flat < 0.0).mean() * 100.0)
    over_ratio = float((flat > 1.0).mean() * 100.0)
    return (
        f"{name} min {float(flat.min()):.4f} max {float(flat.max()):.4f} "
        f"mean {float(flat.mean()):.4f} p1 {float(p1):.4f} p99 {float(p99):.4f} "
        f"<0 {neg_ratio:.2f}% >1 {over_ratio:.2f}%"
    )


def _auto_scale_rgb(rgb: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    """自动拉伸 RGB 图像的对比度 (根据百分位数)。"""
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if high_p <= low_p:
        return np.clip(rgb, 0.0, 1.0)
    lo, hi = np.percentile(rgb, [low_p, high_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.clip(rgb, 0.0, 1.0)
    rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0)


def _compute_channel_percentiles(
    rgb: np.ndarray,
    low_p: float,
    high_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """分别计算每个通道的百分位数 (用于独立通道拉伸)。"""
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC rgb array, got shape {rgb.shape}")
    if high_p <= low_p:
        lo = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hi = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return lo, hi
    flat = rgb.reshape(-1, 3)
    lo, hi = np.percentile(flat, [low_p, high_p], axis=0)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        lo = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hi = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return lo.astype(np.float32), hi.astype(np.float32)


def _apply_scale_rgb(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """应用给定的最大最小值进行归一化。"""
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
    scale_params: tuple[np.ndarray, np.ndarray] | None = None,
    per_channel: bool = False,
) -> np.ndarray:
    """将多波段图像转换为 RGB 可视化格式。"""
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {chw.shape}")
    if max(rgb_indices) >= chw.shape[0]:
        raise ValueError(f"RGB indices {rgb_indices} out of range for shape {chw.shape}")
    
    # 提取 RGB 波段并转为 HWC
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
    diffusion: GaussianDiffusion,
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
    # Import metrics inside function to avoid circular import issues if any
    from sarcloud.utils.metrics import psnr, ssim, mae

    """生成并保存训练过程中的可视化样本 (Input/Preds/GT 对比图)。支持多步数。"""
    backup = None
    try:
        if plt is None:
            return
        if len(dataset) == 0:
            return

        # 读取可视化配置
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
        
        # 随机抽取样本
        rng = random.Random(base_seed + epoch)
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
        init_method = sampling_cfg.get("init_method", "noise")
        noise_ratio = float(sampling_cfg.get("noise_ratio", 1.0))
        
        # 多步数采样循环
        preds_map = {}
        with torch.inference_mode():
            for step_cnt in vis_steps:
                if use_tqdm:
                    # Avoid spamming log, just concise
                    pass
                pred = sample_batch(
                    model, diffusion, y, s1,
                    steps=step_cnt,
                    schedule_cfg=sampling_cfg,
                    init_method=init_method,
                    noise_ratio=noise_ratio,
                )
                pred_cpu = pred.cpu()
                preds_map[step_cnt] = pred_cpu.numpy()
                
                # 计算真实复原指标 (针对这 8 张图的平均)
                x0_cpu = x0.cpu()
                pred_clipped = torch.clamp(pred_cpu, 0.0, 1.0)
                
                val_psnr = psnr(pred_clipped, x0_cpu)
                val_ssim = ssim(pred_clipped, x0_cpu)
                val_mae = mae(pred_clipped, x0_cpu)
                
                log_message(
                    f"[Vis] Step {step_cnt}: PSNR {val_psnr:.2f} SSIM {val_ssim:.4f} MAE {val_mae:.4f}",
                    logger, console=True, use_tqdm=use_tqdm
                )

        y_np = y.cpu().numpy()
        x0_np = x0.cpu().numpy()
        
        # 绘图: Cloudy + [Pred_s1, Pred_s2...] + Clear
        cols = 2 + len(vis_steps)
        fig, axes = plt.subplots(sample_count, cols, figsize=(cols * 3, sample_count * 3), squeeze=False)
        
        for row, idx in enumerate(indices):
            # Scale params calculation
            scale_params = None
            if auto_scale and auto_scale_ref != "self":
                if auto_scale_ref == "cloudy":
                    ref_rgb = to_rgb(y_np[row], rgb_indices)
                else:
                    ref_rgb = to_rgb(x0_np[row], rgb_indices)
                
                if per_channel:
                    scale_params = _compute_channel_percentiles(
                        ref_rgb, scale_percentiles[0], scale_percentiles[1]
                    )
                else:
                    lo, hi = np.percentile(ref_rgb, [scale_percentiles[0], scale_percentiles[1]])
                    scale_params = (np.array([lo, lo, lo]), np.array([hi, hi, hi]))

            # Cloudy
            cloudy_rgb = to_rgb(
                y_np[row], rgb_indices, 
                auto_scale=auto_scale and auto_scale_ref=="self", 
                scale_params=scale_params, per_channel=per_channel
            )
            axes[row, 0].imshow(cloudy_rgb)
            axes[row, 0].set_title(f"Cloudy ({idx})", fontsize=8)
            axes[row, 0].axis("off")
            
            # Preds
            for i, step_cnt in enumerate(vis_steps):
                pred_np = preds_map[step_cnt][row]
                
                # Check out of range logic
                pred_out_ratio = float(((pred_np < 0.0) | (pred_np > 1.0)).mean() * 100.0)
                pred_use_self = auto_scale and (auto_scale_ref == "self" or pred_out_ratio > 5.0)
                
                pred_rgb = to_rgb(
                    pred_np, rgb_indices,
                    auto_scale=pred_use_self,
                    scale_params=None if pred_use_self else scale_params,
                    per_channel=per_channel
                )
                axes[row, i + 1].imshow(pred_rgb)
                axes[row, i + 1].set_title(f"Step {step_cnt}", fontsize=8)
                axes[row, i + 1].axis("off")
                
            # Clear
            clear_rgb = to_rgb(
                x0_np[row], rgb_indices,
                auto_scale=auto_scale and auto_scale_ref=="self",
                scale_params=scale_params, per_channel=per_channel
            )
            axes[row, -1].imshow(clear_rgb)
            axes[row, -1].set_title("Clear", fontsize=8)
            axes[row, -1].axis("off")

        plt.tight_layout()
        vis_dir = out_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        epoch_path = vis_dir / f"diffusion_vis_epoch_{epoch+1:03d}.png"
        fig.savefig(epoch_path, dpi=150)
        # Also save latest
        fig.savefig(vis_dir / "diffusion_vis.png", dpi=150)
        plt.close(fig)
        log_message(f"Saved diffusion visualization to {epoch_path}", logger, console=True, use_tqdm=use_tqdm)
    except Exception as exc:  # pragma: no cover
        log_message(f"Diffusion visualization failed: {exc}", logger, console=True, use_tqdm=use_tqdm)
        import traceback
        traceback.print_exc()
    finally:
        if backup is not None:
            restore_weights(model, backup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # 1. 初始化
    cfg = load_config(args.config)
    ddp, rank, world_size, local_rank = init_distributed()
    is_main = rank == 0 # 是否为主进程 (负责打印日志和保存模型)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    base_seed = cfg.get("seed", 42)
    # 为每个进程设置不同的种子，防止数据重复
    set_seed(base_seed + rank)

    # 2. 构建数据集
    data_cfg = cfg["sen12ms"]
    dataset = build_dataset(data_cfg)

    # 3. 构建 DataLoader
    sampler = None
    if ddp:
        # 分布式采样器，确保每个 GPU 拿到不同的数据切片
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=base_seed)
    
    loader = build_loader(
        dataset,
        data_cfg,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=ddp,  # DDP 训练时必须丢弃不完整的最后一个 batch，否则会导致通信死锁
        collate_fn=collate_sen12mscr,
    )

    # 评估数据集
    eval_cfg = cfg.get("test") or cfg.get("val") or data_cfg
    eval_dataset = build_dataset(eval_cfg)
    
    eval_sampler = None
    if ddp:
        # 评估时通常 shuffle=False
        eval_sampler = DistributedEvalSamplerNoPad(eval_dataset, num_replicas=world_size, rank=rank)
        
    eval_loader = build_loader(
        eval_dataset,
        eval_cfg,
        batch_size=eval_cfg.get("batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        sampler=eval_sampler,
        drop_last=False,
        collate_fn=collate_sen12mscr,
    )
    
    # 限制评估批次 (调试用)
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
        # 模型中存在未使用的参数 (可能是某些条件分支未触发)，必须设为 True
        model = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # 额外内存优化
        )

    # 5. 初始化扩散工具
    schedule_cfg = cfg["schedule"]
    diffusion = GaussianDiffusion(
        timesteps=schedule_cfg["timesteps"],
        schedule_type=schedule_cfg.get("type", "cosine"),
        beta_start=schedule_cfg.get("beta_start", 1e-4),
        beta_end=schedule_cfg.get("beta_end", 2e-2),
        device=device,
        x0_clip_min=schedule_cfg.get("x0_clip_min", -1.0),
        x0_clip_max=schedule_cfg.get("x0_clip_max", 2.0),
    )

    # 6. 优化器与调度器
    base_lr = cfg["train"].get("lr", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    # 学习率调度器：Warmup + Cosine Annealing
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

    # 7. 混合精度与 EMA
    amp_enabled = bool(cfg["train"].get("amp", False))
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled) if amp_enabled else None
    
    # EMA 总是基于原始模型 (model.module) 而非 DDP 包装器
    ema_model = model.module if ddp else model
    ema = EMA(ema_model, decay=cfg["train"].get("ema_decay", 0.999))

    # --- Resume Logic ---
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank} if ddp else device
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        
        # Load model weights
        if "model_state" in checkpoint:
            # Handle DDP state dict (strip 'module.' prefix if needed, or handle by model wrapper)
            # Since we wrapped model in DDP, loading state dict with 'module.' prefix is fine if DDP.
            # If checkpoint was saved as model.module.state_dict() (which it is in line 952), then it has no 'module.' prefix usually?
            # Line 952: model_state = model.module.state_dict() if ddp else model.state_dict()
            # So saved state dict does NOT have 'module.' prefix.
            # But currently `model` IS a DDP wrapper (if ddp=True).
            # So we should load into model.module.
            load_target = model.module if ddp else model
            missing, unexpected = load_target.load_state_dict(checkpoint["model_state"], strict=False)
            if is_main:
                if missing:
                    print(f"WARNING: Resume checkpoint missing keys: {missing}")
                if unexpected:
                    print(f"WARNING: Resume checkpoint unexpected keys: {unexpected}")
                print(f"Loaded model weights from {ckpt_path} (strict=False)")
        
        # Load EMA
        if "ema_state" in checkpoint:
            ema.shadow = checkpoint["ema_state"]
            if is_main:
                print(f"Loaded EMA state from {ckpt_path}")
                
        # Load epoch
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            if is_main:
                print(f"Resuming from epoch {start_epoch}")

    # 8. 日志目录设置
    output_cfg = cfg.get("output", {})
    out_dir = Path(output_cfg["dir"])
    fmt = output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S")
    run_ts = time.strftime(fmt) if is_main else ""
    
    # 广播时间戳，确保所有进程使用相同的输出目录
    if ddp:
        ts_holder = [run_ts]
        dist.broadcast_object_list(ts_holder, src=0)
        run_ts = ts_holder[0]
        
    auto_timestamp = output_cfg.get("auto_timestamp")
    if auto_timestamp is None:
        auto_timestamp = True
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
        logger = logging.getLogger("train_diffusion")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
        log_message(f"运行时间戳: {run_ts}", logger, console=False, use_tqdm=False)
        log_message(f"输出目录: {out_dir}", logger, console=False, use_tqdm=False)
        log_message(f"World size: {world_size}, Rank: {rank}", logger, console=False, use_tqdm=False)
        log_message(f"配置文件: {args.config}", logger, console=False, use_tqdm=False)

    # 9. 训练主循环
    best_loss = math.inf
    for epoch in range(start_epoch, cfg["train"]["num_epochs"]):
        model.train()
        if ddp and sampler is not None:
            sampler.set_epoch(epoch) # 这一步至关重要，否则每个 Epoch 数据顺序一样
            
        iterator = loader
        if tqdm is not None and is_main:
            iterator = tqdm(loader, desc=f"Train {epoch+1}/{cfg['train']['num_epochs']}", ncols=80)
            
        epoch_loss = 0.0
        steps = 0
        for s1, s2_cloudy, s2_clear, _alpha in iterator:
            s1 = s1.to(device)
            y = s2_cloudy.to(device)
            x0 = s2_clear.to(device)

            # 随机采样 t
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            # 前向过程得到 x_t
            x_t = diffusion.q_sample(x0, t, noise)

            # 计算辅助 Loss 的时间权重
            aux_time_weight = compute_time_weight(
                t,
                diffusion.timesteps,
                min_weight=cfg["loss"].get("aux_min_weight", 0.1),
                max_weight=cfg["loss"].get("aux_max_weight", 1.0),
            )

            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度上下文
            with torch.amp.autocast(amp_device, enabled=amp_enabled):
                eps_pred = model(x_t, t, y, s1)
                
                # 主 Loss: 预测噪声
                loss_diff = F.mse_loss(eps_pred, noise)
                
                # 辅助 Loss: 预测原图 + 梯度
                x0_pred = diffusion.predict_x0_from_eps(x_t, t, eps_pred, clip=False)
                # 使用 diffusion 对象的 clip 范围进行数值保护
                x0_pred = x0_pred.clamp(diffusion.x0_clip_min, diffusion.x0_clip_max)
                
                # 按样本时间步加权 (E[w * loss])
                loss_recon_raw = F.l1_loss(x0_pred, x0, reduction="none")
                grad_weight_map = torch.ones_like(x0[:, :1, :, :])
                weight_map = grad_weight_map * aux_time_weight
                loss_recon = weighted_mean(loss_recon_raw, aux_time_weight)
                loss_grad = grad_l1_loss(x0_pred, x0, weight_map)

                recon_weight = cfg["loss"].get("recon_weight", cfg["loss"].get("cloud_weight", 1.0))
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
                        f"ERROR: 发现 Loss 为 NaN/Inf (Epoch {epoch+1} Step {steps}) - BEFORE optimizer.step()",
                        logger, console=True, use_tqdm=True
                    )
                    log_message(f"  loss_diff: {loss_diff.item()}", logger, console=True, use_tqdm=True)
                    log_message(f"  loss_recon: {loss_recon.item()}", logger, console=True, use_tqdm=True)
                    log_message(f"  loss_grad: {loss_grad.item()}", logger, console=True, use_tqdm=True)
                raise RuntimeError(f"NaN/Inf loss at epoch {epoch+1} step {steps}")

            # 反向传播与优化
            if amp_enabled:
                assert scaler is not None
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # 梯度裁剪
            if cfg["train"].get("grad_clip", 0.0) > 0:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # 更新 EMA 权重
            ema.update(ema_model)

            epoch_loss += float(loss.item())
            steps += 1
            if tqdm is not None and is_main:
                iterator.set_postfix_str(f"loss={loss.item():.4f}")

        # 更新学习率
        if use_scheduler and scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = epoch_loss / max(1, steps)
        if is_main:
            log_message(
                f"Epoch {epoch+1}/{cfg['train']['num_epochs']} - train_loss {train_loss:.4f} lr {current_lr:.2e}",
                logger, console=True, use_tqdm=True
            )

        # 10. 评估与保存
        eval_metrics = None
        should_eval = True # 在 DDP 下所有进程都要参与评估 (为了 all_reduce)
        
        if should_eval:
            eval_model = model.module if ddp else model
            eval_metrics = evaluate(
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
                use_ema=True, # 评估时使用 EMA 权重，效果更稳
                ddp=ddp,
            )
            
        if is_main:
            if eval_metrics is not None:
                log_message(
                    "Epoch "
                    f"{epoch+1}/{cfg['train']['num_epochs']} - "
                    f"test_loss {eval_metrics['loss']:.4f} diff {eval_metrics['diff']:.4f} "
                    f"recon {eval_metrics['recon']:.4f} grad {eval_metrics['grad']:.4f}\n"
                    f"  MAE {eval_metrics.get('mae', 0):.4f} MSE {eval_metrics.get('mse', 0):.4f} "
                    f"RMSE {eval_metrics.get('rmse', 0):.4f} PSNR {eval_metrics.get('psnr', 0):.2f}\n"
                    f"  SSIM {eval_metrics.get('ssim', 0):.4f} MS-SSIM {eval_metrics.get('ms_ssim', 0):.4f} "
                    f"SAM {eval_metrics.get('sam', 0):.2f} ERGAS {eval_metrics.get('ergas', 0):.2f}\n"
                    f"  CC {eval_metrics.get('cc', 0):.4f} UIQI {eval_metrics.get('uiqi', 0):.4f} "
                    f"RASE {eval_metrics.get('rase', 0):.2f}",
                    logger, console=True, use_tqdm=True
                )
            
            # 只有主进程保存图片
            save_vis_samples(
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
            dist.barrier() # 等待所有进程完成

        if is_main:
            model_state = model.module.state_dict() if ddp else model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state": model_state,
                "ema_state": ema.shadow,
                "config": cfg,
                "train_loss": train_loss,
                "test_metrics": eval_metrics,
            }
            torch.save(checkpoint, out_dir / "diffusion_last.pth")
            torch.save({"ema_state": ema.shadow}, out_dir / "diffusion_ema.pth")
            log_message(
                f"Epoch {epoch+1}/{cfg['train']['num_epochs']} - checkpoints saved",
                logger, console=True, use_tqdm=True
            )
            
            # 保存最佳模型
            if eval_metrics is not None:
                test_loss = float(eval_metrics.get("loss", float("nan")))
                if math.isfinite(test_loss) and test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(checkpoint, out_dir / "diffusion_best.pth")
                    log_message(
                        f"Epoch {epoch+1}/{cfg['train']['num_epochs']} - saved diffusion_best.pth "
                        f"(test_loss {test_loss:.4f})",
                        logger, console=True, use_tqdm=True
                    )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
