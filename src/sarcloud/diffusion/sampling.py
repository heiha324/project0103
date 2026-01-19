"""扩散模型采样工具函数 (Diffusion Sampling Utilities)。

本模块提供统一的采样接口，避免在训练和推理脚本中重复代码。
支持多种初始化策略和采样方法 (DDPM/DDIM)。
"""

from __future__ import annotations

from typing import Literal

import torch

from sarcloud.diffusion.gaussian import GaussianDiffusion


def sample_batch(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    y: torch.Tensor,
    s: torch.Tensor,
    steps: int,
    schedule_cfg: dict,
    init_method: Literal["noise", "noisy_input", "mixed"] = "noise",
    noise_ratio: float = 1.0,
) -> torch.Tensor:
    """对一个 Batch 进行完整的去噪采样 (Inference)。

    Args:
        model (torch.nn.Module): 条件 U-Net 去噪网络。
        diffusion (GaussianDiffusion): 扩散模型工具类。
        y (torch.Tensor): 条件图像 (有云光学图像)，形状 (B, C, H, W)。
        s (torch.Tensor): SAR 条件图像，形状 (B, C_s, H, W)。
        steps (int): 采样步数。
        schedule_cfg (dict): 采样配置，包含:
            - method (str): "ddpm" 或 "ddim"
            - eta (float): DDIM 随机性系数 (0.0 为确定性)
        init_method (str): 初始化方法:
            - "noise": 纯高斯噪声 (默认，标准扩散)
            - "noisy_input": 从有云图像加噪开始 (图像修复)
            - "mixed": 噪声与条件图像的混合
        noise_ratio (float): mixed 模式下的噪声比例 (0~1)。

    Returns:
        torch.Tensor: 去噪后的图像，形状 (B, C, H, W)。
    """
    model.eval()
    device = y.device

    # 1. 初始化采样起点
    if init_method == "noisy_input":
        # 从有云图像加噪开始，保留更多结构信息
        t_start = torch.full((y.size(0),), diffusion.timesteps - 1, device=device, dtype=torch.long)
        x = diffusion.q_sample(y, t_start)
    elif init_method == "mixed":
        # 混合初始化: 部分噪声 + 部分条件图像
        noise = torch.randn_like(y)
        x = noise_ratio * noise + (1.0 - noise_ratio) * y
    else:
        # 标准纯噪声初始化
        x = torch.randn_like(y)

    # 2. 获取采样时间步序列
    t_seq = diffusion.sample_timesteps(steps)

    # 3. 确定采样方法
    method = schedule_cfg.get("method", "ddim")
    eta = float(schedule_cfg.get("eta", 0.0))
    use_ddim = method != "ddpm" or steps < diffusion.timesteps

    # 4. 逐步去噪
    for step, t in enumerate(t_seq):
        t_batch = torch.full((y.size(0),), t, device=device, dtype=torch.long)
        t_prev = t_seq[step + 1] if step + 1 < len(t_seq) else None
        t_prev_batch = (
            torch.full((y.size(0),), t_prev, device=device, dtype=torch.long)
            if t_prev is not None
            else None
        )

        with torch.no_grad():
            eps = model(x, t_batch, y, s)
            if use_ddim:
                x = diffusion.ddim_step(x, t_batch, t_prev_batch, eps, eta=eta)
            else:
                x = diffusion.p_sample(x, t_batch, eps)

    return x


def sample_with_progress(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    y: torch.Tensor,
    s: torch.Tensor,
    steps: int,
    schedule_cfg: dict,
    init_method: Literal["noise", "noisy_input", "mixed"] = "noise",
    noise_ratio: float = 1.0,
    progress_callback=None,
) -> torch.Tensor:
    """带进度回调的采样函数。

    与 sample_batch 相同，但支持进度回调用于显示采样进度。

    Args:
        progress_callback: 可选的回调函数，接收 (current_step, total_steps) 参数。
        其他参数同 sample_batch。

    Returns:
        torch.Tensor: 去噪后的图像。
    """
    model.eval()
    device = y.device

    # 初始化
    if init_method == "noisy_input":
        t_start = torch.full((y.size(0),), diffusion.timesteps - 1, device=device, dtype=torch.long)
        x = diffusion.q_sample(y, t_start)
    elif init_method == "mixed":
        noise = torch.randn_like(y)
        x = noise_ratio * noise + (1.0 - noise_ratio) * y
    else:
        x = torch.randn_like(y)

    t_seq = diffusion.sample_timesteps(steps)
    method = schedule_cfg.get("method", "ddim")
    eta = float(schedule_cfg.get("eta", 0.0))
    use_ddim = method != "ddpm" or steps < diffusion.timesteps

    for step, t in enumerate(t_seq):
        if progress_callback is not None:
            progress_callback(step, len(t_seq))

        t_batch = torch.full((y.size(0),), t, device=device, dtype=torch.long)
        t_prev = t_seq[step + 1] if step + 1 < len(t_seq) else None
        t_prev_batch = (
            torch.full((y.size(0),), t_prev, device=device, dtype=torch.long)
            if t_prev is not None
            else None
        )

        with torch.no_grad():
            eps = model(x, t_batch, y, s)
            if use_ddim:
                x = diffusion.ddim_step(x, t_batch, t_prev_batch, eps, eta=eta)
            else:
                x = diffusion.p_sample(x, t_batch, eps)

    if progress_callback is not None:
        progress_callback(len(t_seq), len(t_seq))

    return x
