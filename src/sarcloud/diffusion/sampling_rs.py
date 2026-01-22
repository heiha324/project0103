"""残差偏移扩散模型的采样工具函数 (Residual Shifting Sampling Utilities)。

本模块提供 ResidualShiftingDiffusion 的采样接口。
与标准 DDPM 采样的关键区别是：采样起点为 y + noise（有云图像加噪），
而不是纯高斯噪声。
"""

from __future__ import annotations

from typing import Literal

import torch

from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion


def sample_batch_rs(
    model: torch.nn.Module,
    diffusion: ResidualShiftingDiffusion,
    y: torch.Tensor,
    s: torch.Tensor,
    steps: int,
    schedule_cfg: dict,
) -> torch.Tensor:
    """Residual Shifting 扩散的批量采样。

    与标准 DDPM 采样的区别：
    1. 采样起点是 prior_sample(y) = y + noise，而不是纯噪声
    2. 每步去噪时都需要传入条件 y

    Args:
        model (torch.nn.Module): 条件 U-Net 去噪网络。
        diffusion (ResidualShiftingDiffusion): 残差偏移扩散模型实例。
        y (torch.Tensor): 有云光学图像 (B, C, H, W)。
        s (torch.Tensor): SAR 条件图像 (B, C_s, H, W)。
        steps (int): 采样步数。
        schedule_cfg (dict): 采样配置，包含:
            - method (str): "ddpm" 或 "ddim"
            - eta (float): DDIM 随机性系数 (0.0 为确定性)

    Returns:
        torch.Tensor: 去云后的清晰图像 (B, C, H, W)。
    """
    model.eval()
    device = y.device

    # 1. 从先验分布采样 (y + noise)
    x = diffusion.prior_sample(y)

    # 2. 获取采样时间步序列
    t_seq = diffusion.sample_timesteps(steps)

    # 3. 采样参数
    method = schedule_cfg.get("method", "ddim")
    eta = float(schedule_cfg.get("eta", 0.0))
    use_ddim = method == "ddim" or steps < diffusion.timesteps

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
            # 模型预测噪声
            eps = model(x, t_batch, y, s)

            # 单步采样
            if use_ddim:
                x = diffusion.ddim_step(x, y, t_batch, t_prev_batch, eps, eta=eta)
            else:
                x = diffusion.p_sample(x, y, t_batch, eps)

    return x


def sample_with_progress_rs(
    model: torch.nn.Module,
    diffusion: ResidualShiftingDiffusion,
    y: torch.Tensor,
    s: torch.Tensor,
    steps: int,
    schedule_cfg: dict,
    progress_callback=None,
) -> torch.Tensor:
    """带进度回调的 Residual Shifting 采样函数。

    与 sample_batch_rs 相同，但支持进度回调用于显示采样进度。

    Args:
        model (torch.nn.Module): 条件 U-Net 去噪网络。
        diffusion (ResidualShiftingDiffusion): 残差偏移扩散模型实例。
        y (torch.Tensor): 有云光学图像 (B, C, H, W)。
        s (torch.Tensor): SAR 条件图像 (B, C_s, H, W)。
        steps (int): 采样步数。
        schedule_cfg (dict): 采样配置。
        progress_callback: 可选的回调函数，接收 (current_step, total_steps) 参数。

    Returns:
        torch.Tensor: 去云后的清晰图像。
    """
    model.eval()
    device = y.device

    # 1. 从先验分布采样
    x = diffusion.prior_sample(y)

    # 2. 获取采样时间步序列
    t_seq = diffusion.sample_timesteps(steps)

    # 3. 采样参数
    method = schedule_cfg.get("method", "ddim")
    eta = float(schedule_cfg.get("eta", 0.0))
    use_ddim = method == "ddim" or steps < diffusion.timesteps

    # 4. 逐步去噪
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
                x = diffusion.ddim_step(x, y, t_batch, t_prev_batch, eps, eta=eta)
            else:
                x = diffusion.p_sample(x, y, t_batch, eps)

    if progress_callback is not None:
        progress_callback(len(t_seq), len(t_seq))

    return x


def sample_intermediate_rs(
    model: torch.nn.Module,
    diffusion: ResidualShiftingDiffusion,
    y: torch.Tensor,
    s: torch.Tensor,
    steps: int,
    schedule_cfg: dict,
    save_steps: list[int] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """采样并保存中间结果 (用于可视化)。

    Args:
        model (torch.nn.Module): 条件 U-Net 去噪网络。
        diffusion (ResidualShiftingDiffusion): 残差偏移扩散模型实例。
        y (torch.Tensor): 有云光学图像 (B, C, H, W)。
        s (torch.Tensor): SAR 条件图像 (B, C_s, H, W)。
        steps (int): 采样步数。
        schedule_cfg (dict): 采样配置。
        save_steps (list[int], optional): 需要保存的步数列表。

    Returns:
        tuple: (最终结果, 中间结果字典 {step: tensor})
    """
    model.eval()
    device = y.device

    if save_steps is None:
        save_steps = []

    intermediates: dict[int, torch.Tensor] = {}

    # 1. 从先验分布采样
    x = diffusion.prior_sample(y)

    # 2. 获取采样时间步序列
    t_seq = diffusion.sample_timesteps(steps)

    # 3. 采样参数
    method = schedule_cfg.get("method", "ddim")
    eta = float(schedule_cfg.get("eta", 0.0))
    use_ddim = method == "ddim" or steps < diffusion.timesteps

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
                x = diffusion.ddim_step(x, y, t_batch, t_prev_batch, eps, eta=eta)
            else:
                x = diffusion.p_sample(x, y, t_batch, eps)

        # 保存中间结果
        current_step = step + 1
        if current_step in save_steps:
            intermediates[current_step] = x.clone()

    return x, intermediates
