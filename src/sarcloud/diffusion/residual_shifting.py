"""残差偏移扩散模型 (Residual Shifting Diffusion)。

与标准 DDPM 不同，这个扩散过程建立在"有云→清晰"的转换上。
前向过程：x_t = η_t*(y-x0) + x0 + sqrt(η_t)*κ*ε = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*ε
反向过程：从 y+noise 逐步还原到 x0

核心思想：
- t=0 时，η_0≈0，x_0 = x0（清晰图像）
- t=T 时，η_T≈1，x_T ≈ y + noise（有云图像+噪声）
- 模型学习从有云图像向清晰图像的转换，而不是从纯噪声生成

参考: EDM-CR (Efficient Diffusion Model for Cloud Removal)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from sarcloud.diffusion.timesteps import make_time_sequence


def make_eta_schedule(
    timesteps: int,
    schedule_type: str = "exponential",
    min_eta: float = 0.001,
    max_eta: float = 0.99,
    power: float = 2.0,
) -> np.ndarray:
    """生成 η 调度表 (sqrt_eta)。

    η_t 控制 x_t 从 x0 向 y 偏移的程度：
    - η_0 ≈ 0: x_0 = x0 (清晰图)
    - η_T ≈ 1: x_T ≈ y (有云图)

    Args:
        timesteps (int): 总扩散步数。
        schedule_type (str): 调度类型 ("exponential" 或 "linear")。
        min_eta (float): η 的最小值 (对应 t=0)。
        max_eta (float): η 的最大值 (对应 t=T)。
        power (float): 指数调度的幂次 (仅用于 exponential)。

    Returns:
        np.ndarray: sqrt(η) 的调度表，形状为 (timesteps,)。
    """
    if schedule_type == "exponential":
        # 指数调度 (EDM-CR 默认)
        # 使模型在早期（接近 y）有更多的调整空间
        # 注意: 这里需要在 sqrt(η) 空间做指数调度
        sqrt_etas_start = np.sqrt(min_eta)
        sqrt_etas_end = np.sqrt(max_eta)
        increaser = np.exp(
            np.log(sqrt_etas_end / sqrt_etas_start) / (timesteps - 1)
        )
        base = np.ones(timesteps) * increaser
        power_timestep = np.linspace(0, 1, timesteps, endpoint=True) ** power
        power_timestep *= (timesteps - 1)
        sqrt_etas = np.power(base, power_timestep) * sqrt_etas_start
    elif schedule_type == "linear":
        # 线性调度
        sqrt_etas = np.linspace(np.sqrt(min_eta), np.sqrt(max_eta), timesteps)
    elif schedule_type == "cosine":
        # 余弦调度 (类似 Improved DDPM)
        steps = np.linspace(0, 1, timesteps)
        # 从 min_eta 平滑过渡到 max_eta
        sqrt_etas = np.sqrt(min_eta) + (np.sqrt(max_eta) - np.sqrt(min_eta)) * (
            1 - np.cos(steps * np.pi / 2)
        )
    else:
        raise ValueError(f"未知的调度类型 (Unknown schedule type): {schedule_type}")

    return sqrt_etas.astype(np.float64)


def extract_np(arr: np.ndarray, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """从 numpy 数组中按索引提取值并广播到目标形状。

    Args:
        arr (np.ndarray): 数据源数组 (1-D)。
        t (torch.Tensor): 时间步索引 (Batch Size,)。
        shape (torch.Size): 目标形状 (B, C, H, W)。

    Returns:
        torch.Tensor: 提取并广播后的张量，形状为 (B, 1, 1, 1)。
    """
    res = torch.from_numpy(arr).to(device=t.device, dtype=torch.float32)
    res = res.gather(0, t)
    return res.view(-1, *([1] * (len(shape) - 1)))


class ResidualShiftingDiffusion:
    """残差偏移扩散模型 (Residual Shifting Diffusion)。

    核心公式:
    - 前向: x_t = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*ε
    - 后验均值: x_{t-1} = (η_{t-1}/η_t)*x_t + (α_t/η_t)*x0
      其中 α_t = η_t - η_{t-1}

    与标准 DDPM 的关键区别：
    1. 采样起点是 y + noise（有云图像），而不是纯噪声
    2. 扩散过程直接建模从有云到清晰的转换
    3. 条件 y 参与扩散过程本身，而不仅是模型输入
    """

    def __init__(
        self,
        timesteps: int = 1000,
        kappa: float = 1.0,
        schedule_type: str = "exponential",
        min_eta: float = 0.001,
        max_eta: float = 0.99,
        power: float = 2.0,
        x0_clip_min: float = 0.0,
        x0_clip_max: float = 1.0,
    ) -> None:
        """初始化残差偏移扩散模型。

        Args:
            timesteps (int): 总扩散步数 T。
            kappa (float): 噪声强度系数，控制每步加噪的幅度。
            schedule_type (str): η 调度类型 ("exponential", "linear", "cosine")。
            min_eta (float): η 的最小值 (对应 t=0 附近)。
            max_eta (float): η 的最大值 (对应 t=T 附近)。
            power (float): 指数调度的幂次参数。
            x0_clip_min (float): 预测 x0 的截断下限。
            x0_clip_max (float): 预测 x0 的截断上限。
        """
        self.timesteps = timesteps
        self.kappa = kappa
        self.x0_clip_min = x0_clip_min
        self.x0_clip_max = x0_clip_max

        # 生成 η 调度表
        self.sqrt_etas = make_eta_schedule(
            timesteps, schedule_type, min_eta, max_eta, power
        )
        self.etas = self.sqrt_etas**2

        # 预计算系数
        # η_{t-1}，在 t=0 时为 0
        self.etas_prev = np.append(0.0, self.etas[:-1])
        # α_t = η_t - η_{t-1}，表示每步的 η 增量
        self.alpha = self.etas - self.etas_prev

        # 后验方差: var = κ² * η_{t-1} / η_t * α_t
        # 这是从 q(x_{t-1} | x_t, x0) 推导出的方差
        with np.errstate(divide="ignore", invalid="ignore"):
            self.posterior_variance = (
                self.kappa**2 * self.etas_prev / self.etas * self.alpha
            )
        # 边界处理：t=0 时使用 t=1 的值
        self.posterior_variance = np.nan_to_num(
            self.posterior_variance, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.posterior_variance[0] = self.posterior_variance[1] if timesteps > 1 else 0.0
        self.posterior_log_variance = np.log(np.clip(self.posterior_variance, 1e-20, None))

        # 后验均值系数
        # coef1 = η_{t-1} / η_t (x_t 的系数)
        # coef2 = α_t / η_t (x0 的系数)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.posterior_mean_coef1 = self.etas_prev / self.etas
            self.posterior_mean_coef2 = self.alpha / self.etas
        # 边界处理
        self.posterior_mean_coef1 = np.nan_to_num(
            self.posterior_mean_coef1, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.posterior_mean_coef2 = np.nan_to_num(
            self.posterior_mean_coef2, nan=1.0, posinf=1.0, neginf=1.0
        )
        # t=0 时，x_{-1} = x0，所以 coef1=0, coef2=1
        self.posterior_mean_coef1[0] = 0.0
        self.posterior_mean_coef2[0] = 1.0

    def q_sample(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向扩散过程 (Forward Process)。

        公式: x_t = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*noise

        物理意义：
        - (1-η_t)*x0：清晰图像的贡献，随 t 增大而减少
        - η_t*y：有云图像的贡献，随 t 增大而增加
        - sqrt(η_t)*κ*noise：噪声项

        Args:
            x0 (torch.Tensor): 原始清晰图像 (B, C, H, W)。
            y (torch.Tensor): 有云图像 (B, C, H, W)。
            t (torch.Tensor): 时间步索引 (B,)。
            noise (torch.Tensor, optional): 高斯噪声，如果不提供则随机生成。

        Returns:
            torch.Tensor: 加噪后的图像 x_t。
        """
        if noise is None:
            noise = torch.randn_like(x0)

        eta_t = extract_np(self.etas, t, x0.shape)
        sqrt_eta_t = extract_np(self.sqrt_etas, t, x0.shape)

        # x_t = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*noise
        return (1.0 - eta_t) * x0 + eta_t * y + sqrt_eta_t * self.kappa * noise

    def prior_sample(
        self,
        y: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """从先验分布采样 (采样起点)。

        公式: x_T = y + κ*sqrt(η_T)*noise

        这是采样的起点，与标准 DDPM 从纯噪声开始不同，
        Residual Shifting 从有云图像加少量噪声开始。

        Args:
            y (torch.Tensor): 有云图像 (B, C, H, W)。
            noise (torch.Tensor, optional): 高斯噪声。

        Returns:
            torch.Tensor: 采样起点 x_T。
        """
        if noise is None:
            noise = torch.randn_like(y)

        t = torch.full((y.shape[0],), self.timesteps - 1, device=y.device, dtype=torch.long)
        sqrt_eta_T = extract_np(self.sqrt_etas, t, y.shape)

        return y + self.kappa * sqrt_eta_T * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        clip: bool = True,
    ) -> torch.Tensor:
        """根据预测的噪声还原 x0。

        从前向公式反推：
        x_t = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*ε
        => x0 = (x_t - η_t*y - sqrt(η_t)*κ*ε) / (1-η_t)

        Args:
            x_t (torch.Tensor): 当前时刻的加噪图像。
            y (torch.Tensor): 有云图像（条件）。
            t (torch.Tensor): 时间步。
            eps (torch.Tensor): 模型预测的噪声。
            clip (bool): 是否对 x0 进行数值截断。

        Returns:
            torch.Tensor: 预测的清晰图像 x0。
        """
        eta_t = extract_np(self.etas, t, x_t.shape)
        sqrt_eta_t = extract_np(self.sqrt_etas, t, x_t.shape)

        # 避免除零 (当 η_t ≈ 1 时，1-η_t ≈ 0)
        denom = (1.0 - eta_t).clamp_min(1e-5)
        x0_pred = (x_t - eta_t * y - sqrt_eta_t * self.kappa * eps) / denom

        if clip:
            x0_pred = x0_pred.clamp(self.x0_clip_min, self.x0_clip_max)

        return x0_pred

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算后验分布 q(x_{t-1} | x_t, x0, y) 的均值和方差。

        后验均值推导 (从前向公式):
        x_t = (1-η_t)*x0 + η_t*y + sqrt(η_t)*κ*ε
        x_{t-1} = (1-η_{t-1})*x0 + η_{t-1}*y + sqrt(η_{t-1})*κ*ε'

        后验均值: μ = coef_xt*x_t + coef_x0*x0 + coef_y*y

        Args:
            x0 (torch.Tensor): 原始/预测的清晰图像。
            x_t (torch.Tensor): 当前时刻图像。
            y (torch.Tensor): 有云图像（条件）。
            t (torch.Tensor): 时间步。

        Returns:
            Tuple: (后验均值, 后验方差, 后验对数方差)
        """
        eta_t = extract_np(self.etas, t, x_t.shape)
        eta_prev = extract_np(self.etas_prev, t, x_t.shape)
        sqrt_eta_t = extract_np(self.sqrt_etas, t, x_t.shape)
        sqrt_eta_prev = extract_np(np.sqrt(self.etas_prev), t, x_t.shape)

        # 使用方向向量公式: r = sqrt(η_{t-1}/η_t)
        # 避免除以零
        r = sqrt_eta_prev / sqrt_eta_t.clamp_min(1e-8)

        coef_x0 = 1.0 - eta_prev - r * (1.0 - eta_t)
        coef_y = eta_prev - r * eta_t

        posterior_mean = r * x_t + coef_x0 * x0 + coef_y * y
        posterior_variance = extract_np(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract_np(self.posterior_log_variance, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM 风格的单步采样 (反向过程)。

        给定当前图像 x_t 和预测噪声 eps，采样 x_{t-1}。

        Args:
            x_t (torch.Tensor): 当前时刻图像。
            y (torch.Tensor): 有云图像（条件）。
            t (torch.Tensor): 当前时间步。
            eps (torch.Tensor): 模型预测的噪声。

        Returns:
            torch.Tensor: 上一时刻的图像 x_{t-1}。
        """
        # 1. 预测 x0
        x0_pred = self.predict_x0_from_eps(x_t, y, t, eps, clip=True)

        # 2. 如果是最后一步 (t=0)，直接返回 x0
        if (t == 0).all():
            return x0_pred

        # 3. 计算后验均值和方差
        mean, variance, _ = self.q_posterior_mean_variance(x0_pred, x_t, y, t)

        # 4. 添加噪声 (t > 0 时)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return mean + nonzero_mask * torch.sqrt(variance) * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor | None,
        eps: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM 风格的单步采样 (支持跳步加速)。

        DDIM 允许使用更少的步数进行采样，通过跳过中间时间步。

        Args:
            x_t (torch.Tensor): 当前时刻图像。
            y (torch.Tensor): 有云图像（条件）。
            t (torch.Tensor): 当前时间步。
            t_prev (torch.Tensor | None): 目标时间步 (t_prev < t)。
            eps (torch.Tensor): 模型预测的噪声。
            eta (float): 随机性系数 (0.0 为确定性，1.0 接近 DDPM)。

        Returns:
            torch.Tensor: 目标时刻的图像 x_{t_prev}。
        """
        # 1. 预测 x0
        x0_pred = self.predict_x0_from_eps(x_t, y, t, eps, clip=True)

        # 2. 如果没有下一步，直接返回 x0
        if t_prev is None:
            return x0_pred

        # 3. 获取当前和目标时刻的 η 值
        eta_t = extract_np(self.etas, t, x_t.shape)
        eta_prev = extract_np(self.etas, t_prev, x_t.shape)
        sqrt_eta_t = extract_np(self.sqrt_etas, t, x_t.shape)
        sqrt_eta_prev = extract_np(self.sqrt_etas, t_prev, x_t.shape)

        if eta == 0.0:
            # 确定性采样 (DDIM)
            # 使用方向向量保持轨迹一致性
            # r = sqrt(η_{t_prev} / η_t)
            r = sqrt_eta_prev / sqrt_eta_t.clamp_min(1e-8)

            # x_{t_prev} = r*x_t + coef_x0*x0_pred + coef_y*y
            coef_x0 = 1.0 - eta_prev - r * (1.0 - eta_t)
            coef_y = eta_prev - r * eta_t

            return r * x_t + coef_x0 * x0_pred + coef_y * y
        else:
            # 带随机性的采样
            # 计算 sigma (类似 DDIM 的公式)
            alpha_t = eta_t - extract_np(self.etas_prev, t, x_t.shape)
            sigma_sq = (
                eta**2
                * self.kappa**2
                * eta_prev
                / eta_t.clamp_min(1e-8)
                * alpha_t
            )
            sigma = torch.sqrt(sigma_sq.clamp_min(0.0))

            noise = torch.randn_like(x_t)

            # 使用方向向量公式
            r = sqrt_eta_prev / sqrt_eta_t.clamp_min(1e-8)

            # 调整方向系数以保持方差平衡
            # dir_scale 需要保证方差正确
            # 这里简化处理：我们假设 residual shifting 的主要路径遵循 ODE
            # 而随机性作为附加项。为了严谨，我们应该重新推导
            # 但常用的 trick 是保持确定性部分的主导地位
            
            # 更稳健的做法是沿用确定性路径并添加噪声，
            # 这里的 sigma 已经很小了（通常 eta < 1.0）
            # 我们使用与 EDM 类似的修正方式
            
            # 重新计算方向系数以补偿 sigma 的影响
            dir_scale = torch.sqrt(
                (eta_prev * self.kappa**2 - sigma**2).clamp_min(0.0)
            ) / (self.kappa * sqrt_eta_prev.clamp_min(1e-8) + 1e-8)
             
            # 如果 sqrt_eta_prev 接近 0 (t_prev=0)，dir_scale 会变得不稳定
            # 当 t_prev=0 时，我们只应该有 x0_pred，不应该有噪声项（理论上）
            # 但为了统一，我们使用 r 作为主要缩放
            
            # 简化方案：直接在确定性结果上加噪声
            # x_prev_det = r * x_t + coef_x0 * x0_pred + coef_y * y
            # return x_prev_det + sigma * noise
            
            # 但这会改变方差。严格的公式：
            # dir_coef = sqrt(η_{t-1}*κ² - σ²) / (sqrt(η_t)*κ)
            # 这里的 dir_coef 对应上面的 r * correction
            
            # 让我们使用上面推导的严格公式
            # r_corrected = dir_scale (如果不除以 sqrt_eta_prev) * sqrt_eta_prev / sqrt_eta_t
            
            # 让我们回退到最清晰的实现：
            # 1. 计算确定性部分的方向向量 eps_t = (x_t - (1-η)*x0 - η*y) / (sqrt(η)*κ)
            #    注意：这里的 eps_t 就是我们预测的 eps (如果我们相信模型预测)
            #    或者我们可以用 x0_pred 反推 (保持自洽性)
            #    反推：eps_implied = (x_t - (1-eta_t)*x0_pred - eta_t*y) / (sqrt_eta_t * kappa)
            
            # 2. x_{t-1} = (1-eta_prev)*x0_pred + eta_prev*y + sqrt(eta_prev * kappa**2 - sigma**2) * eps_implied + sigma * noise
            
            eps_implied = (x_t - (1.0 - eta_t) * x0_pred - eta_t * y) / (sqrt_eta_t * self.kappa).clamp_min(1e-8)
            
            coef_eps = torch.sqrt((eta_prev * self.kappa**2 - sigma**2).clamp_min(0.0))
            
            return (
                (1.0 - eta_prev) * x0_pred
                + eta_prev * y
                + coef_eps * eps_implied
                + sigma * noise
            )

    def sample_timesteps(self, steps: int) -> List[int]:
        """生成采样所需的时间步序列。

        用于 DDIM 跳步采样。

        Args:
            steps (int): 采样步数。

        Returns:
            List[int]: 时间步列表，从 T-1 递减到 0。
        """
        return make_time_sequence(self.timesteps - 1, 0, steps)
