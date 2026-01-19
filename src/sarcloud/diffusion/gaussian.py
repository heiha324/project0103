"""高斯扩散实用工具 (Gaussian Diffusion Utilities)。

本模块实现了高斯扩散模型 (DDPM/DDIM) 的核心数学逻辑，包括：
1. 噪声调度表 (Beta Schedule) 的生成 (线性或余弦调度)。
2. 扩散模型的参数预计算 (alphas, betas 等)。
3. 前向扩散过程 (Forward Process, q_sample): 将图像加噪。
4. 反向去噪过程 (Reverse Process, p_sample/ddim_step): 从噪声中恢复图像。

参考文献:
- DDPM: Ho et al., "Denoising Diffusion Probabilistic Models", 2020.
- DDIM: Song et al., "Denoising Diffusion Implicit Models", 2020.
- Improved DDPM: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", 2021.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F


def make_beta_schedule(
    timesteps: int,
    schedule_type: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    """生成噪声方差表 (Beta Schedule)。

    Beta 表决定了每一步加噪的强度。

    Args:
        timesteps (int): 总的扩散步数 T (例如 1000)。
        schedule_type (str): 调度类型，支持 "linear" (线性) 或 "cosine" (余弦)。
        beta_start (float): 线性调度的起始 Beta 值 (通常很小，如 1e-4)。
        beta_end (float): 线性调度的结束 Beta 值 (通常为 0.02)。

    Returns:
        torch.Tensor: 形状为 (timesteps,) 的 Beta 张量，值域通常在 [0, 1] 之间。
    """
    if schedule_type == "linear":
        # 线性插值生成 Beta，这是最原始的 DDPM 做法
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    
    if schedule_type == "cosine":
        # 余弦调度 (Improved DDPM, Nichol & Dhariwal, 2021)
        # 这种调度在 t 较小时噪声增加得更慢，有助于保留更多信息，
        # 防止图像在扩散早期就完全变成噪声。
        steps = timesteps + 1
        s = 0.008
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        
        # 计算 alpha_bar (累积乘积)
        # f(t) = cos(((t/T + s) / (1 + s)) * pi/2)^2
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # 从 alpha_bar 反推 beta: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # 截断 beta 防止数值不稳定（最大不超过 0.999）
        return betas.clamp(0.0, 0.999)
        
    raise ValueError(f"未知的调度类型 (Unknown schedule type): {schedule_type}")


def extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """从张量 a 中根据索引 t 提取值，并重塑形状以支持广播运算。

    用于将形状为 (T,) 的系数张量（如 betas）提取为 (Batch,)，
    并广播到 (Batch, Channels, Height, Width) 以便与图像进行逐像素计算。

    Args:
        a (torch.Tensor): 数据源张量 (例如 betas 或 alphas)。
        t (torch.Tensor): 时间步索引张量 (Batch Size,)。
        shape (torch.Size): 目标张量的形状 (B, C, H, W)。

    Returns:
        torch.Tensor: 提取并重塑后的张量，形状为 (B, 1, 1, 1)。
    """
    out = a.gather(0, t)
    # 重塑为 (B, 1, 1, ..., 1) 以便与图像张量进行广播计算
    return out.view(-1, *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    """高斯扩散模型核心逻辑类。

    该类预计算了扩散过程中需要用到的所有系数（alphas, betas 等），
    并提供了前向加噪 (q_sample) 和反向去噪采样 (p_sample/ddim_step) 的方法。
    """

    def __init__(
        self,
        timesteps: int,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
        x0_clip_min: float = -1.0,
        x0_clip_max: float = 2.0,
    ) -> None:
        """初始化扩散模型参数并预计算系数。

        Args:
            timesteps (int): 总扩散步数。
            schedule_type (str): 噪声调度类型 ("linear" 或 "cosine")。
            beta_start (float): 线性调度起始 Beta。
            beta_end (float): 线性调度结束 Beta。
            device (torch.device, optional): 张量所在的设备。
            x0_clip_min (float): x0 预测值截断下限。
            x0_clip_max (float): x0 预测值截断上限。
        """
        self.timesteps = timesteps
        self.x0_clip_min = x0_clip_min
        self.x0_clip_max = x0_clip_max
        
        # 1. 生成 Beta 调度表
        betas = make_beta_schedule(timesteps, schedule_type, beta_start, beta_end)
        if device is not None:
            betas = betas.to(device)
        self.betas = betas
        
        # 2. 计算 Alpha 相关系数
        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - betas
        # alpha_cumprod (alpha_bar) = 累积连乘 alpha
        # 这是直接从 x0 预测 xt 的关键系数
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 3. 预计算数值稳定的辅助变量
        self.min_sqrt_alphas_cumprod = 1e-5
        self.min_alphas_cumprod = self.min_sqrt_alphas_cumprod ** 2
        
        # alpha_cumprod_prev (前一步的 alpha_bar)
        # 在 t=0 之前补 1.0，用于处理边界条件
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=betas.device), self.alphas_cumprod[:-1]], dim=0
        )
        
        # 4. 预计算用于 q_sample 的系数 (用于前向加噪)
        # sqrt(alpha_bar)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # sqrt(1 - alpha_bar)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 5. 预计算用于 p_sample 的系数 (用于反向去噪)
        # 1 / sqrt(alpha)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 计算后验方差 (Posterior Variance) - 用于 DDPM 采样中的随机噪声项
        # var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        # 这是贝叶斯公式推导出的真实后验方差
        denom = (1.0 - self.alphas_cumprod).clamp_min(self.min_alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / denom

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向扩散过程 (Forward Process)。

        根据公式: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        直接从原始图像 x0 采样出任意时刻 t 的加噪图像 x_t。
        这利用了高斯分布的可加性，无需一步步加噪。

        Args:
            x0 (torch.Tensor): 原始清晰图像 (Batch, Channel, Height, Width)。
            t (torch.Tensor): 时间步索引 (Batch,)。
            noise (torch.Tensor, optional): 高斯噪声。如果不提供则随机生成。

        Returns:
            torch.Tensor: 加噪后的图像 x_t。
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_cumprod = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        return sqrt_cumprod * x0 + sqrt_one_minus * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        clip: bool = False,
    ) -> torch.Tensor:
        """根据预测的噪声 epsilon，倒推预测 x0 (原始图像)。
        
        公式: x0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
        
        这个函数在计算重建损失 (Reconstruction Loss) 和采样过程中非常重要。

        Args:
            x_t (torch.Tensor): 当前时刻的加噪图像。
            t (torch.Tensor): 时间步。
            eps (torch.Tensor): 模型预测出的噪声。
            clip (bool): 是否对预测的 x0 进行数值截断。
                - 训练时通常设为 False，允许溢出让 Loss 惩罚。
                - 采样时通常设为 True，防止数值发散。

        Returns:
            torch.Tensor: 预测的 x0。
        """
        sqrt_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_cumprod = sqrt_cumprod.clamp_min(self.min_sqrt_alphas_cumprod)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        x0_pred = (x_t - sqrt_one_minus * eps) / sqrt_cumprod
        
        if clip:
            x0_pred = x0_pred.clamp(self.x0_clip_min, self.x0_clip_max)
        
        return x0_pred

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """DDPM 采样步 (标准反向过程)。

        根据当前图像 x_t 和预测噪声 eps，计算上一时刻的图像 x_{t-1}。
        包含随机噪声项，因此是随机采样过程。

        Args:
            x_t (torch.Tensor): 当前时刻图像。
            t (torch.Tensor): 当前时间步。
            eps (torch.Tensor): 模型预测的噪声。

        Returns:
            torch.Tensor: 上一时刻的图像 x_{t-1}。
        """
        # 1. 预测 x0 (包含 clamp)
        if (t == 0).all():
            return self.predict_x0_from_eps(x_t, t, eps)
            
        # 2. 计算后验分布的均值 mu_tilde(x_t, x0)
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = sqrt_one_minus.clamp_min(self.min_sqrt_alphas_cumprod)
        
        # mean = (1 / sqrt(alpha)) * (x_t - (beta / sqrt(1 - alpha_bar)) * eps)
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus * eps)
        
        # 3. 添加方差项 (随机噪声 z)
        # x_{t-1} = mean + sqrt(variance) * z
        variance = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        
        return mean + torch.sqrt(variance) * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor | None,
        eps: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM (Denoising Diffusion Implicit Models) 采样步。
        
        DDIM 是一种确定性或半确定性的采样方法，允许跳步采样 (加速生成)。
        它将扩散过程重新表述为非马尔可夫过程。

        Args:
            x_t (torch.Tensor): 当前时刻图像。
            t (torch.Tensor): 当前时间步。
            t_prev (torch.Tensor | None): 下一个目标时间步 (通常是 t - step_size)。
            eps (torch.Tensor): 模型预测的噪声。
            eta (float): 随机性系数 (0.0 为确定性采样，1.0 接近 DDPM)。

        Returns:
            torch.Tensor: 下一时刻的图像 x_{t_prev}。
        """
        # 1. 获取当前时刻的 alpha_bar
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)
        safe_alpha_bar_t = alpha_bar_t.clamp_min(self.min_alphas_cumprod)
        sqrt_alpha_bar_t = torch.sqrt(safe_alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - safe_alpha_bar_t)
        
        # 2. 预测 x0 (使用类属性的 clip 范围)
        x0_pred = (x_t - sqrt_one_minus * eps) / sqrt_alpha_bar_t
        x0_pred = x0_pred.clamp(self.x0_clip_min, self.x0_clip_max)
        
        if t_prev is None:
            return x0_pred
            
        # 3. 获取目标时刻 (t_prev) 的 alpha_bar
        alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)
        safe_alpha_bar_prev = alpha_bar_prev.clamp_min(self.min_alphas_cumprod)
        
        # 4. 计算 DDIM 更新公式
        if eta == 0.0:
            # 确定性更新 (Deterministic)
            # x_{t-1} = sqrt(alpha_bar_prev) * x0_pred + sqrt(1 - alpha_bar_prev) * eps
            return torch.sqrt(safe_alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - safe_alpha_bar_prev) * eps
        
        # 带有随机性的更新 (Stochastic) - 如果 eta > 0
        # 计算 sigma (标准差)
        sigma_sq = (
            (1.0 - safe_alpha_bar_prev)
            / (1.0 - safe_alpha_bar_t)
            * (1.0 - safe_alpha_bar_t / safe_alpha_bar_prev)
        )
        sigma = eta * torch.sqrt(torch.clamp(sigma_sq, min=0.0))
        noise = torch.randn_like(x_t)
        
        # 指向预测噪声的方向分量 (Direction pointing to x_t)
        dir_eps = torch.sqrt(torch.clamp(1.0 - safe_alpha_bar_prev - sigma ** 2, min=0.0)) * eps
        
        # 组合各项：信号项 + 方向项 + 随机噪声项
        return torch.sqrt(safe_alpha_bar_prev) * x0_pred + dir_eps + sigma * noise

    def sample_timesteps(self, steps: int) -> List[int]:
        """生成采样所需的时间步序列。
        
        用于 DDIM 跳步采样。例如，如果总步数是 1000，但只想采样 50 步，
        则生成 [999, 979, ..., 0] 这样的序列。

        Args:
            steps (int): 采样总步数 (例如 50)。
            
        Returns:
            List[int]: 时间步列表，从 T-1 递减到 0。
        """
        return torch.linspace(self.timesteps - 1, 0, steps).long().tolist()


def schedule_weight(step: int, total: int, stage1: float, stage2: float, stage3: float,
                    frac1: float, frac2: float) -> float:
    """计算分阶段的权重调度 (辅助函数)。
    
    允许在训练的不同阶段使用不同的 Loss 权重。目前主要用于实验性的动态 Loss 调整。
    """
    if total <= 1:
        return stage3
    t1 = int(total * frac1)
    t2 = int(total * (frac1 + frac2))
    if step < t1:
        return stage1
    if step < t2:
        if t2 == t1:
            return stage2
        return stage1 + (stage2 - stage1) * (step - t1) / (t2 - t1)
    if total - t2 <= 1:
        return stage3
    return stage2 + (stage3 - stage2) * (step - t2) / (total - t2 - 1)
