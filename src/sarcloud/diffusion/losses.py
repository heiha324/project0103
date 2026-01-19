"""扩散模型训练所需的损失函数工具 (Loss Helpers)。

本模块包含辅助损失函数的实现，特别是用于增强图像纹理和边缘清晰度的梯度损失 (Gradient Loss)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_map(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """计算图像的梯度图 (边缘强度)。

    使用 Sobel 算子分别计算水平 (x) 和垂直 (y) 方向的梯度，并合成总梯度幅值。
    这有助于模型关注图像的高频细节（纹理、边缘）。

    Args:
        x (torch.Tensor): 输入图像张量 (Batch, Channel, Height, Width)。
        eps (float): 防止 sqrt(0) 的小数值，默认 1e-4。

    Returns:
        torch.Tensor: 梯度幅值图，形状与输入相同。
    """
    # 定义 Sobel 算子核
    # sobel_x: 检测垂直边缘 (水平方向的梯度)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=x.device, dtype=x.dtype)
    # sobel_y: 检测水平边缘 (垂直方向的梯度)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=x.device, dtype=x.dtype)
    
    # 重塑为卷积核形状 (Out_C, In_C, K, K) -> (1, 1, 3, 3)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # 扩展卷积核以匹配输入通道数 (使用分组卷积处理每个通道)
    channels = x.shape[1]
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    
    # 使用卷积计算梯度
    # groups=channels 意味着每个通道独立计算梯度，互不干扰
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
    
    # 计算梯度幅值: sqrt(dx^2 + dy^2)
    # 加上 epsilon 防止 sqrt(0) 导致的梯度消失或 NaN
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)


def grad_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算加权梯度 L1 损失 (Weighted Gradient L1 Loss)。

    该损失函数衡量预测图和真实图在“纹理/边缘”上的差异。
    使用加权平均 (Masked Mean) 而非简单平均，是为了处理某些区域权重极小甚至为 0 的情况。

    Args:
        pred (torch.Tensor): 模型预测的图像 (通常是 x0_pred)。
        target (torch.Tensor): 真实目标图像 (x0)。
        weight (torch.Tensor): 像素级权重图 (例如时间权重)。
        eps (float): 防止除零的小数。

    Returns:
        torch.Tensor: 标量损失值。
    """
    # 1. 计算两者的梯度图
    grad_pred = gradient_map(pred)
    grad_target = gradient_map(target)
    
    # 2. 调整权重形状以进行广播 (Broadcasting)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1) # (B, H, W) -> (B, 1, H, W)
    
    # 如果权重是单通道但图像是多通道，则复制权重
    if weight.shape[1] == 1 and grad_pred.shape[1] != 1:
        weight = weight.expand(-1, grad_pred.shape[1], -1, -1)
    
    # 3. 计算梯度的 L1 差异
    diff = torch.abs(grad_pred - grad_target)
    
    # 4. 应用权重并计算平均值
    # 使用 sum(weighted_diff) / sum(weight) 而非 mean()
    # 这样可以正确处理稀疏权重 (例如掩码边缘)
    weighted_diff = weight * diff
    denom = weight.sum().clamp_min(eps)
    
    return weighted_diff.sum() / denom
