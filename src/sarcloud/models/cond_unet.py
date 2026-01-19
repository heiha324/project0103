"""多模态条件 U-Net 模型 (Conditional U-Net with Optical and SAR Encoders)。

该模型是扩散去噪的核心网络，其架构特点包括：
1. **多模态输入**: 同时也接收光学图像 (Optical) 和 SAR 图像作为条件。
2. **时间嵌入 (Time Embedding)**: 将扩散时间步 t 编码为向量，注入到每个残差块中。
3. **双编码器结构**: 分别提取光学和 SAR 特征，并在解码过程中融合。
4. **ResBlock + Attention**: 标准的 U-Net 积木。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码 (Sinusoidal Positional Embedding)。
    
    将标量时间步 t 转换为高维向量，用于告诉网络当前的去噪进度。
    公式参考 Transformer 的位置编码: sin(t * freq), cos(t * freq)。
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步张量 (Batch Size,)。
        
        Returns:
            Embedding 张量 (Batch Size, Dim)。
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    """标准的 ResNet 残差块，集成了时间嵌入注入。
    
    结构: GroupNorm -> SiLU -> Conv -> + TimeEmb -> GroupNorm -> SiLU -> Conv -> + Skip
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # 时间嵌入的投影层 (将 time_emb 映射到特征通道数)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # 如果输入输出通道不一致，使用 1x1 卷积调整 skip connection
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 (B, C_in, H, W)。
            t_emb: 时间嵌入向量 (B, time_dim)。
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 注入时间信息 (Scale & Shift 或 直接相加)
        # 这里采用直接相加: Feature + Time
        time = self.time_mlp(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = h + time
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    """下采样层 (使用步长为2的卷积)。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样层 (使用转置卷积)。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """条件特征编码器 (用于提取 Optical 或 SAR 图像的特征)。
    
    结构类似于 U-Net 的左半部分 (Contracting path)。
    """
    def __init__(self, in_ch: int, base_ch: int, depth: int, time_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = in_ch
        
        # 构建下采样层级
        for i in range(depth):
            out_ch = base_ch * (2**i)
            self.blocks.append(ResBlock(ch, out_ch, time_dim))
            self.downs.append(Downsample(out_ch))
            ch = out_ch
            
        self.bottleneck = ResBlock(ch, ch * 2, time_dim)
        self.out_channels = ch * 2

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            feats: 中间层的特征列表 (用于 Skip Connections)。
            x: 最终的 Bottleneck 特征。
        """
        feats: List[torch.Tensor] = []
        for block, down in zip(self.blocks, self.downs):
            x = block(x, t_emb)
            feats.append(x)
            x = down(x)
        x = self.bottleneck(x, t_emb)
        return feats, x


class CondFuse(nn.Module):
    """特征融合模块 (Condition Fusion)。
    
    用于融合 光学特征 (Optical) 和 SAR 特征。
    采用门控机制 (Gating Mechanism):
    Fused = Optical + Sigmoid(Gate(Optical)) * SAR
    """
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gate = nn.Conv2d(ch, ch, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        # 如果尺寸不匹配 (通常不需要，为了鲁棒性)
        if f_opt.shape != f_sar.shape:
            f_sar = F.interpolate(f_sar, size=f_opt.shape[-2:], mode="bilinear", align_corners=False)
            
        # 计算门控权重 (0~1)
        gate = torch.sigmoid(self.gate(f_opt))
        # 加权融合
        fused = f_opt + gate * f_sar
        return self.proj(fused)


class ConditionalUNet(nn.Module):
    """主去噪网络 (Conditional U-Net)。
    
    输入:
        x_t: 当前时刻的噪声图像。
        t: 时间步。
        y: 条件图像 1 (Cloudy Optical)。
        s: 条件图像 2 (SAR)。
    输出:
        eps: 预测的噪声。
    """
    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        s_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.depth = depth

        # 1. 时间嵌入处理层
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # 2. 条件编码器 (双塔结构)
        self.opt_encoder = Encoder(y_channels, base_channels, depth, time_dim)
        self.sar_encoder = Encoder(s_channels, base_channels, depth, time_dim)

        # 3. 主 U-Net 编码器 (Encoder Path)
        self.input_conv = nn.Conv2d(x_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fuse_blocks = nn.ModuleList() # 用于融合条件特征

        ch = base_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.down_blocks.append(ResBlock(ch, out_ch, time_dim))
            self.fuse_blocks.append(CondFuse(out_ch))
            self.downs.append(Downsample(out_ch))
            ch = out_ch

        # 4. Bottleneck (中间层)
        # 主路最后一个 down block 输出通道数 = base_channels * 2^(depth-1)
        main_bot_ch = base_channels * (2 ** (depth - 1))
        # 条件编码器 bottleneck 输出通道数 = base_channels * 2^(depth-1) * 2 = base_channels * 2^depth
        cond_bot_ch = self.opt_encoder.out_channels
        # 投影层：将主路特征投影到 bottleneck 通道数
        self.main_to_bot_proj = nn.Conv2d(main_bot_ch, cond_bot_ch, kernel_size=1)
        
        self.mid_block1 = ResBlock(cond_bot_ch, cond_bot_ch, time_dim)
        self.mid_block2 = ResBlock(cond_bot_ch, cond_bot_ch, time_dim)

        # 5. 主 U-Net 解码器 (Decoder Path)
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = self.opt_encoder.out_channels
        for i in reversed(range(depth)):
            out_ch = base_channels * (2**i)
            self.ups.append(Upsample(ch))
            # 输入通道 = 上一层输出 + Skip Connection
            self.up_blocks.append(ResBlock(ch + out_ch, out_ch, time_dim))
            ch = out_ch

        # 6. 输出层
        self.output_norm = nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)
        self.output_conv = nn.Conv2d(ch, x_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 编码时间步
        t_emb = self.time_mlp(t)

        # 2. 提取条件特征 (Optical & SAR)
        opt_in = y
        opt_feats, opt_bot = self.opt_encoder(opt_in, t_emb)
        sar_feats, sar_bot = self.sar_encoder(s, t_emb)

        # 3. 主路编码过程
        x = self.input_conv(x_t)
        skips: List[torch.Tensor] = []
        for i in range(self.depth):
            # 融合当前层级的条件特征
            fused = self.fuse_blocks[i](opt_feats[i], sar_feats[i])
            x = self.down_blocks[i](x, t_emb)
            # 将融合特征注入主路
            x = x + fused
            skips.append(x)
            x = self.downs[i](x)

        # 4. Bottleneck 处理
        # 融合 Bottleneck 处的条件特征与主路特征
        if opt_bot.shape != sar_bot.shape:
            sar_bot = F.interpolate(sar_bot, size=opt_bot.shape[-2:], mode="bilinear", align_corners=False)
        
        # 空间尺寸对齐
        if x.shape[-2:] != opt_bot.shape[-2:]:
            x = F.interpolate(x, size=opt_bot.shape[-2:], mode="bilinear", align_corners=False)
        
        # 使用投影层将主路特征投影到 bottleneck 通道数，然后三者融合
        x_proj = self.main_to_bot_proj(x)
        x = x_proj + opt_bot + sar_bot
        
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        # 5. 解码过程
        for i in reversed(range(self.depth)):
            idx = self.depth - 1 - i
            x = self.ups[idx](x)
            
            # 获取 Skip Connection
            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            
            # 拼接
            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[idx](x, t_emb)

        # 6. 输出预测
        x = self.output_norm(x)
        x = F.silu(x)
        return self.output_conv(x)
