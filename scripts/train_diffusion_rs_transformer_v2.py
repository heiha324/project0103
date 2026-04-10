#!/usr/bin/env python3
"""Train Residual Shifting diffusion with a transformer denoiser."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import train_diffusion_rs as base
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion.losses import grad_l1_loss
from sarcloud.training.ema import EMA
from sarcloud.utils.config import load_config


SAMPLING_METRIC_PROTOCOL = f"{base.METRIC_PROTOCOL}:sampling"


def _valid_group_count(channels: int, max_groups: int = 32) -> int:
    """Return the largest valid GroupNorm group count."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _align_channels(channels: int, num_heads: int) -> int:
    """Align channels so each attention stage is divisible by head count."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    return max(num_heads, math.ceil(channels / num_heads) * num_heads)


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class LayerNorm2d(nn.Module):
    """LayerNorm applied on the channel dimension of BCHW tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class AdaLayerNorm2d(nn.Module):
    """LayerNorm with timestep-conditioned affine parameters for 2D features."""

    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ConvResidualBlock(nn.Module):
    """Residual convolution block used for high-resolution/detail paths."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_group_count(in_ch), in_ch)
        self.norm2 = nn.GroupNorm(_valid_group_count(out_ch), out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConvStage(nn.Module):
    """Shallow convolutional stage for condition pyramids."""

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ConvResidualBlock(channels, channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Downsample2d(nn.Module):
    """Spatial downsampling with a convolutional projection."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvResidualBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Upsample2d(nn.Module):
    """Bilinear upsampling followed by local refinement."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            ConvResidualBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class WindowAttention2d(nn.Module):
    """Local window attention operating on BCHW feature maps."""

    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.window_size = max(int(window_size), 1)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.window_size
        pad_w = (-width) % self.window_size
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height, width

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        ws = self.window_size
        x = x.view(batch_size, channels, height // ws, ws, width // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, ws * ws, channels)

    def _window_reverse(
        self,
        windows: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        ws = self.window_size
        channels = windows.shape[-1]
        x = windows.view(batch_size, height // ws, width // ws, ws, ws, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(batch_size, channels, height, width)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x_pad, orig_h, orig_w = self._pad(x)
        batch_size, _channels, padded_h, padded_w = x_pad.shape
        query = self._window_partition(x_pad)

        if context is None:
            key_value = query
        else:
            if context.shape[-2:] != x.shape[-2:]:
                context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
            context_pad, _, _ = self._pad(context)
            key_value = self._window_partition(context_pad)

        out, _ = self.attn(query, key_value, key_value, need_weights=False)
        out = self._window_reverse(out, batch_size, padded_h, padded_w)
        return out[:, :, :orig_h, :orig_w]


class FeedForward2d(nn.Module):
    """Transformer feed-forward network with depthwise local mixing."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = max(int(dim * mlp_ratio), dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock2d(nn.Module):
    """Windowed transformer block with timestep conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_dim: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm2d(dim, time_dim)
        self.norm2 = AdaLayerNorm2d(dim, time_dim)
        self.norm3 = AdaLayerNorm2d(dim, time_dim)
        self.self_attn = WindowAttention2d(dim, num_heads, window_size, dropout=dropout)
        self.cross_attn = WindowAttention2d(dim, num_heads, window_size, dropout=dropout)
        self.ffn = FeedForward2d(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x, t_emb))
        x = x + self.cross_attn(self.norm2(x, t_emb), cond)
        x = x + self.ffn(self.norm3(x, t_emb))
        return x


class TransformerStage(nn.Module):
    """A stack of transformer blocks at the same spatial resolution."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        time_dim: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock2d(
                    dim=dim,
                    num_heads=num_heads,
                    time_dim=time_dim,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond, t_emb)
        return x


class SEChannelAttention2d(nn.Module):
    """Squeeze-and-Excitation style channel recalibration for BCHW features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        reduction = max(1, reduction)
        hidden_channels = max(channels // reduction, 32)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class CondFuse2d(nn.Module):
    """Fuse cloudy-optical and SAR condition features with SE + spatial gating."""

    def __init__(self, channels: int, se_reduction: int = 16) -> None:
        super().__init__()
        cat_channels = channels * 3
        self.y_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.s_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.se = SEChannelAttention2d(cat_channels, reduction=se_reduction)
        self.gates = nn.Sequential(
            nn.Conv2d(cat_channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            ConvResidualBlock(channels, channels),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if s.shape[-2:] != x.shape[-2:]:
            s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)

        y_feat = self.y_proj(y)
        s_feat = self.s_proj(s)
        cat_feat = torch.cat([x, y_feat, s_feat], dim=1)
        cat_feat = self.se(cat_feat)
        y_gate, s_gate = self.gates(cat_feat).chunk(2, dim=1)
        y_gate = torch.sigmoid(y_gate)
        s_gate = torch.sigmoid(s_gate)
        cond = y_gate * y_feat + s_gate * s_feat
        return self.out(cond)


class ConditionalTransformer(nn.Module):
    """Hierarchical transformer denoiser with multi-scale optical/SAR conditioning."""

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        s_channels: int,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        patch_size: int = 16,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        refine_channels: int = 256,
        detail_min_channels: int = 128,
        cond_se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.x_channels = x_channels
        self.stem_stride = max(2, patch_size // 4)
        self.window_size = min(8, max(4, patch_size // 2))

        stage_heads = [max(1, num_heads // 4), max(1, num_heads // 2), max(1, num_heads)]
        stage_dims = [
            _align_channels(max(embed_dim // 4, stage_heads[0] * 8), stage_heads[0]),
            _align_channels(max(embed_dim // 2, stage_heads[1] * 8), stage_heads[1]),
            _align_channels(max(embed_dim, stage_heads[2] * 8), stage_heads[2]),
        ]
        detail_channels = max(detail_min_channels, stage_dims[0] // 2, x_channels + y_channels)
        cond_stage_depth = max(1, depth // 8)
        shared_depth = max(1, depth // 5)
        bottleneck_depth = max(1, depth - shared_depth * 5)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.detail_stem = nn.Sequential(
            nn.Conv2d(x_channels + y_channels, detail_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ConvResidualBlock(detail_channels, detail_channels),
            ConvResidualBlock(detail_channels, detail_channels),
        )

        self.x_stem = nn.Sequential(
            nn.Conv2d(x_channels, stage_dims[0], kernel_size=7, stride=self.stem_stride, padding=3),
            nn.GELU(),
            ConvResidualBlock(stage_dims[0], stage_dims[0]),
        )
        self.y_stem = nn.Sequential(
            nn.Conv2d(y_channels, stage_dims[0], kernel_size=7, stride=self.stem_stride, padding=3),
            nn.GELU(),
            ConvResidualBlock(stage_dims[0], stage_dims[0]),
        )
        self.s_stem = nn.Sequential(
            nn.Conv2d(s_channels, stage_dims[0], kernel_size=7, stride=self.stem_stride, padding=3),
            nn.GELU(),
            ConvResidualBlock(stage_dims[0], stage_dims[0]),
        )

        self.x_stages = nn.ModuleList(
            [
                TransformerStage(
                    dim=dim,
                    num_heads=heads,
                    depth=shared_depth,
                    time_dim=time_dim,
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for dim, heads in zip(stage_dims, stage_heads)
            ]
        )
        self.y_stages = nn.ModuleList([ConvStage(dim, cond_stage_depth) for dim in stage_dims])
        self.s_stages = nn.ModuleList([ConvStage(dim, cond_stage_depth) for dim in stage_dims])
        self.fuse_blocks = nn.ModuleList(
            [CondFuse2d(dim, se_reduction=cond_se_reduction) for dim in stage_dims]
        )

        self.x_downs = nn.ModuleList(
            [Downsample2d(stage_dims[i], stage_dims[i + 1]) for i in range(len(stage_dims) - 1)]
        )
        self.y_downs = nn.ModuleList(
            [Downsample2d(stage_dims[i], stage_dims[i + 1]) for i in range(len(stage_dims) - 1)]
        )
        self.s_downs = nn.ModuleList(
            [Downsample2d(stage_dims[i], stage_dims[i + 1]) for i in range(len(stage_dims) - 1)]
        )

        self.bottleneck = TransformerStage(
            dim=stage_dims[-1],
            num_heads=stage_heads[-1],
            depth=bottleneck_depth,
            time_dim=time_dim,
            window_size=self.window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.ups = nn.ModuleList()
        self.skip_merges = nn.ModuleList()
        self.dec_cond_projs = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for stage_index in reversed(range(len(stage_dims) - 1)):
            out_dim = stage_dims[stage_index]
            heads = stage_heads[stage_index]
            self.ups.append(Upsample2d(stage_dims[stage_index + 1], out_dim))
            self.skip_merges.append(
                nn.Sequential(
                    nn.Conv2d(out_dim * 2, out_dim, kernel_size=1),
                    nn.GELU(),
                    ConvResidualBlock(out_dim, out_dim),
                )
            )
            self.dec_cond_projs.append(
                nn.Sequential(
                    nn.Conv2d(out_dim * 2, out_dim, kernel_size=1),
                    nn.GELU(),
                    ConvResidualBlock(out_dim, out_dim),
                )
            )
            self.dec_stages.append(
                TransformerStage(
                    dim=out_dim,
                    num_heads=heads,
                    depth=shared_depth,
                    time_dim=time_dim,
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )

        final_hidden = max(refine_channels, stage_dims[0])
        self.final_head = nn.Sequential(
            nn.Conv2d(stage_dims[0] + detail_channels, stage_dims[0], kernel_size=3, padding=1),
            nn.GELU(),
            ConvResidualBlock(stage_dims[0], stage_dims[0]),
            nn.Conv2d(stage_dims[0], final_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(final_hidden, x_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t.float())

        detail = self.detail_stem(torch.cat([x_t, y], dim=1))
        x = self.x_stem(x_t)
        y_feat = self.y_stem(y)
        s_feat = self.s_stem(s)

        encoder_skips: list[torch.Tensor] = []
        cond_skips: list[torch.Tensor] = []
        for idx in range(len(self.x_stages)):
            y_feat = self.y_stages[idx](y_feat)
            s_feat = self.s_stages[idx](s_feat)
            cond = self.fuse_blocks[idx](x, y_feat, s_feat)
            x = self.x_stages[idx](x, cond, t_emb)
            encoder_skips.append(x)
            cond_skips.append(cond)

            if idx < len(self.x_downs):
                x = self.x_downs[idx](x)
                y_feat = self.y_downs[idx](y_feat)
                s_feat = self.s_downs[idx](s_feat)

        x = self.bottleneck(x, cond_skips[-1], t_emb)

        for idx, up in enumerate(self.ups):
            skip_index = len(self.x_stages) - 2 - idx
            x = up(x)
            skip = encoder_skips[skip_index]
            cond = cond_skips[skip_index]

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = self.skip_merges[idx](torch.cat([x, skip], dim=1))
            dec_cond = self.dec_cond_projs[idx](torch.cat([cond, skip], dim=1))
            x = self.dec_stages[idx](x, dec_cond, t_emb)

        if x.shape[-2:] != detail.shape[-2:]:
            x = F.interpolate(x, size=detail.shape[-2:], mode="bilinear", align_corners=False)

        eps = self.final_head(torch.cat([x, detail], dim=1))
        return eps


def evaluate_sampling_steps(
    model: torch.nn.Module,
    loader,
    diffusion: ResidualShiftingDiffusion,
    cfg: dict[str, Any],
    device: torch.device,
    amp_device: str,
    amp_enabled: bool,
    sampling_steps: list[int],
    desc: str,
    max_batches: int,
    use_tqdm: bool,
    ema: EMA | None = None,
    use_ema: bool = False,
    ddp: bool = False,
) -> dict[int, dict[str, float]]:
    """按给定采样步数执行完整采样评估并聚合指标。"""
    backup = None
    if use_ema and ema is not None:
        backup = base.apply_ema_weights(model, ema)

    metric_keys = [
        "mae",
        "mse",
        "rmse",
        "nrmse",
        "psnr",
        "ssim",
        "ms_ssim",
        "sam",
        "ergas",
        "cc",
        "uiqi",
        "rase",
    ]
    totals: dict[int, dict[str, float]] = {
        int(step): {k: 0.0 for k in metric_keys} for step in sampling_steps
    }
    sample_count = 0

    show_progress = use_tqdm and (not ddp or dist.get_rank() == 0)
    iterator = loader
    if base.tqdm is not None and show_progress:
        iterator = base.tqdm(loader, desc=desc, ncols=80, leave=False)

    sampling_cfg = cfg.get("sampling", {})
    model.eval()
    with torch.no_grad():
        for step_idx, (s1, s2_cloudy, s2_clear, _alpha) in enumerate(iterator, start=1):
            if max_batches > 0 and step_idx > max_batches:
                break

            s1 = s1.to(device)
            y = s2_cloudy.to(device)
            x0 = s2_clear.to(device)
            batch_size = int(x0.size(0))

            for step_cnt in sampling_steps:
                with torch.amp.autocast(amp_device, enabled=amp_enabled):
                    x0_pred = base.sample_batch_rs(
                        model,
                        diffusion,
                        y,
                        s1,
                        steps=int(step_cnt),
                        schedule_cfg=sampling_cfg,
                    )
                    x0_pred = x0_pred.clamp(0.0, 1.0)

                    x0_p = x0_pred.detach().float()
                    x0_t = x0.detach().float()
                    base.accumulate_image_metric_totals(
                        totals[int(step_cnt)],
                        x0_p,
                        x0_t,
                        metric_keys=tuple(metric_keys),
                    )

            sample_count += batch_size

    if backup is not None:
        base.restore_weights(model, backup)

    if sample_count == 0:
        return {int(step): {k: float("nan") for k in metric_keys} for step in sampling_steps}

    if ddp:
        values: list[float] = []
        for step_cnt in sampling_steps:
            values.extend(totals[int(step_cnt)][k] for k in metric_keys)
        values.append(float(sample_count))
        tensor = torch.tensor(values, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_samples = tensor[-1].item()
        if total_samples <= 0:
            return {int(step): {k: float("nan") for k in metric_keys} for step in sampling_steps}
        out: dict[int, dict[str, float]] = {}
        idx = 0
        for step_cnt in sampling_steps:
            out_step: dict[str, float] = {}
            for k in metric_keys:
                out_step[k] = tensor[idx].item() / total_samples
                idx += 1
            out[int(step_cnt)] = out_step
        return out

    return {
        int(step_cnt): {k: totals[int(step_cnt)][k] / sample_count for k in metric_keys}
        for step_cnt in sampling_steps
    }


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
        eval_sampler = base.build_distributed_eval_sampler(
            eval_dataset,
            ddp=True,
            rank=rank,
            world_size=world_size,
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
        embed_dim=model_cfg.get("embed_dim", 512),
        depth=model_cfg.get("depth", 8),
        num_heads=model_cfg.get("num_heads", 16),
        patch_size=model_cfg.get("patch_size", 16),
        time_dim=model_cfg.get("time_dim", 256),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        refine_channels=model_cfg.get("refine_channels", 256),
        detail_min_channels=model_cfg.get("detail_min_channels", 128),
        cond_se_reduction=model_cfg.get("cond_se_reduction", 16),
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
    best_psnr_20 = -math.inf
    best_train_loss = math.inf
    best_psnr_epoch = -1
    best_train_loss_epoch = -1
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

        loaded_sampling_protocol = checkpoint.get("sampling_metrics_protocol")
        if loaded_sampling_protocol == SAMPLING_METRIC_PROTOCOL:
            resume_best_psnr_20 = checkpoint.get("best_psnr_20")
            if resume_best_psnr_20 is None:
                sampling_metrics = checkpoint.get("sampling_eval_metrics") or {}
                step20_metrics = sampling_metrics.get(20) or sampling_metrics.get("20") or {}
                resume_best_psnr_20 = step20_metrics.get("psnr")
            if resume_best_psnr_20 is not None and math.isfinite(float(resume_best_psnr_20)):
                best_psnr_20 = float(resume_best_psnr_20)
                if is_main:
                    print(f"Loaded best 20-step PSNR {best_psnr_20:.4f} from {ckpt_path}")
            if checkpoint.get("best_psnr_epoch") is not None:
                best_psnr_epoch = int(checkpoint.get("best_psnr_epoch"))
        elif is_main:
            print(
                "Sampling metric protocol changed; keeping model/optimizer state "
                "but resetting best 20-step PSNR tracking."
            )

        resume_best_train_loss = checkpoint.get("best_train_loss")
        if resume_best_train_loss is None:
            resume_best_train_loss = checkpoint.get("train_loss")
        if resume_best_train_loss is not None and math.isfinite(float(resume_best_train_loss)):
            best_train_loss = float(resume_best_train_loss)
            if is_main:
                print(f"Loaded best train loss {best_train_loss:.4f} from {ckpt_path}")

        if checkpoint.get("best_train_loss_epoch") is not None:
            best_train_loss_epoch = int(checkpoint.get("best_train_loss_epoch"))

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
    save_every_epochs = max(1, int(output_cfg.get("save_every_epochs", 3)))
    psnr_eval_steps = output_cfg.get("psnr_eval_steps", [5, 10, 20])
    if isinstance(psnr_eval_steps, int):
        psnr_eval_steps = [int(psnr_eval_steps)]
    else:
        psnr_eval_steps = [int(step) for step in psnr_eval_steps]
    psnr_eval_steps = sorted({step for step in psnr_eval_steps if step > 0})
    if 20 not in psnr_eval_steps:
        psnr_eval_steps.append(20)
        psnr_eval_steps = sorted(psnr_eval_steps)
    psnr_eval_max_batches = int(output_cfg.get("psnr_eval_max_batches", 0))
    if ddp and psnr_eval_max_batches > 0:
        psnr_eval_max_batches = max(1, (psnr_eval_max_batches + world_size - 1) // world_size)

    vis_cfg = cfg.setdefault("vis", {})
    vis_cfg["steps"] = [5, 10, 20, 50, 100]

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
        base.log_message(f"周期保存间隔: {save_every_epochs} epochs", logger, console=False, use_tqdm=False)
        base.log_message(
            f"PSNR评估步数: {psnr_eval_steps} (max_batches={psnr_eval_max_batches}, 0=全量)",
            logger,
            console=False,
            use_tqdm=False,
        )
        base.log_message(
            f"可视化步数固定为: {vis_cfg['steps']}",
            logger,
            console=False,
            use_tqdm=False,
        )

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

        run_sampling_eval = (epoch + 1) % save_every_epochs == 0
        sampling_eval_metrics: dict[int, dict[str, float]] | None = None
        if run_sampling_eval:
            sampling_eval_metrics = evaluate_sampling_steps(
                eval_model,
                eval_loader,
                diffusion,
                cfg,
                device,
                amp_device,
                amp_enabled,
                sampling_steps=psnr_eval_steps,
                desc=f"Test Sampling {'/'.join(str(s) for s in psnr_eval_steps)}",
                max_batches=psnr_eval_max_batches,
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

            if sampling_eval_metrics is not None:
                for step_cnt in psnr_eval_steps:
                    step_metrics = sampling_eval_metrics.get(int(step_cnt)) or {}
                    base.log_message(
                        f"Epoch {epoch+1}/{num_epochs} - [Sampling {step_cnt}] "
                        f"PSNR {step_metrics.get('psnr', float('nan')):.2f} "
                        f"SSIM {step_metrics.get('ssim', float('nan')):.4f} "
                        f"MAE {step_metrics.get('mae', float('nan')):.4f}",
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
            checkpoint = {
                "epoch": epoch,
                "model_state": model_state,
                "ema_state": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "config": cfg,
                "train_loss": train_loss,
                "test_metrics": eval_metrics,
                "sampling_eval_metrics": sampling_eval_metrics,
                "test_metrics_protocol": base.METRIC_PROTOCOL,
                "sampling_metrics_protocol": SAMPLING_METRIC_PROTOCOL,
                "save_every_epochs": save_every_epochs,
                "psnr_eval_steps": psnr_eval_steps,
                "best_psnr_20": best_psnr_20,
                "best_psnr_epoch": best_psnr_epoch,
                "best_train_loss": best_train_loss,
                "best_train_loss_epoch": best_train_loss_epoch,
            }

            if math.isfinite(train_loss) and train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_loss_epoch = epoch + 1
                checkpoint["best_train_loss"] = best_train_loss
                checkpoint["best_train_loss_epoch"] = best_train_loss_epoch
                torch.save(checkpoint, out_dir / "diffusion_rs_transformer_best_train_loss.pth")
                base.log_message(
                    f"Epoch {epoch+1}/{num_epochs} - saved diffusion_rs_transformer_best_train_loss.pth "
                    f"(train_loss {train_loss:.4f})",
                    logger,
                    console=True,
                    use_tqdm=True,
                )

            if sampling_eval_metrics is not None:
                step20_metrics = sampling_eval_metrics.get(20) or {}
                step20_psnr = float(step20_metrics.get("psnr", float("nan")))
                if math.isfinite(step20_psnr) and step20_psnr > best_psnr_20:
                    best_psnr_20 = step20_psnr
                    best_psnr_epoch = epoch + 1
                    checkpoint["best_psnr_20"] = best_psnr_20
                    checkpoint["best_psnr_epoch"] = best_psnr_epoch
                    torch.save(checkpoint, out_dir / "diffusion_rs_transformer_best_psnr.pth")
                    base.log_message(
                        f"Epoch {epoch+1}/{num_epochs} - saved diffusion_rs_transformer_best_psnr.pth "
                        f"(20-step PSNR {step20_psnr:.4f})",
                        logger,
                        console=True,
                        use_tqdm=True,
                    )

            if run_sampling_eval:
                checkpoint["best_psnr_20"] = best_psnr_20
                checkpoint["best_psnr_epoch"] = best_psnr_epoch
                checkpoint["best_train_loss"] = best_train_loss
                checkpoint["best_train_loss_epoch"] = best_train_loss_epoch
                torch.save(checkpoint, out_dir / "diffusion_rs_transformer_last.pth")
                torch.save({"ema_state": ema.shadow}, out_dir / "diffusion_rs_transformer_ema.pth")
                periodic_ckpt_path = out_dir / f"diffusion_rs_transformer_epoch_{epoch+1:04d}.pth"
                torch.save(checkpoint, periodic_ckpt_path)
                base.log_message(
                    f"Epoch {epoch+1}/{num_epochs} - periodic checkpoint saved (every {save_every_epochs} epochs)",
                    logger,
                    console=True,
                    use_tqdm=True,
                )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
