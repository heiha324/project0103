"""Hierarchical conditional transformer denoiser for SAR-guided cloud removal."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


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


class CondFuse2d(nn.Module):
    """Fuse cloudy-optical and SAR condition features with stage-aware gating."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.y_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.s_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gates = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=1),
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
        y_gate, s_gate = torch.sigmoid(self.gates(torch.cat([x, y_feat, s_feat], dim=1))).chunk(2, dim=1)
        cond = y_gate * y_feat + s_gate * s_feat
        return self.out(cond)


class ConditionalTransformer(nn.Module):
    """Hierarchical transformer denoiser with multi-scale optical/SAR conditioning."""

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        s_channels: int,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        patch_size: int = 16,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        refine_channels: int = 128,
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
        detail_channels = max(32, stage_dims[0] // 2)
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
        self.fuse_blocks = nn.ModuleList([CondFuse2d(dim) for dim in stage_dims])

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
