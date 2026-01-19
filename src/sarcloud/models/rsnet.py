"""RS-Net (lightweight U-Net style) for cloud detection."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RSNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        depth: int = 4,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.down_blocks.append(ConvBlock(ch, out_ch, use_batchnorm))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            ch = out_ch

        self.bottleneck = ConvBlock(ch, ch * 2, use_batchnorm)
        ch = ch * 2

        for i in reversed(range(depth)):
            out_ch = base_channels * (2 ** i)
            self.up_blocks.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(ch, out_ch, use_batchnorm))
            ch = out_ch

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for i in range(self.depth):
            up = self.up_blocks[2 * i]
            conv = self.up_blocks[2 * i + 1]
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.head(x)
