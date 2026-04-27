from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RSNet(nn.Module):
    """
    RSNet (teacher) for cloud/thin/thick/shadow segmentation:
    U-Net style decoder with a ResNet50 encoder.
    """

    def __init__(
        self,
        *,
        in_channels: int = 13,
        num_classes: int = 4,
        encoder_weights: Literal["none"] = "none",
    ) -> None:
        super().__init__()
        if encoder_weights != "none":
            raise ValueError("Offline project: only encoder_weights='none' is supported")

        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.enc_conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.enc_pool = backbone.maxpool  # /4
        self.enc1 = backbone.layer1  # /4, 256
        self.enc2 = backbone.layer2  # /8, 512
        self.enc3 = backbone.layer3  # /16, 1024
        self.enc4 = backbone.layer4  # /32, 2048

        self.up4 = _UpBlock(2048, 1024, 1024)
        self.up3 = _UpBlock(1024, 512, 512)
        self.up2 = _UpBlock(512, 256, 256)
        self.up1 = _UpBlock(256, 64, 64)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc_conv1(x)        # /2
        x1 = self.enc1(self.enc_pool(x0))  # /4
        x2 = self.enc2(x1)            # /8
        x3 = self.enc3(x2)            # /16
        x4 = self.enc4(x3)            # /32

        d3 = self.up4(x4, x3)         # /16
        d2 = self.up3(d3, x2)         # /8
        d1 = self.up2(d2, x1)         # /4
        d0 = self.up1(d1, x0)         # /2
        d0 = self.up0(d0)             # /1
        return self.head(d0)

