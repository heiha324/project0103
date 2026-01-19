"""Exponential moving average helper."""

from __future__ import annotations

from typing import Iterable

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = param.detach().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
