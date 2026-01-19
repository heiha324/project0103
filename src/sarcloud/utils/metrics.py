"""Metrics and losses for cloud detection and restoration."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def _flatten_binary(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = preds.view(-1)
    targets = targets.view(-1)
    return preds, targets


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return float((intersection + 1e-6) / (union + 1e-6))


def precision_recall(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return float(precision), float(recall)


def false_positive_rate(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    fp = (preds * (1 - targets)).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    return float((fp + 1e-6) / (fp + tn + 1e-6))


def overall_accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    tp = (preds * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    total = preds.numel()
    return float((tp + tn + 1e-6) / (total + 1e-6))


def f1_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (preds >= threshold).float()
    preds, targets = _flatten_binary(preds, targets)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return float((2 * precision * recall) / (precision + recall + 1e-6))


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < eps:
        return 100.0
    return float(20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse)))
