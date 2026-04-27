from __future__ import annotations

import numpy as np
import torch


def rsnet_pred_to_dual_mask(pred: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Map RSNet 4-class prediction to FASC-Diff dual-channel mask.

    RSNet classes:
      0: clear
      1: thick cloud   -> channel 0
      2: thin cloud    -> channel 1
      3: shadow        -> channel 1

    Returns:
      mask: (2,H,W) with {0,1}
    """
    if isinstance(pred, np.ndarray):
        if pred.ndim != 2:
            raise ValueError(f"Expected (H,W), got {pred.shape}")
        thick = (pred == 1).astype(np.uint8)
        thin = ((pred == 2) | (pred == 3)).astype(np.uint8)
        return np.stack([thick, thin], axis=0)

    if isinstance(pred, torch.Tensor):
        if pred.ndim != 2:
            raise ValueError(f"Expected (H,W), got {tuple(pred.shape)}")
        thick = (pred == 1).to(torch.uint8)
        thin = ((pred == 2) | (pred == 3)).to(torch.uint8)
        return torch.stack([thick, thin], dim=0)

    raise TypeError(f"Unsupported type: {type(pred)}")


def normalized_entropy(prob: np.ndarray | torch.Tensor, *, eps: float = 1e-8) -> np.ndarray | torch.Tensor:
    """
    Compute normalized entropy map in [0,1] from class probabilities.
    Expects prob shaped (K,H,W) for numpy or torch.
    """
    if isinstance(prob, np.ndarray):
        if prob.ndim != 3:
            raise ValueError(f"Expected (K,H,W), got {prob.shape}")
        k = prob.shape[0]
        p = np.clip(prob.astype(np.float32), eps, 1.0)
        ent = -np.sum(p * np.log(p), axis=0)
        return (ent / np.log(float(k))).astype(np.float32)

    if isinstance(prob, torch.Tensor):
        if prob.ndim != 3:
            raise ValueError(f"Expected (K,H,W), got {tuple(prob.shape)}")
        k = prob.shape[0]
        p = prob.to(dtype=torch.float32).clamp(min=eps, max=1.0)
        ent = -(p * torch.log(p)).sum(dim=0)
        return (ent / float(np.log(float(k)))).to(dtype=torch.float32)

    raise TypeError(f"Unsupported type: {type(prob)}")


def prob_to_m_soft(prob: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Soft semantic mask: M_soft = P_thick + P_thin from RSNet probabilities.
    Expects prob shaped (4,H,W) with class order: 0 clear, 1 thick, 2 thin, 3 shadow.
    """
    if isinstance(prob, np.ndarray):
        if prob.shape[0] < 3:
            raise ValueError(f"Expected prob with >=3 classes, got {prob.shape}")
        return (prob[1] + prob[2]).astype(np.float32)
    if isinstance(prob, torch.Tensor):
        if prob.shape[0] < 3:
            raise ValueError(f"Expected prob with >=3 classes, got {tuple(prob.shape)}")
        return (prob[1] + prob[2]).to(dtype=torch.float32)
    raise TypeError(f"Unsupported type: {type(prob)}")


def prob_to_p_thick(prob: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Thick-cloud probability map P_thick from RSNet probabilities.
    """
    if isinstance(prob, np.ndarray):
        return prob[1].astype(np.float32)
    if isinstance(prob, torch.Tensor):
        return prob[1].to(dtype=torch.float32)
    raise TypeError(f"Unsupported type: {type(prob)}")
