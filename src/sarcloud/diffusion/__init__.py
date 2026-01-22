"""Diffusion utilities.

本模块提供两种扩散模型实现：
1. GaussianDiffusion: 标准 DDPM/DDIM 扩散模型（清晰→噪声→清晰）
2. ResidualShiftingDiffusion: 残差偏移扩散模型（有云→清晰）
"""

from .gaussian import GaussianDiffusion
from .residual_shifting import ResidualShiftingDiffusion
from .sampling import sample_batch, sample_with_progress
from .sampling_rs import sample_batch_rs, sample_with_progress_rs, sample_intermediate_rs

__all__ = [
    # 标准 DDPM
    "GaussianDiffusion",
    "sample_batch",
    "sample_with_progress",
    # Residual Shifting
    "ResidualShiftingDiffusion",
    "sample_batch_rs",
    "sample_with_progress_rs",
    "sample_intermediate_rs",
]
