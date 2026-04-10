"""图像 IO 和归一化辅助工具 (Image IO and Normalization Helpers)。

本模块提供了：
1. 鲁棒的图像加载函数 (支持 .npy, .npz, .tif)。
2. 通道维度调整工具 (HWC -> CHW)。
3. 特定传感器的归一化逻辑 (Sentinel-1, Sentinel-2)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    """确保数组格式为 (Channels, Height, Width)。

    自动检测并纠正 HWC (Height, Width, Channels) 格式。
    注意：这是一个启发式方法，基于通道数通常较小 (<=13) 的假设。

    Args:
        arr (np.ndarray): 输入数组。

    Returns:
        np.ndarray: CHW 格式的数组。
    """
    if arr.ndim == 2:
        # 如果是 2D 数组 (H, W)，增加一个通道维度 -> (1, H, W)
        return arr[None, ...]
        
    if arr.ndim == 3:
        # 启发式判断: 如果第一个维度很小 (通常通道数 < 13)，则认为是 CHW
        # 如果最后一个维度很小且第一个维度很大，则认为是 HWC
        if arr.shape[0] in (1, 2, 3, 4, 6, 8, 13) and arr.shape[0] < arr.shape[-1]:
            return arr
        # 转换为 CHW: (H, W, C) -> (C, H, W)
        return np.transpose(arr, (2, 0, 1))
        
    raise ValueError(f"不支持的数组形状 (Unsupported array shape): {arr.shape}")


def load_array(path: str | Path) -> np.ndarray:
    """加载 Numpy 格式的数组 (.npy 或 .npz)。

    Args:
        path (str | Path): 文件路径。

    Returns:
        np.ndarray: 加载后的数组。
    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        # 自动提取第一个数组
        if "arr_0" in data:
            return data["arr_0"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError(f"在 {path} 中发现多个数组，请指定 key")
    raise ValueError(f"不支持的数组格式: {path}")


def load_tif(path: str | Path) -> np.ndarray:
    """加载 GeoTIFF 图像。

    尝试按顺序使用 rasterio -> tifffile -> imageio 加载，以最大化兼容性。
    自动转换为 CHW 格式。

    Args:
        path (str | Path): .tif 文件路径。

    Returns:
        np.ndarray: 图像数组 (CHW)。
    """
    path = Path(path)
    
    # 1. 优先使用 rasterio (地理空间标准库)
    try:
        import rasterio
    except Exception:
        rasterio = None

    if rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read() # rasterio 默认读取为 CHW
        return _ensure_chw(arr)

    # 2. 其次使用 tifffile (科学计算常用)
    try:
        import tifffile
    except Exception:
        tifffile = None

    if tifffile is not None:
        arr = tifffile.imread(path)
        return _ensure_chw(arr)

    # 3. 最后使用 imageio (通用图像库)
    try:
        import imageio.v3 as iio
    except Exception:
        import imageio as iio

    arr = iio.imread(path)
    return _ensure_chw(arr)


def _prepare_clip_value_numpy(
    value: float | Sequence[float] | np.ndarray,
    channels: int,
    name: str,
) -> np.ndarray | np.float32:
    """将归一化参数转换为 numpy 标量或 (C, 1, 1) 形状。"""
    if np.isscalar(value):
        return np.float32(value)

    value_arr = np.asarray(value, dtype=np.float32)
    if value_arr.ndim == 0:
        return np.float32(value_arr.item())

    flat = value_arr.reshape(-1)
    if flat.size == 1:
        return np.float32(flat.item())
    if flat.size != channels:
        raise ValueError(
            f"{name} 通道数不匹配: 期望 1 或 {channels} 个值, 实际 {flat.size}"
        )
    return flat.reshape(channels, 1, 1)


def _prepare_clip_value_torch(
    value: float | Sequence[float] | np.ndarray,
    channels: int,
    name: str,
    device: "torch.device",
) -> "torch.Tensor":
    """将归一化参数转换为 torch 标量或 (C, 1, 1) 形状。"""
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    elif np.isscalar(value):
        return torch.tensor(float(value), dtype=torch.float32, device=device)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)

    if tensor.ndim == 0:
        return tensor

    flat = tensor.reshape(-1)
    if flat.numel() == 1:
        return flat[0]
    if flat.numel() != channels:
        raise ValueError(
            f"{name} 通道数不匹配: 期望 1 或 {channels} 个值, 实际 {flat.numel()}"
        )
    return flat.view(channels, 1, 1)


def _validate_clip_range_numpy(denom: np.ndarray | np.float32) -> None:
    """检查 numpy 分母是否均为正值。"""
    flat = np.asarray(denom, dtype=np.float32).reshape(-1)
    invalid = np.where(flat <= 0)[0]
    if invalid.size == 0:
        return
    if flat.size == 1:
        raise ValueError("clip_max must be > clip_min")
    bad = ", ".join(str(int(idx)) for idx in invalid[:8])
    suffix = " ..." if invalid.size > 8 else ""
    raise ValueError(
        f"clip_max 与 clip_min 在以下通道不合法 (clip_max<=clip_min): {bad}{suffix}"
    )


def _validate_clip_range_torch(denom: "torch.Tensor") -> None:
    """检查 torch 分母是否均为正值。"""
    import torch

    flat = denom.reshape(-1)
    invalid = torch.nonzero(flat <= 0, as_tuple=False).flatten()
    if invalid.numel() == 0:
        return
    if flat.numel() == 1:
        raise ValueError("clip_max must be > clip_min")
    bad = ", ".join(str(int(idx)) for idx in invalid[:8].tolist())
    suffix = " ..." if invalid.numel() > 8 else ""
    raise ValueError(
        f"clip_max 与 clip_min 在以下通道不合法 (clip_max<=clip_min): {bad}{suffix}"
    )


def normalize_s2(
    arr: np.ndarray | "torch.Tensor",
    clip_min: float | Sequence[float] | np.ndarray = 0.0,
    clip_max: float | Sequence[float] | np.ndarray = 10000.0,
) -> np.ndarray | "torch.Tensor":
    """归一化 Sentinel-2 光学影像。

    Sentinel-2 L1C/L2A 数据通常存储为 uint16，数值范围 0-10000 代表反射率 0.0-1.0。
    本函数将其线性映射到 [0, 1] 区间。

    Args:
        arr: 原始数组，形状必须为 (C, H, W)，支持 np.ndarray 或 torch.Tensor。
        clip_min: 截断下限，可为标量或长度为 C 的序列。
        clip_max: 截断上限，可为标量或长度为 C 的序列。

    Returns:
        归一化后的浮点数组，范围 [0, 1]，类型与输入保持一致。
    """
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(arr, torch.Tensor):
        if arr.ndim != 3:
            raise ValueError(
                f"normalize_s2 期望输入为 (C, H, W)，实际收到形状: {tuple(arr.shape)}"
            )
        arr = arr.to(dtype=torch.float32)
        channels = int(arr.shape[0])
        clip_min_v = _prepare_clip_value_torch(clip_min, channels, "clip_min", arr.device)
        clip_max_v = _prepare_clip_value_torch(clip_max, channels, "clip_max", arr.device)
        denom = clip_max_v - clip_min_v
        _validate_clip_range_torch(denom)
        arr = torch.clamp(arr, min=clip_min_v, max=clip_max_v)
        return (arr - clip_min_v) / denom

    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"normalize_s2 仅支持 np.ndarray 或 torch.Tensor，实际类型: {type(arr)}"
        )
    if arr.ndim != 3:
        raise ValueError(f"normalize_s2 期望输入为 (C, H, W)，实际收到形状: {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    channels = int(arr.shape[0])
    clip_min_v = _prepare_clip_value_numpy(clip_min, channels, "clip_min")
    clip_max_v = _prepare_clip_value_numpy(clip_max, channels, "clip_max")
    denom = clip_max_v - clip_min_v
    _validate_clip_range_numpy(denom)
    arr = np.clip(arr, clip_min_v, clip_max_v)
    return (arr - clip_min_v) / denom


def normalize_s1_db(
    arr: np.ndarray,
    db_min: float = -25.0,
    db_max: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """归一化 Sentinel-1 SAR 影像 (从线性强度转 dB 再归一化)。

    1. 将线性强度值转换为分贝 (dB): 10 * log10(x)。
    2. 将 dB 值截断并归一化到 [0, 1]。

    Args:
        arr (np.ndarray): 线性强度值数组。
        db_min (float): dB 截断下限 (例如 -25 dB)。
        db_max (float): dB 截断上限 (例如 0 dB)。
        eps (float): 防止 log(0) 的小数值。

    Returns:
        np.ndarray: 归一化后的数组 [0, 1]。
    """
    arr = arr.astype(np.float32)
    # 转换为 dB
    db = 10.0 * np.log10(arr + eps)
    # 截断
    db = np.clip(db, db_min, db_max)
    # 归一化
    return (db - db_min) / (db_max - db_min)


def normalize_s1_db_values(
    arr: np.ndarray,
    db_min: float = -25.0,
    db_max: float = 0.0,
) -> np.ndarray:
    """归一化 Sentinel-1 SAR 影像 (输入已经是 dB 值)。

    直接对 dB 值进行截断和归一化。

    Args:
        arr (np.ndarray): dB 值数组。
        db_min, db_max: 归一化范围。

    Returns:
        np.ndarray: 归一化后的数组 [0, 1]。
    """
    arr = arr.astype(np.float32)
    arr = np.clip(arr, db_min, db_max)
    return (arr - db_min) / (db_max - db_min)


def select_bands(arr: np.ndarray, band_indices: Optional[Iterable[int]]) -> np.ndarray:
    """选择特定波段。

    Args:
        arr (np.ndarray): 输入数组 (C, H, W)。
        band_indices (list[int]): 需要保留的波段索引列表 (0-based)。

    Returns:
        np.ndarray: 筛选后的数组。
    """
    if band_indices is None:
        return arr
    band_indices = list(band_indices)
    if arr.ndim == 3:
        return arr[band_indices, ...]
    raise ValueError("波段选择需要 CHW 格式的数组 (Band selection expects CHW array)")


def to_numpy(tensor) -> np.ndarray:
    """辅助函数：将 PyTorch Tensor 转为 Numpy 数组 (CPU)。"""
    return tensor.detach().cpu().numpy()
