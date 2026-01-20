"""SEN12MS-CR 数据集加载器 (带可选的 Alpha 缓存)。

该模块负责加载 SEN12MS-CR 多模态遥感数据集，支持：
1. 自动配对 SAR (Sentinel-1) 和 光学 (Sentinel-2) 影像。
2. 读取 .npy 或 .tif 格式的数据。
3. 应用数据归一化和波段选择。
4. 加载预计算的 Alpha 掩码 (可选)。
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from sarcloud.utils.image import (
    load_array,
    load_tif,
    normalize_s1_db,
    normalize_s1_db_values,
    normalize_s2,
    select_bands,
    _ensure_chw,
)

# 模块级 Logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sen12Sample:
    """单个数据样本的路径集合 (预处理好的 .npy 版本)。"""
    s1_path: Path          # Sentinel-1 SAR 数据路径
    s2_cloudy_path: Path   # Sentinel-2 有云光学图像路径
    s2_clear_path: Path    # Sentinel-2 无云光学图像路径 (Ground Truth)
    alpha_path: Optional[Path] = None  # Alpha 通道 (云掩码) 路径，可选


class Sen12MSCRDataset(Dataset):
    """SEN12MS-CR 数据集加载器 (针对 .npy 格式优化)。
    
    通常用于训练，因为读取 .npy 比 .tif 更快。
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str,
        s1_subdir: str = "s1",
        s2_cloudy_subdir: str = "s2_cloudy",
        s2_clear_subdir: str = "s2_clear",
        alpha_subdir: Optional[str] = None,
        image_ext: str = ".npy",
        bands: Optional[Sequence[int]] = None,
        s2_clip_min: float = 0.0,
        s2_clip_max: float = 10000.0,
        s1_db_min: float = -25.0,
        s1_db_max: float = 0.0,
    ) -> None:
        """
        Args:
            root: 数据集根目录。
            split: 数据划分 (例如 "train", "test", "val")。
            s1_subdir: SAR 数据子目录名。
            s2_cloudy_subdir: 有云数据子目录名。
            s2_clear_subdir: 无云数据子目录名。
            alpha_subdir: Alpha 掩码子目录名 (可选)。
            image_ext: 图像扩展名 (默认 ".npy")。
            bands: 需要选择的波段索引列表 (例如 [1, 2, 3] 对应 RGB)。
            s2_clip_min: Sentinel-2 归一化最小值 (通常为 0)。
            s2_clip_max: Sentinel-2 归一化最大值 (通常为 10000)。
            s1_db_min: Sentinel-1 dB 值截断下限。
            s1_db_max: Sentinel-1 dB 值截断上限。
        """
        self.root = Path(root)
        self.split = split
        self.s1_dir = self.root / split / s1_subdir
        self.s2_cloudy_dir = self.root / split / s2_cloudy_subdir
        self.s2_clear_dir = self.root / split / s2_clear_subdir
        
        # 处理 Alpha 目录路径
        if alpha_subdir:
            alpha_path = Path(alpha_subdir)
            if alpha_path.is_absolute():
                self.alpha_dir = alpha_path
            else:
                # 尝试多种可能的路径结构
                candidate = self.root / split / alpha_subdir
                if candidate.exists():
                    self.alpha_dir = candidate
                else:
                    fallback = self.root / alpha_subdir
                    self.alpha_dir = fallback if fallback.exists() else Path(alpha_subdir)
        else:
            self.alpha_dir = None
            
        self.image_ext = image_ext
        self.bands = bands
        self.s2_clip_min = s2_clip_min
        self.s2_clip_max = s2_clip_max
        self.s1_db_min = s1_db_min
        self.s1_db_max = s1_db_max

        # 构建样本列表
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("未找到 SEN12MS-CR 样本，请检查数据集路径配置")

    def _build_samples(self) -> List[Sen12Sample]:
        """扫描目录并配对所有相关联的文件。"""
        # 以有云图像为基准进行扫描
        cloudy_paths = sorted(self.s2_cloudy_dir.glob(f"*{self.image_ext}"))
        samples: List[Sen12Sample] = []
        
        for cloudy_path in cloudy_paths:
            stem = cloudy_path.stem
            # 假设文件名相同，只是目录不同
            s1_path = self.s1_dir / f"{stem}{self.image_ext}"
            clear_path = self.s2_clear_dir / f"{stem}{self.image_ext}"
            
            # 确保对应文件存在
            if not s1_path.exists() or not clear_path.exists():
                continue
                
            alpha_path = None
            if self.alpha_dir:
                candidate = self.alpha_dir / f"{stem}{self.image_ext}"
                if candidate.exists():
                    alpha_path = candidate
                    
            samples.append(Sen12Sample(s1_path, cloudy_path, clear_path, alpha_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """读取并预处理单个样本。
        
        返回:
            tuple: (s1_tensor, s2_cloudy_tensor, s2_clear_tensor, alpha_tensor)
            所有张量形状均为 (C, H, W)，值域归一化到 [0, 1] (S1 除外，见下)。
        """
        sample = self.samples[idx]
        
        # 1. 加载数据并确保是 (C, H, W) 格式
        s1 = _ensure_chw(load_array(sample.s1_path))
        s2_cloudy = _ensure_chw(load_array(sample.s2_cloudy_path))
        s2_clear = _ensure_chw(load_array(sample.s2_clear_path))

        # 2. 选择指定波段 (如果配置了 bands)
        s2_cloudy = select_bands(s2_cloudy, self.bands)
        s2_clear = select_bands(s2_clear, self.bands)

        # 3. 归一化
        # Sentinel-2: [0, 10000] -> [0, 1]
        s2_cloudy = normalize_s2(s2_cloudy, self.s2_clip_min, self.s2_clip_max)
        s2_clear = normalize_s2(s2_clear, self.s2_clip_min, self.s2_clip_max)
        # Sentinel-1: dB 值归一化 -> [0, 1]
        s1 = normalize_s1_db(s1, self.s1_db_min, self.s1_db_max)

        # 4. 处理 Alpha 通道 (如果存在)
        alpha = None
        if sample.alpha_path is not None:
            alpha = load_array(sample.alpha_path).astype(np.float32)
            # 确保 Alpha 在 [0, 1] 之间
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
            if alpha.ndim == 2:
                alpha = alpha[None, ...] # 增加通道维度

        # 5. 转为 PyTorch Tensor
        s1_t = torch.from_numpy(s1).float()
        s2_cloudy_t = torch.from_numpy(s2_cloudy).float()
        s2_clear_t = torch.from_numpy(s2_clear).float()
        alpha_t = torch.from_numpy(alpha).float() if alpha is not None else None

        # 6. NaN/Inf 安全检查 (重要!)
        # 如果数据中有损坏的值，将其替换为 0，防止训练崩溃
        if torch.isnan(s1_t).any() or torch.isinf(s1_t).any():
            logger.warning(f"NaN/Inf in S1 {sample.s1_path}")
            s1_t = torch.nan_to_num(s1_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(s2_cloudy_t).any() or torch.isinf(s2_cloudy_t).any():
            logger.warning(f"NaN/Inf in S2 Cloudy {sample.s2_cloudy_path}")
            s2_cloudy_t = torch.nan_to_num(s2_cloudy_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(s2_clear_t).any() or torch.isinf(s2_clear_t).any():
            logger.warning(f"NaN/Inf in S2 Clear {sample.s2_clear_path}")
            s2_clear_t = torch.nan_to_num(s2_clear_t, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha_t is not None and (torch.isnan(alpha_t).any() or torch.isinf(alpha_t).any()):
            logger.warning(f"NaN/Inf in Alpha {sample.alpha_path}")
            alpha_t = torch.nan_to_num(alpha_t, nan=0.0, posinf=0.0, neginf=0.0)

        return s1_t, s2_cloudy_t, s2_clear_t, alpha_t


@dataclass(frozen=True)
class Sen12RawSample:
    """单个数据样本的路径集合 (原始 .tif 版本)。"""
    s1_path: Path
    s2_cloudy_path: Path
    s2_clear_path: Path
    alpha_path: Optional[Path] = None


class Sen12MSCRRawDataset(Dataset):
    """SEN12MS-CR 原始数据集读取器 (针对官方 .tif 目录结构)。
    
    该类处理复杂的文件目录结构匹配，直接从原始 TIF 文件加载数据。
    """

    def __init__(
        self,
        root: str | Path,
        alpha_root: Optional[str | Path] = None,
        split_csv: Optional[str | Path] = None,
        split: Optional[str] = None,
        bands: Optional[Sequence[int]] = None,
        s2_clip_min: float = 0.0,
        s2_clip_max: float = 10000.0,
        s1_db_min: float = -25.0,
        s1_db_max: float = 0.0,
        alpha_ext: str = ".npy",
        roi_glob: Optional[str] = None,
    ) -> None:
        """
        Args:
            root: 数据集根目录。
            alpha_root: Alpha 缓存根目录。
            split_csv: 包含数据集划分信息的 CSV 文件路径。
            split: 要加载的划分 (例如 "train", "test")，需要与 CSV 中的列匹配。
            bands: 波段选择。
            s2_clip_min, s2_clip_max: S2 归一化范围。
            s1_db_min, s1_db_max: S1 归一化范围。
            alpha_ext: Alpha 文件扩展名。
            roi_glob: 用于过滤 ROI 的 Glob 模式 (调试用)。
        """
        self.root = Path(root)
        self.alpha_root = Path(alpha_root) if alpha_root is not None else None
        self.split_csv = Path(split_csv) if split_csv is not None else None
        self.split = split
        self.bands = bands
        self.s2_clip_min = s2_clip_min
        self.s2_clip_max = s2_clip_max
        self.s1_db_min = s1_db_min
        self.s1_db_max = s1_db_max
        self.alpha_ext = alpha_ext
        self.roi_glob = roi_glob

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("未找到 SEN12MS-CR Raw 样本，请检查路径")

    def _resolve_pair_paths(self, cloudy_path: Path) -> tuple[Path, Path]:
        """根据 cloudy 图像路径，解析对应的 S1 和 Clear 图像路径。
        
        官方数据集结构非常复杂，需要进行字符串替换来定位配对文件。
        """
        roi_cloudy = cloudy_path.parents[1].name  # ROI_xx
        subdir_cloudy = cloudy_path.parent.name   # s2_cloudy_xx
        filename_cloudy = cloudy_path.name        # ROI_xx_Patch_xx.tif
        
        # 路径替换逻辑
        roi_s1 = roi_cloudy.replace("s2_cloudy", "s1")
        roi_s2 = roi_cloudy.replace("s2_cloudy", "s2") # 注意：clear 数据有时在 s2 目录下
        subdir_s1 = subdir_cloudy.replace("s2_cloudy", "s1")
        subdir_s2 = subdir_cloudy.replace("s2_cloudy", "s2")
        filename_s1 = filename_cloudy.replace("s2_cloudy", "s1")
        filename_s2 = filename_cloudy.replace("s2_cloudy", "s2")

        s1_path = self.root / roi_s1 / subdir_s1 / filename_s1
        s2_path = self.root / roi_s2 / subdir_s2 / filename_s2
        return s1_path, s2_path

    def _load_split_csv(self) -> List[Path]:
        """从 CSV 文件加载指定划分 (train/val/test) 的文件列表。
        
        支持两种 CSV 格式:
        1. 有表头格式: split,cloudy_path,... (旧格式)
        2. 无表头格式: split_id,s1,s2_cloudFree,s2_cloudy,filename (SEN12MS-CR 官方补充格式)
           - split_id: 1=train, 2=val, 3=test
           - filename: ROIs{scene}_{season}_{tile}_p{patch}.tif
        """
        if self.split_csv is None:
            return []
        if not self.split_csv.exists():
            raise RuntimeError(f"未找到 Split CSV: {self.split_csv}")
        
        # 将 split 名称映射到数字 ID
        split_id_map = {"train": "1", "val": "2", "test": "3"}
        target_split_id = split_id_map.get(self.split, self.split) if self.split else None
            
        cloudy_paths: List[Path] = []
        with self.split_csv.open("r", encoding="utf-8", newline="") as f:
            # 先读取第一行判断是否有表头
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针
            
            # 检测是否为无表头的官方补充格式 (第一列是数字 1/2/3)
            first_col = first_line.split(",")[0].strip()
            is_supplementary_format = first_col in ("1", "2", "3")
            
            if is_supplementary_format:
                # 无表头格式: split_id,s1,s2_cloudFree,s2_cloudy,filename
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 5:
                        continue
                    row_split_id = row[0].strip()
                    filename = row[4].strip()
                    
                    # 过滤划分
                    if target_split_id and row_split_id != target_split_id:
                        continue
                    
                    # 从文件名解析路径
                    # 文件名格式: ROIs{scene}_{season}_{tile}_p{patch}.tif
                    # 例如: ROIs1158_spring_101_p675.tif
                    # 对应目录: ROIs1158_spring_s2_cloudy/s2_cloudy_101/ROIs1158_spring_s2_cloudy_101_p675.tif
                    parts = filename.replace(".tif", "").split("_")
                    if len(parts) >= 4:
                        # 解析: ROIs1158, spring, 101, p675
                        scene = parts[0]       # ROIs1158
                        season = parts[1]      # spring
                        tile = parts[2]        # 101
                        patch = parts[3]       # p675
                        
                        # 构建完整路径
                        roi_dir = f"{scene}_{season}_s2_cloudy"
                        subdir = f"s2_cloudy_{tile}"
                        full_filename = f"{scene}_{season}_s2_cloudy_{tile}_{patch}.tif"
                        cloudy_path = self.root / roi_dir / subdir / full_filename
                        cloudy_paths.append(cloudy_path)
            else:
                # 有表头格式 (旧格式)
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise RuntimeError(f"CSV 缺失表头: {self.split_csv}")
                for row in reader:
                    # 过滤划分
                    if self.split and row.get("split") != self.split:
                        continue
                    
                    # 尝试获取路径列
                    path_str = (
                        row.get("cloudy_path")
                        or row.get("cloudy_relpath")
                        or row.get("s2_cloudy_path")
                        or ""
                    )
                    if not path_str:
                        continue
                        
                    candidate = Path(path_str)
                    if not candidate.is_absolute():
                        candidate = self.root / candidate
                    cloudy_paths.append(candidate)
                
        if not cloudy_paths:
            raise RuntimeError(f"在 CSV {self.split_csv} 中未找到 split='{self.split}' 的样本")
        return cloudy_paths

    def _build_samples(self) -> List[Sen12RawSample]:
        """构建 Raw 数据样本列表。"""
        # 1. 获取所有 cloudy 图像路径
        if self.split_csv is not None:
            cloudy_paths = self._load_split_csv()
        else:
            # 如果没有 CSV，则遍历目录
            pattern = "ROIs*_s2_cloudy/s2_cloudy_*/*.tif"
            if self.roi_glob:
                pattern = f"{self.roi_glob}/s2_cloudy_*/*.tif"
            cloudy_paths = sorted(self.root.glob(pattern))
            
        # 2. 配对其他模态
        samples: List[Sen12RawSample] = []
        for cloudy_path in cloudy_paths:
            s1_path, s2_path = self._resolve_pair_paths(cloudy_path)
            
            if not s1_path.exists() or not s2_path.exists():
                # 文件缺失跳过
                continue
                
            alpha_path = None
            if self.alpha_root is not None:
                rel = cloudy_path.relative_to(self.root)
                alpha_path = (self.alpha_root / rel).with_suffix(self.alpha_ext)
                if not alpha_path.exists():
                    alpha_path = None
                    
            samples.append(Sen12RawSample(s1_path, cloudy_path, s2_path, alpha_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """加载原始 TIF 样本并归一化。"""
        sample = self.samples[idx]
        
        # 使用 load_tif 读取 .tif 文件
        s1 = load_tif(sample.s1_path)
        s2_cloudy = load_tif(sample.s2_cloudy_path)
        s2_clear = load_tif(sample.s2_clear_path)

        # 波段选择
        s2_cloudy = select_bands(s2_cloudy, self.bands)
        s2_clear = select_bands(s2_clear, self.bands)

        # 归一化
        s2_cloudy = normalize_s2(s2_cloudy, self.s2_clip_min, self.s2_clip_max)
        s2_clear = normalize_s2(s2_clear, self.s2_clip_min, self.s2_clip_max)
        # 注意：Raw 数据集这里用 normalize_s1_db_values
        # 假设输入已经是处理过的 dB 值或需要特定处理
        s1 = normalize_s1_db_values(s1, self.s1_db_min, self.s1_db_max)

        alpha = None
        if sample.alpha_path is not None:
            alpha = load_array(sample.alpha_path).astype(np.float32)
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
            if alpha.ndim == 2:
                alpha = alpha[None, ...]

        s1_t = torch.from_numpy(s1).float()
        s2_cloudy_t = torch.from_numpy(s2_cloudy).float()
        s2_clear_t = torch.from_numpy(s2_clear).float()
        alpha_t = torch.from_numpy(alpha).float() if alpha is not None else None

        # 安全检查
        if torch.isnan(s1_t).any() or torch.isinf(s1_t).any():
            logger.warning(f"NaN/Inf in S1 {sample.s1_path}")
            s1_t = torch.nan_to_num(s1_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(s2_cloudy_t).any() or torch.isinf(s2_cloudy_t).any():
            logger.warning(f"NaN/Inf in S2 Cloudy {sample.s2_cloudy_path}")
            s2_cloudy_t = torch.nan_to_num(s2_cloudy_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(s2_clear_t).any() or torch.isinf(s2_clear_t).any():
            logger.warning(f"NaN/Inf in S2 Clear {sample.s2_clear_path}")
            s2_clear_t = torch.nan_to_num(s2_clear_t, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha_t is not None and (torch.isnan(alpha_t).any() or torch.isinf(alpha_t).any()):
            logger.warning(f"NaN/Inf in Alpha {sample.alpha_path}")
            alpha_t = torch.nan_to_num(alpha_t, nan=0.0, posinf=0.0, neginf=0.0)

        return s1_t, s2_cloudy_t, s2_clear_t, alpha_t


def collate_sen12mscr(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """自定义 Batch 整理函数 (Collate Function)。
    
    处理可能为 None 的 Alpha 通道，将其正确堆叠。
    """
    if not batch:
        raise ValueError("Empty batch")
    sample = batch[0]
    
    # 兼容性处理：如果没有 Alpha，只有3个返回值
    if len(sample) == 3:
        s1, s2_cloudy, s2_clear = zip(*batch)
        return torch.stack(s1, dim=0), torch.stack(s2_cloudy, dim=0), torch.stack(s2_clear, dim=0), None
        
    if len(sample) != 4:
        raise ValueError(f"Unexpected sample size: {len(sample)}")
        
    s1_list, cloudy_list, clear_list, alpha_list = zip(*batch)
    
    # 只有当所有样本都有 Alpha 时才堆叠 Alpha
    alpha_batch = None
    if all(alpha is not None for alpha in alpha_list):
        alpha_batch = torch.stack([alpha for alpha in alpha_list if alpha is not None], dim=0)
        
    return (
        torch.stack(s1_list, dim=0),
        torch.stack(cloudy_list, dim=0),
        torch.stack(clear_list, dim=0),
        alpha_batch,
    )
