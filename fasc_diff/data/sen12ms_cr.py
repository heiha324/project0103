from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    # Heuristic: channels usually small (<= 13), H/W larger.
    if arr.shape[0] in (1, 2, 3, 4, 6, 8, 13) and arr.shape[0] < arr.shape[-1]:
        return arr
    return np.transpose(arr, (2, 0, 1))


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError(f"Multiple arrays in {path}, please store a single array")
    raise ValueError(f"Unsupported file format: {path}")


def load_tif(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        import rasterio
    except Exception:  # pragma: no cover
        rasterio = None
    if rasterio is None:
        raise RuntimeError("rasterio is required to read .tif files")
    with rasterio.open(path) as src:
        arr = src.read()
    return _ensure_chw(arr)


def select_bands(arr: np.ndarray, band_indices: Sequence[int] | None) -> np.ndarray:
    if band_indices is None:
        return arr
    if arr.ndim != 3:
        raise ValueError("Band selection expects CHW array")
    return arr[list(band_indices), ...]


def normalize_s2(arr: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.clip(arr, clip_min, clip_max)
    denom = clip_max - clip_min
    if denom <= 0:
        raise ValueError("clip_max must be > clip_min")
    return (arr - clip_min) / denom


def normalize_s1_db(arr: np.ndarray, db_min: float, db_max: float, eps: float = 1e-6) -> np.ndarray:
    arr = arr.astype(np.float32)
    db = 10.0 * np.log10(arr + eps)
    db = np.clip(db, db_min, db_max)
    return (db - db_min) / (db_max - db_min)


def normalize_s1_db_values(arr: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.clip(arr, db_min, db_max)
    return (arr - db_min) / (db_max - db_min)


def _to_minus1_1(x01: np.ndarray) -> np.ndarray:
    return x01 * 2.0 - 1.0


@dataclass(frozen=True)
class Sen12Sample:
    s1_path: Path
    s2_cloudy_path: Path
    s2_clear_path: Path
    alpha_path: Optional[Path] = None


class Sen12MSCRDataset(Dataset[dict[str, torch.Tensor]]):
    """
    SEN12MS-CR 预处理版加载器（目录结构: root/split/{s1,s2_cloudy,s2_clear}/*.npy）。

    返回字段：
      - sar: (2,H,W) float32 in [-1,1]
      - opt_cloudy: (C,H,W) float32 in [-1,1]
      - opt_clean: (C,H,W) float32 in [-1,1]
      - alpha: (1,H,W) float32 in [0,1] 或 None
      - relpath: cloudy 文件相对 root 的路径（用于对齐缓存）
      - id: 样本 stem
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        s1_subdir: str = "s1",
        s2_cloudy_subdir: str = "s2_cloudy",
        s2_clear_subdir: str = "s2_clear",
        alpha_subdir: str | None = None,
        image_ext: str = ".npy",
        bands: Sequence[int] | None = None,
        s2_clip_min: float = 0.0,
        s2_clip_max: float = 10000.0,
        s1_db_min: float = -25.0,
        s1_db_max: float = 0.0,
        s1_input: str = "linear",  # linear->db, or "db"
        to_minus1_1: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.s1_dir = self.root / split / s1_subdir
        self.s2_cloudy_dir = self.root / split / s2_cloudy_subdir
        self.s2_clear_dir = self.root / split / s2_clear_subdir

        self.alpha_dir: Path | None = None
        if alpha_subdir:
            alpha_path = Path(alpha_subdir)
            if alpha_path.is_absolute():
                self.alpha_dir = alpha_path
            else:
                candidate = self.root / split / alpha_subdir
                if candidate.exists():
                    self.alpha_dir = candidate
                else:
                    fallback = self.root / alpha_subdir
                    self.alpha_dir = fallback if fallback.exists() else Path(alpha_subdir)

        self.image_ext = image_ext
        self.bands = bands
        self.s2_clip_min = float(s2_clip_min)
        self.s2_clip_max = float(s2_clip_max)
        self.s1_db_min = float(s1_db_min)
        self.s1_db_max = float(s1_db_max)
        self.s1_input = str(s1_input).lower()
        if self.s1_input not in {"linear", "db"}:
            raise ValueError("s1_input must be 'linear' or 'db'")
        self.to_minus1_1 = bool(to_minus1_1)

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("未找到 SEN12MS-CR 样本，请检查数据集路径配置")

    def _build_samples(self) -> list[Sen12Sample]:
        cloudy_paths = sorted(self.s2_cloudy_dir.glob(f"*{self.image_ext}"))
        samples: list[Sen12Sample] = []
        for cloudy_path in cloudy_paths:
            stem = cloudy_path.stem
            s1_path = self.s1_dir / f"{stem}{self.image_ext}"
            clear_path = self.s2_clear_dir / f"{stem}{self.image_ext}"
            if not s1_path.exists() or not clear_path.exists():
                continue

            alpha_path = None
            if self.alpha_dir is not None:
                candidate = self.alpha_dir / f"{stem}{self.image_ext}"
                if candidate.exists():
                    alpha_path = candidate
            samples.append(Sen12Sample(s1_path=s1_path, s2_cloudy_path=cloudy_path, s2_clear_path=clear_path, alpha_path=alpha_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        s1 = _ensure_chw(load_array(sample.s1_path))
        s2_cloudy = _ensure_chw(load_array(sample.s2_cloudy_path))
        s2_clear = _ensure_chw(load_array(sample.s2_clear_path))

        s2_cloudy = select_bands(s2_cloudy, self.bands)
        s2_clear = select_bands(s2_clear, self.bands)

        s2_cloudy = normalize_s2(s2_cloudy, self.s2_clip_min, self.s2_clip_max)
        s2_clear = normalize_s2(s2_clear, self.s2_clip_min, self.s2_clip_max)
        if self.s1_input == "db":
            s1 = normalize_s1_db_values(s1, self.s1_db_min, self.s1_db_max)
        else:
            s1 = normalize_s1_db(s1, self.s1_db_min, self.s1_db_max)

        if self.to_minus1_1:
            s2_cloudy = _to_minus1_1(s2_cloudy)
            s2_clear = _to_minus1_1(s2_clear)
            s1 = _to_minus1_1(s1)

        alpha = None
        if sample.alpha_path is not None:
            a = load_array(sample.alpha_path).astype(np.float32)
            if a.max() > 1.0:
                a = a / 255.0
            if a.ndim == 2:
                a = a[None, ...]
            alpha = a

        sar_t = torch.from_numpy(s1).float()
        cloudy_t = torch.from_numpy(s2_cloudy).float()
        clear_t = torch.from_numpy(s2_clear).float()
        alpha_t = torch.from_numpy(alpha).float() if alpha is not None else None

        # NaN/Inf safety
        if torch.isnan(sar_t).any() or torch.isinf(sar_t).any():
            logger.warning(f"NaN/Inf in S1 {sample.s1_path}")
            sar_t = torch.nan_to_num(sar_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(cloudy_t).any() or torch.isinf(cloudy_t).any():
            logger.warning(f"NaN/Inf in S2 Cloudy {sample.s2_cloudy_path}")
            cloudy_t = torch.nan_to_num(cloudy_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(clear_t).any() or torch.isinf(clear_t).any():
            logger.warning(f"NaN/Inf in S2 Clear {sample.s2_clear_path}")
            clear_t = torch.nan_to_num(clear_t, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha_t is not None and (torch.isnan(alpha_t).any() or torch.isinf(alpha_t).any()):
            logger.warning(f"NaN/Inf in Alpha {sample.alpha_path}")
            alpha_t = torch.nan_to_num(alpha_t, nan=0.0, posinf=0.0, neginf=0.0)

        relpath = sample.s2_cloudy_path.relative_to(self.root).as_posix()
        return {
            "sar": sar_t,
            "opt_cloudy": cloudy_t,
            "opt_clean": clear_t,
            "alpha": alpha_t if alpha_t is not None else torch.tensor([], dtype=torch.float32),
            "has_alpha": torch.tensor([1 if alpha_t is not None else 0], dtype=torch.int64),
            "relpath": relpath,
            "id": sample.s2_cloudy_path.stem,
        }


@dataclass(frozen=True)
class Sen12RawSample:
    s1_path: Path
    s2_cloudy_path: Path
    s2_clear_path: Path
    alpha_path: Optional[Path] = None


class Sen12MSCRRawDataset(Dataset[dict[str, torch.Tensor]]):
    """
    SEN12MS-CR 原始版加载器（官方复杂目录 + 可选 split CSV）。

    读取 `.tif`，并按 dB 值归一化 SAR（参考 project0103 实现）。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        alpha_root: str | Path | None = None,
        split_csv: str | Path | None = None,
        split: str | None = None,
        bands: Sequence[int] | None = None,
        s2_clip_min: float = 0.0,
        s2_clip_max: float = 10000.0,
        s1_db_min: float = -25.0,
        s1_db_max: float = 0.0,
        alpha_ext: str = ".npy",
        roi_glob: str | None = None,
        to_minus1_1: bool = True,
    ) -> None:
        self.root = Path(root)
        self.alpha_root = Path(alpha_root) if alpha_root is not None else None
        self.split_csv = Path(split_csv) if split_csv is not None else None
        self.split = split
        self.bands = bands
        self.s2_clip_min = float(s2_clip_min)
        self.s2_clip_max = float(s2_clip_max)
        self.s1_db_min = float(s1_db_min)
        self.s1_db_max = float(s1_db_max)
        self.alpha_ext = alpha_ext
        self.roi_glob = roi_glob
        self.to_minus1_1 = bool(to_minus1_1)

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("未找到 SEN12MS-CR Raw 样本，请检查路径")

    def _resolve_pair_paths(self, cloudy_path: Path) -> tuple[Path, Path]:
        roi_cloudy = cloudy_path.parents[1].name
        subdir_cloudy = cloudy_path.parent.name
        filename_cloudy = cloudy_path.name

        roi_s1 = roi_cloudy.replace("s2_cloudy", "s1")
        roi_s2 = roi_cloudy.replace("s2_cloudy", "s2")
        subdir_s1 = subdir_cloudy.replace("s2_cloudy", "s1")
        subdir_s2 = subdir_cloudy.replace("s2_cloudy", "s2")
        filename_s1 = filename_cloudy.replace("s2_cloudy", "s1")
        filename_s2 = filename_cloudy.replace("s2_cloudy", "s2")

        s1_path = self.root / roi_s1 / subdir_s1 / filename_s1
        s2_path = self.root / roi_s2 / subdir_s2 / filename_s2
        return s1_path, s2_path

    def _load_split_csv(self) -> list[Path]:
        if self.split_csv is None:
            return []
        if not self.split_csv.exists():
            raise RuntimeError(f"Split CSV not found: {self.split_csv}")

        split_id_map = {"train": "1", "val": "2", "test": "3"}
        target_split_id = split_id_map.get(self.split, self.split) if self.split else None

        cloudy_paths: list[Path] = []
        with self.split_csv.open("r", encoding="utf-8", newline="") as f:
            first_line = f.readline().strip()
            f.seek(0)
            first_col = first_line.split(",")[0].strip()
            is_supplementary_format = first_col in ("1", "2", "3")

            if is_supplementary_format:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 5:
                        continue
                    row_split_id = row[0].strip()
                    filename = row[4].strip()
                    if target_split_id and row_split_id != target_split_id:
                        continue

                    parts = filename.replace(".tif", "").split("_")
                    if len(parts) < 4:
                        continue
                    scene, season, tile, patch = parts[0], parts[1], parts[2], parts[3]
                    roi_dir = f"{scene}_{season}_s2_cloudy"
                    subdir = f"s2_cloudy_{tile}"
                    full_filename = f"{scene}_{season}_s2_cloudy_{tile}_{patch}.tif"
                    cloudy_paths.append(self.root / roi_dir / subdir / full_filename)
            else:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise RuntimeError(f"CSV missing header: {self.split_csv}")
                for row in reader:
                    if self.split and row.get("split") != self.split:
                        continue
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
        return cloudy_paths

    def _build_samples(self) -> list[Sen12RawSample]:
        if self.split_csv is not None:
            cloudy_paths = self._load_split_csv()
        else:
            pattern = "ROIs*_s2_cloudy/s2_cloudy_*/*.tif"
            if self.roi_glob:
                pattern = f"{self.roi_glob}/s2_cloudy_*/*.tif"
            cloudy_paths = sorted(self.root.glob(pattern))

        samples: list[Sen12RawSample] = []
        for cloudy_path in cloudy_paths:
            s1_path, s2_path = self._resolve_pair_paths(cloudy_path)
            if not s1_path.exists() or not s2_path.exists():
                continue
            alpha_path = None
            if self.alpha_root is not None:
                rel = cloudy_path.relative_to(self.root)
                candidate = (self.alpha_root / rel).with_suffix(self.alpha_ext)
                if candidate.exists():
                    alpha_path = candidate
            samples.append(Sen12RawSample(s1_path=s1_path, s2_cloudy_path=cloudy_path, s2_clear_path=s2_path, alpha_path=alpha_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        s1 = load_tif(sample.s1_path)
        s2_cloudy = load_tif(sample.s2_cloudy_path)
        s2_clear = load_tif(sample.s2_clear_path)

        s2_cloudy = select_bands(s2_cloudy, self.bands)
        s2_clear = select_bands(s2_clear, self.bands)

        s2_cloudy = normalize_s2(s2_cloudy, self.s2_clip_min, self.s2_clip_max)
        s2_clear = normalize_s2(s2_clear, self.s2_clip_min, self.s2_clip_max)
        s1 = normalize_s1_db_values(s1, self.s1_db_min, self.s1_db_max)

        if self.to_minus1_1:
            s2_cloudy = _to_minus1_1(s2_cloudy)
            s2_clear = _to_minus1_1(s2_clear)
            s1 = _to_minus1_1(s1)

        alpha = None
        if sample.alpha_path is not None:
            a = load_array(sample.alpha_path).astype(np.float32)
            if a.max() > 1.0:
                a = a / 255.0
            if a.ndim == 2:
                a = a[None, ...]
            alpha = a

        sar_t = torch.from_numpy(s1).float()
        cloudy_t = torch.from_numpy(s2_cloudy).float()
        clear_t = torch.from_numpy(s2_clear).float()
        alpha_t = torch.from_numpy(alpha).float() if alpha is not None else None

        if torch.isnan(sar_t).any() or torch.isinf(sar_t).any():
            logger.warning(f"NaN/Inf in S1 {sample.s1_path}")
            sar_t = torch.nan_to_num(sar_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(cloudy_t).any() or torch.isinf(cloudy_t).any():
            logger.warning(f"NaN/Inf in S2 Cloudy {sample.s2_cloudy_path}")
            cloudy_t = torch.nan_to_num(cloudy_t, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(clear_t).any() or torch.isinf(clear_t).any():
            logger.warning(f"NaN/Inf in S2 Clear {sample.s2_clear_path}")
            clear_t = torch.nan_to_num(clear_t, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha_t is not None and (torch.isnan(alpha_t).any() or torch.isinf(alpha_t).any()):
            logger.warning(f"NaN/Inf in Alpha {sample.alpha_path}")
            alpha_t = torch.nan_to_num(alpha_t, nan=0.0, posinf=0.0, neginf=0.0)

        relpath = sample.s2_cloudy_path.relative_to(self.root).as_posix()
        return {
            "sar": sar_t,
            "opt_cloudy": cloudy_t,
            "opt_clean": clear_t,
            "alpha": alpha_t if alpha_t is not None else torch.tensor([], dtype=torch.float32),
            "has_alpha": torch.tensor([1 if alpha_t is not None else 0], dtype=torch.int64),
            "relpath": relpath,
            "id": sample.s2_cloudy_path.stem,
        }


def collate_sen12mscr(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch")

    sar = torch.stack([b["sar"] for b in batch], dim=0)
    opt_cloudy = torch.stack([b["opt_cloudy"] for b in batch], dim=0)
    opt_clean = torch.stack([b["opt_clean"] for b in batch], dim=0)

    has_alpha = torch.stack([b["has_alpha"] for b in batch], dim=0).view(-1)
    alpha_list = [b["alpha"] for b in batch]
    alpha = None
    if bool((has_alpha == 1).all()):
        alpha = torch.stack(alpha_list, dim=0)

    # Keep relpath/id as python objects outside collate return (not needed in training hotpath).
    return {
        "sar": sar,
        "opt_cloudy": opt_cloudy,
        "opt_clean": opt_clean,
        "alpha": alpha,
        "has_alpha": has_alpha,
        "relpath": [b["relpath"] for b in batch],  # type: ignore[list-item]
        "id": [b["id"] for b in batch],            # type: ignore[list-item]
    }

