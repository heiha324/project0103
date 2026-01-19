"""Cloud detection dataset utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
try:  # pragma: no cover - optional for DDP
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None
from torch.utils.data import Dataset, Sampler

from sarcloud.data.transforms import random_flip_rotate
from sarcloud.utils.image import load_array, normalize_s2, select_bands, _ensure_chw


@dataclass(frozen=True)
class CropRecord:
    image_path: Path
    mask_path: Path
    base_x: int
    base_y: int
    cloud_ratio: float


class CloudCropDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        images_subdir: str,
        masks_subdir: str,
        image_ext: str = ".npy",
        mask_ext: str = ".npy",
        crop_size: int = 256,
        base_stride: int = 128,
        jitter: int = 16,
        cloud_min_ratio: float = 0.01,
        cloud_keep_ratio: float = 0.2,
        bands: Optional[Sequence[int]] = None,
        s2_clip_min: float = 0.0,
        s2_clip_max: float = 10000.0,
        augment: bool = True,
        max_resample: int = 10,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / images_subdir
        self.masks_dir = self.root / masks_subdir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.crop_size = crop_size
        self.base_stride = base_stride
        self.jitter = jitter
        self.cloud_min_ratio = cloud_min_ratio
        self.cloud_keep_ratio = cloud_keep_ratio
        self.bands = bands
        self.s2_clip_min = s2_clip_min
        self.s2_clip_max = s2_clip_max
        self.augment = augment
        self.max_resample = max_resample

        self.records = self._build_records()
        if not self.records:
            raise RuntimeError("No crop records found; check dataset paths and extensions")

    def _build_records(self) -> List[CropRecord]:
        images = sorted(self.images_dir.glob(f"*{self.image_ext}"))
        records: List[CropRecord] = []
        for img_path in images:
            mask_path = self.masks_dir / (img_path.stem + self.mask_ext)
            if not mask_path.exists():
                continue
            mask = load_array(mask_path)
            if mask.ndim == 3:
                mask = mask.squeeze()
            if mask.max() > 1.0:
                mask = (mask > 0).astype(np.float32)
            h, w = mask.shape[-2:]
            for base_y in (0, self.base_stride):
                for base_x in (0, self.base_stride):
                    if base_y + self.crop_size > h or base_x + self.crop_size > w:
                        continue
                    crop = mask[base_y : base_y + self.crop_size, base_x : base_x + self.crop_size]
                    ratio = float(crop.mean())
                    records.append(CropRecord(img_path, mask_path, base_x, base_y, ratio))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _random_crop(self, record: CropRecord) -> Tuple[np.ndarray, np.ndarray]:
        mask = load_array(record.mask_path)
        image = load_array(record.image_path)
        if mask.ndim == 3:
            mask = mask.squeeze()
        if mask.max() > 1.0:
            mask = (mask > 0).astype(np.float32)
        image = _ensure_chw(image)
        image = select_bands(image, self.bands)
        image = normalize_s2(image, clip_min=self.s2_clip_min, clip_max=self.s2_clip_max)

        h, w = mask.shape[-2:]
        for _ in range(self.max_resample):
            dx = random.randint(-self.jitter, self.jitter)
            dy = random.randint(-self.jitter, self.jitter)
            x = max(0, min(record.base_x + dx, w - self.crop_size))
            y = max(0, min(record.base_y + dy, h - self.crop_size))
            crop_img = image[:, y : y + self.crop_size, x : x + self.crop_size]
            crop_mask = mask[y : y + self.crop_size, x : x + self.crop_size]
            cloud_ratio = float(crop_mask.mean())
            if cloud_ratio >= self.cloud_min_ratio or random.random() < self.cloud_keep_ratio:
                return crop_img, crop_mask
        return crop_img, crop_mask

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image, mask = self._random_crop(record)
        if self.augment:
            image, mask = random_flip_rotate(image, mask)
        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return image_t, mask_t


class WHUOriCropDataset(CloudCropDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        level_subdir: str = "level1_10m",
        image_ext: str = ".npy",
        mask_ext: str = ".npy",
        crop_size: int = 256,
        base_stride: int = 128,
        jitter: int = 16,
        cloud_min_ratio: float = 0.01,
        cloud_keep_ratio: float = 0.2,
        bands: Optional[Sequence[int]] = None,
        s2_clip_min: float = 0.0,
        s2_clip_max: float = 1.0,
        augment: bool = True,
        max_resample: int = 10,
    ) -> None:
        self.level_subdir = level_subdir
        images_subdir = f"{split}/Img"
        masks_subdir = f"{split}/Mask"
        super().__init__(
            root=root,
            images_subdir=images_subdir,
            masks_subdir=masks_subdir,
            image_ext=image_ext,
            mask_ext=mask_ext,
            crop_size=crop_size,
            base_stride=base_stride,
            jitter=jitter,
            cloud_min_ratio=cloud_min_ratio,
            cloud_keep_ratio=cloud_keep_ratio,
            bands=bands,
            s2_clip_min=s2_clip_min,
            s2_clip_max=s2_clip_max,
            augment=augment,
            max_resample=max_resample,
        )

    def _build_records(self) -> List[CropRecord]:
        pattern = f"*/{self.level_subdir}/*{self.image_ext}"
        images = sorted(self.images_dir.glob(pattern))
        records: List[CropRecord] = []
        for img_path in images:
            scene = img_path.parents[1].name
            mask_path = self.masks_dir / scene / f"{img_path.stem}{self.mask_ext}"
            if not mask_path.exists():
                continue
            mask = load_array(mask_path)
            if mask.ndim == 3:
                mask = mask.squeeze()
            if mask.max() > 1.0:
                mask = (mask > 0).astype(np.float32)
            h, w = mask.shape[-2:]
            for base_y in (0, self.base_stride):
                for base_x in (0, self.base_stride):
                    if base_y + self.crop_size > h or base_x + self.crop_size > w:
                        continue
                    crop = mask[base_y : base_y + self.crop_size, base_x : base_x + self.crop_size]
                    ratio = float(crop.mean())
                    records.append(CropRecord(img_path, mask_path, base_x, base_y, ratio))
        return records


class CloudBucketSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: CloudCropDataset,
        batch_size: int,
        heavy_ratio: float = 0.4,
        light_ratio: float = 0.4,
        clear_ratio: float = 0.2,
        heavy_thresh: float = 0.10,
        clear_thresh: float = 0.01,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.heavy_ratio = heavy_ratio
        self.light_ratio = light_ratio
        self.clear_ratio = clear_ratio
        self.heavy_thresh = heavy_thresh
        self.clear_thresh = clear_thresh

        self.heavy = [i for i, rec in enumerate(dataset.records) if rec.cloud_ratio >= heavy_thresh]
        self.light = [
            i
            for i, rec in enumerate(dataset.records)
            if clear_thresh <= rec.cloud_ratio < heavy_thresh
        ]
        self.clear = [i for i, rec in enumerate(dataset.records) if rec.cloud_ratio < clear_thresh]
        if not any([self.heavy, self.light, self.clear]):
            raise RuntimeError("No samples found for any cloud bucket")

    def __len__(self) -> int:
        return max(1, len(self.dataset) // self.batch_size)

    def _sample_bucket(self, bucket: List[int], num: int) -> List[int]:
        if not bucket:
            return []
        return [random.choice(bucket) for _ in range(num)]

    def __iter__(self):
        for _ in range(len(self)):
            heavy_n = int(self.batch_size * self.heavy_ratio)
            light_n = int(self.batch_size * self.light_ratio)
            clear_n = self.batch_size - heavy_n - light_n

            indices = []
            indices += self._sample_bucket(self.heavy, heavy_n)
            indices += self._sample_bucket(self.light, light_n)
            indices += self._sample_bucket(self.clear, clear_n)
            random.shuffle(indices)
            yield indices


class DistributedCloudBucketSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: CloudCropDataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
        heavy_ratio: float = 0.4,
        light_ratio: float = 0.4,
        clear_ratio: float = 0.2,
        heavy_thresh: float = 0.10,
        clear_thresh: float = 0.01,
    ) -> None:
        if num_replicas is None or rank is None:
            if dist is not None and dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.heavy_ratio = heavy_ratio
        self.light_ratio = light_ratio
        self.clear_ratio = clear_ratio
        self.heavy_thresh = heavy_thresh
        self.clear_thresh = clear_thresh

        self.heavy = [i for i, rec in enumerate(dataset.records) if rec.cloud_ratio >= heavy_thresh]
        self.light = [
            i
            for i, rec in enumerate(dataset.records)
            if clear_thresh <= rec.cloud_ratio < heavy_thresh
        ]
        self.clear = [i for i, rec in enumerate(dataset.records) if rec.cloud_ratio < clear_thresh]
        if not any([self.heavy, self.light, self.clear]):
            raise RuntimeError("No samples found for any cloud bucket")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return max(1, len(self.dataset) // (self.batch_size * self.num_replicas))

    def _sample_bucket(self, bucket: List[int], num: int, rng: random.Random) -> List[int]:
        if not bucket:
            return []
        return [rng.choice(bucket) for _ in range(num)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000 + self.rank)
        for _ in range(len(self)):
            heavy_n = int(self.batch_size * self.heavy_ratio)
            light_n = int(self.batch_size * self.light_ratio)
            clear_n = self.batch_size - heavy_n - light_n

            indices = []
            indices += self._sample_bucket(self.heavy, heavy_n, rng)
            indices += self._sample_bucket(self.light, light_n, rng)
            indices += self._sample_bucket(self.clear, clear_n, rng)
            rng.shuffle(indices)
            yield indices
