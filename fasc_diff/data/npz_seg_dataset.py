from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    # HWC -> CHW
    if x.shape[0] not in (1, 2, 3, 4, 13) and x.shape[-1] in (1, 2, 3, 4, 13):
        x = np.transpose(x, (2, 0, 1))
    return x


class NPZSegmentationDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Generic segmentation dataset backed by `.npz` files.

    Required keys per sample:
      - image: (C,H,W) or (H,W,C) float32, typically in [0,1]
      - label: (H,W) int64, class indices
    """

    def __init__(
        self,
        root: str | Path,
        *,
        pattern: str = "*.npz",
        image_key: str = "image",
        label_key: str = "label",
        random_flip: bool = True,
    ) -> None:
        self.root = Path(root)
        self.files = sorted(self.root.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched {pattern} under {self.root}")
        self.image_key = image_key
        self.label_key = label_key
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.files)

    def _load_npz(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as z:
            return dict(z.items())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.files[idx]
        z = self._load_npz(path)

        image = _to_chw(np.asarray(z[self.image_key], dtype=np.float32))
        label = np.asarray(z[self.label_key], dtype=np.int64)
        if label.ndim != 2:
            raise ValueError(f"label must be (H,W), got {label.shape} in {path}")

        img_t = torch.from_numpy(image)
        if img_t.min() >= 0.0 and img_t.max() <= 1.0:
            img_t = img_t * 2.0 - 1.0
        y_t = torch.from_numpy(label)

        if self.random_flip:
            if torch.rand(()) < 0.5:
                img_t = torch.flip(img_t, dims=[2])
                y_t = torch.flip(y_t, dims=[1])
            if torch.rand(()) < 0.5:
                img_t = torch.flip(img_t, dims=[1])
                y_t = torch.flip(y_t, dims=[0])

        return {"image": img_t, "label": y_t, "path": str(path)}

