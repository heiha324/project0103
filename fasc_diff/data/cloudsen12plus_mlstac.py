from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rasterio.io import MemoryFile
from torch.utils.data import Dataset


def _find_index_end(header: bytes) -> int:
    """
    MLSTAC files start with a small binary prefix, then a JSON index that maps
    sample_id -> [offset, length], followed by a concatenation of JP2 blobs.

    We locate the end of JSON by searching for the first `}` that is immediately
    followed by the JP2 signature box length `\\x00\\x00\\x00\\x0c`.
    """
    pat = b"}\x00\x00\x00\x0c"
    idx = header.find(pat)
    if idx < 0:
        raise ValueError("Failed to locate MLSTAC index end (pattern `}\\x00\\x00\\x00\\x0c` not found).")
    return idx


@dataclass(frozen=True)
class CloudSEN12PlusMLSTACConfig:
    band_indices: tuple[int, ...] = tuple(range(1, 14))  # 13 spectral bands
    label_band: int = 14  # CM1 (human label), 0..3 for high-quality labels
    clip_min: float = 0.0
    clip_max: float = 10000.0
    to_minus1_1: bool = True


class CloudSEN12PlusMLSTACDataset(Dataset[dict[str, torch.Tensor]]):
    """
    CloudSEN12+ `.mlstac` reader (p509/p2000).

    Each sample is stored as a multi-band JP2 (typically 15 bands):
      - Bands 1..13: Sentinel-2 spectral bands
      - Band 14: CM1 (human label) with classes {0,1,2,3} for high-quality labels
      - Band 15: CM2 (model label)

    This dataset returns:
      - image: (C,H,W) float32 (scaled to [0,1] then optionally mapped to [-1,1])
      - label: (H,W) int64
    """

    def __init__(
        self,
        mlstac_path: str | Path,
        *,
        cfg: CloudSEN12PlusMLSTACConfig = CloudSEN12PlusMLSTACConfig(),
        random_flip: bool = True,
        max_items: int | None = None,
    ) -> None:
        self.path = Path(mlstac_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.cfg = cfg
        self.random_flip = random_flip
        self.max_items = max_items

        self._base_offset, self._items = self._load_index(self.path)
        if self.max_items is not None:
            self._items = self._items[: int(self.max_items)]

        self._fh = None

    @staticmethod
    def _load_index(path: Path) -> tuple[int, list[tuple[str, int, int]]]:
        # Index is small (<~1MB). Read a bounded header window and parse JSON.
        scan_bytes = 4_000_000
        with path.open("rb") as f:
            head = f.read(scan_bytes)

        json_start = head.find(b"{")
        if json_start < 0:
            raise ValueError("Invalid MLSTAC: JSON index start `{` not found.")
        json_end = _find_index_end(head)
        index_bytes = head[json_start : json_end + 1]
        index: dict[str, Any] = json.loads(index_bytes.decode("utf-8"))

        base_offset = json_end + 1
        items: list[tuple[str, int, int]] = []
        for k, v in index.items():
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError(f"Invalid index entry for {k}: {v}")
            off, ln = int(v[0]), int(v[1])
            items.append((str(k), off, ln))
        return base_offset, items

    def __len__(self) -> int:
        return len(self._items)

    def _get_fh(self):
        if self._fh is None:
            self._fh = self.path.open("rb")
        return self._fh

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_id, off, ln = self._items[idx]
        fh = self._get_fh()
        fh.seek(self._base_offset + off)
        blob = fh.read(ln)

        with MemoryFile(blob) as mem:
            with mem.open() as src:
                img = src.read(list(self.cfg.band_indices)).astype(np.float32)
                y = src.read(self.cfg.label_band).astype(np.int64)

        # Scale S2 reflectance to [0,1] then optionally to [-1,1].
        img = np.clip(img, self.cfg.clip_min, self.cfg.clip_max)
        denom = float(self.cfg.clip_max - self.cfg.clip_min) if self.cfg.clip_max != self.cfg.clip_min else 1.0
        img = (img - float(self.cfg.clip_min)) / denom
        img = np.clip(img, 0.0, 1.0)
        if self.cfg.to_minus1_1:
            img = img * 2.0 - 1.0

        x = torch.from_numpy(img)
        label = torch.from_numpy(y)

        if self.random_flip:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])
                label = torch.flip(label, dims=[1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[1])
                label = torch.flip(label, dims=[0])

        return {"image": x, "label": label, "id": sample_id}

