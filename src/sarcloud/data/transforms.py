"""Simple paired augmentations for image/mask tensors."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def random_flip_rotate(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # image: CHW, mask: HW
    if random.random() < 0.5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=1)
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=0)
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(0, 1))
    return image.copy(), mask.copy()
