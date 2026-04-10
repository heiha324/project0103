"""Distributed samplers used by training and evaluation scripts."""

from __future__ import annotations

from torch.utils.data import Sampler


class DistributedEvalSamplerNoPad(Sampler[int]):
    """Split evaluation samples across ranks without duplication or padding."""

    def __init__(self, dataset, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
