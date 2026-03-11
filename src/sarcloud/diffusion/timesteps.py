"""Timesteps utilities.

Generate strictly decreasing integer time sequences with no duplicates and
explicit endpoints for DDPM/DDIM sampling.
"""

from __future__ import annotations

from typing import List

import torch


def make_time_sequence(start_t: int, end_t: int, steps: int) -> List[int]:
    """Generate a strictly decreasing time sequence with endpoints.

    Args:
        start_t (int): Starting timestep (usually T-1).
        end_t (int): Ending timestep (usually 0).
        steps (int): Desired sampling steps.

    Returns:
        List[int]: Sequence including start_t and end_t.
    """
    start_t = int(start_t)
    end_t = int(end_t)
    steps = int(steps)
    if start_t < end_t:
        raise ValueError(f"start_t must be >= end_t, got {start_t} < {end_t}")

    total_steps = start_t - end_t + 1
    if total_steps <= 1:
        return [start_t]

    steps = max(2, steps)
    steps = min(steps, total_steps)

    t_seq = torch.linspace(start_t, end_t, steps, dtype=torch.float32)
    t_seq = torch.round(t_seq).long()
    t_seq = torch.unique_consecutive(t_seq, dim=0)
    seq_list = t_seq.tolist()
    if not seq_list:
        return [start_t, end_t]

    seq_list[0] = start_t
    if seq_list[-1] != end_t:
        seq_list.append(end_t)

    cleaned = [seq_list[0]]
    for val in seq_list[1:]:
        if val < cleaned[-1]:
            cleaned.append(val)

    if cleaned[-1] != end_t:
        if end_t < cleaned[-1]:
            cleaned.append(end_t)
        else:
            cleaned[-1] = end_t

    return cleaned
