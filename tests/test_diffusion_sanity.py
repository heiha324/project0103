"""Sanity checks for diffusion utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sarcloud.diffusion.gaussian import GaussianDiffusion
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion import sampling as sampling_module
from sarcloud.models.cond_unet import ConditionalUNet, ResBlock


class DiffusionSanityTests(unittest.TestCase):
    def _assert_time_sequence(self, seq: list[int], start: int, end: int) -> None:
        self.assertGreaterEqual(len(seq), 1)
        self.assertEqual(seq[0], start)
        self.assertEqual(seq[-1], end)
        self.assertEqual(len(seq), len(set(seq)))
        if len(seq) > 1:
            self.assertTrue(all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)))

    def test_p_sample_clamps_t0(self) -> None:
        diffusion = GaussianDiffusion(
            timesteps=10,
            schedule_type="linear",
            x0_clip_min=0.0,
            x0_clip_max=1.0,
        )
        x_t = torch.zeros((2, 1, 4, 4))
        eps = torch.full_like(x_t, 10.0)
        t = torch.zeros((2,), dtype=torch.long)
        out = diffusion.p_sample(x_t, t, eps)
        self.assertTrue(torch.all(out >= diffusion.x0_clip_min).item())
        self.assertTrue(torch.all(out <= diffusion.x0_clip_max).item())

    def test_time_sequence_helpers(self) -> None:
        diffusion = GaussianDiffusion(timesteps=10, schedule_type="linear")
        seq = diffusion.sample_timesteps(steps=50)
        self._assert_time_sequence(seq, 9, 0)

        rs_diffusion = ResidualShiftingDiffusion(timesteps=12)
        rs_seq = rs_diffusion.sample_timesteps(steps=50)
        self._assert_time_sequence(rs_seq, 11, 0)

        helper_seq = sampling_module._make_time_sequence(7, 0, 50)
        self._assert_time_sequence(helper_seq, 7, 0)

    def test_groupnorm_divisible_channels(self) -> None:
        ResBlock(48, 48, time_dim=32)
        ConditionalUNet(
            x_channels=4,
            y_channels=4,
            s_channels=2,
            base_channels=48,
            depth=1,
            time_dim=32,
        )


if __name__ == "__main__":
    unittest.main()
