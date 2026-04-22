from __future__ import annotations

import unittest

import torch

from config.cylinder_cfg import CylinderPhysicsCfg
from utils.rated_condition import simulate_transient_trajectory


class TransientTest(unittest.TestCase):
    def test_temperature_rises_under_constant_voltage(self) -> None:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_rings = 12
        ring_radius = torch.full((cfg.num_rings,), float(cfg.radius), dtype=torch.float32)
        transient = simulate_transient_trajectory(cfg, ring_radius, 10.0, t_max=2.0, dt=0.5)
        temp_hist = transient["temperature_k"]
        band_power = transient["band_power_w"]
        self.assertGreaterEqual(float(torch.max(temp_hist[-1]).item()), float(torch.max(temp_hist[0]).item()))
        self.assertEqual(band_power.shape[0], 4)
        self.assertTrue(torch.isfinite(band_power).all().item())


if __name__ == "__main__":
    unittest.main()
