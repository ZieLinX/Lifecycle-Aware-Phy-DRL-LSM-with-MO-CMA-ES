from __future__ import annotations

import unittest

import torch

from config.cylinder_cfg import CylinderPhysicsCfg
from utils.rated_condition import simulate_transient_trajectory, summarize_transient_selection


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

    def test_transient_summary_selects_bounded_time(self) -> None:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_rings = 12
        cfg.transient_max_time_s = 2.0
        ring_radius = torch.full((cfg.num_rings,), float(cfg.radius), dtype=torch.float32)
        transient = simulate_transient_trajectory(cfg, ring_radius, 10.0, t_max=2.0, dt=0.5)
        summary = summarize_transient_selection(cfg, transient, baseline_power_w=1.0)
        self.assertGreaterEqual(float(summary["optimal_transient_time_s"].item()), 0.5)
        self.assertLessEqual(float(summary["optimal_transient_time_s"].item()), 2.0)
        self.assertTrue(torch.isfinite(summary["transient_objective"]).item())


if __name__ == "__main__":
    unittest.main()
