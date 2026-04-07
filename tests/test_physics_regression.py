from __future__ import annotations

import math
import unittest

import torch

from config.cylinder_cfg import CylinderPhysicsCfg
from envs.cylinder_env import CylinderPhysicsEnv
from utils.rated_condition import search_rated_condition


class PhysicsRegressionTest(unittest.TestCase):
    def _small_cfg(self) -> CylinderPhysicsCfg:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_segments = 16
        cfg.num_rings = 8
        cfg.search_depth_grid = (0.0, 0.35, 0.70)
        cfg.search_sigma_grid = (cfg.min_sigma, 0.5 * (cfg.min_sigma + cfg.max_sigma))
        cfg.voltage_grid_points = 7
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 5
        cfg.thermal_max_iters = 80
        return cfg

    def test_baseline_volume_matches_closed_form(self) -> None:
        cfg = self._small_cfg()
        env = CylinderPhysicsEnv(cfg)
        volume = env._compute_volume(env.rest_points)
        expected = math.pi * cfg.radius * cfg.radius * cfg.height
        self.assertLess(abs(volume - expected) / expected, 1.0e-4)

    def test_rated_condition_baseline_is_feasible_and_bounded(self) -> None:
        cfg = self._small_cfg()
        ring_radius = torch.full((cfg.num_rings,), float(cfg.radius), dtype=torch.float32)
        initial_volume = math.pi * cfg.radius * cfg.radius * cfg.height
        metrics = search_rated_condition(cfg, ring_radius, initial_volume)
        self.assertGreaterEqual(metrics.voltage_v, cfg.min_voltage)
        self.assertLessEqual(metrics.voltage_v, cfg.max_voltage)
        self.assertLess(metrics.volume_change_ratio, 1.0e-5)
        self.assertTrue(math.isfinite(metrics.max_temperature_k))
        self.assertGreater(metrics.thermal_iterations, 0)
        self.assertTrue(metrics.thermal_residual_k >= 0.0)


if __name__ == "__main__":
    unittest.main()
