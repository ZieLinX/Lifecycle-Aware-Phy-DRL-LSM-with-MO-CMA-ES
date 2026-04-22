from __future__ import annotations

import unittest

import torch

from config.cylinder_cfg import CylinderPhysicsCfg
from utils.feasibility import evaluate_feasibility


class FeasibilityTest(unittest.TestCase):
    def test_hourglass_profile_triggers_violation(self) -> None:
        cfg = CylinderPhysicsCfg()
        ring_radius = torch.tensor(
            [cfg.radius, cfg.radius, 0.25 * cfg.radius, 0.25 * cfg.radius, cfg.radius, cfg.radius],
            dtype=torch.float32,
        )
        report = evaluate_feasibility(cfg, ring_radius)
        self.assertFalse(report.feasible)
        self.assertGreater(report.soft_penalty, 0.0)
        self.assertLess(report.min_diameter_m, cfg.min_neck_diameter_m)


if __name__ == "__main__":
    unittest.main()
