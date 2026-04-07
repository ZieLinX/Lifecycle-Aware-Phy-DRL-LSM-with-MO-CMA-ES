from __future__ import annotations

import math
import unittest

import numpy as np

from config.cylinder_cfg import CylinderPhysicsCfg
from envs.cylinder_env import CylinderPhysicsEnv
from utils.planner import plan_action


class PlannerTest(unittest.TestCase):
    def test_planner_returns_finite_action(self) -> None:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_segments = 12
        cfg.num_rings = 6
        cfg.max_steps = 2
        cfg.search_depth_grid = (0.0, 0.35, 0.70)
        cfg.search_sigma_grid = (cfg.min_sigma, 0.5 * (cfg.min_sigma + cfg.max_sigma))
        cfg.planner_horizon = 1
        cfg.planner_beam_width = 2
        cfg.planner_seed_top_k = 4
        cfg.planner_candidate_top_k = 2
        cfg.planner_local_refine_top_k = 1
        cfg.voltage_grid_points = 5
        cfg.voltage_refine_levels = 1
        env = CylinderPhysicsEnv(cfg)
        env.reset()

        decision = plan_action(env)
        self.assertEqual(decision.action.shape, (3,))
        self.assertTrue(np.all(np.isfinite(decision.action)))
        self.assertTrue(math.isfinite(decision.projected_return))

        reward, _, info = env.evaluate_action(decision.action)
        self.assertTrue(math.isfinite(reward))
        self.assertLessEqual(float(info["volume_change_ratio"]), cfg.volume_tolerance_ratio + 0.2)


if __name__ == "__main__":
    unittest.main()
