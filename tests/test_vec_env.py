from __future__ import annotations

import unittest

import numpy as np

from config.cylinder_cfg import CylinderPhysicsCfg
from envs.cylinder_vec_env import CylinderVecEnv


class VecEnvTest(unittest.TestCase):
    def test_vec_env_step_shapes_are_consistent(self) -> None:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_rings = 12
        cfg.max_steps = 2
        env = CylinderVecEnv(cfg, num_envs=4)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 4)
        actions = np.zeros((4, env.action_dim), dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertEqual(reward.shape, (4,))
        self.assertEqual(terminated.shape, (4,))
        self.assertEqual(truncated.shape, (4,))
        self.assertIn("score", info)
        self.assertIn("optimal_transient_time_s", info)
        self.assertIn("policy_dwell_time_s", info)

    def test_lifetime_ratio_observation_is_capped(self) -> None:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_rings = 12
        cfg.thermal_max_iters = 8
        cfg.voltage_grid_points = 3
        cfg.voltage_refine_levels = 0
        cfg.observation_lifetime_ratio_cap = 2.0
        env = CylinderVecEnv(cfg, num_envs=1)
        obs, _ = env.reset()
        env.current_metrics["lifetime_s"] = env.baseline_metrics["lifetime_s"] * 100.0

        obs = env._build_obs()

        lifetime_ratio_obs = obs[0, env.num_rings * 3 + 4]
        self.assertLessEqual(lifetime_ratio_obs, 2.0)


if __name__ == "__main__":
    unittest.main()
