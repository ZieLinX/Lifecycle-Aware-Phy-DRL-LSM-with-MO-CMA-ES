from __future__ import annotations

import math
import unittest

import numpy as np

from config.cylinder_cfg import CylinderPhysicsCfg
from utils.hybrid_optimizer import build_chebyshev_basis, profile_volume, profiles_from_coefficients, run_hybrid_optimization


class HybridOptimizerTest(unittest.TestCase):
    def _small_cfg(self) -> CylinderPhysicsCfg:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_segments = 12
        cfg.num_rings = 10
        cfg.voltage_grid_points = 5
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 5
        cfg.thermal_max_iters = 60
        cfg.transient_max_time_s = 2.0
        cfg.transient_dt_s = 1.0
        return cfg

    def test_profile_projection_preserves_volume_and_endpoints(self) -> None:
        cfg = self._small_cfg()
        basis = build_chebyshev_basis(cfg.num_rings, 3)
        coeff = np.asarray([[0.10, -0.05, 0.08]], dtype=np.float64)
        profile = profiles_from_coefficients(cfg, coeff, basis)[0].detach().cpu().numpy()
        expected_volume = math.pi * cfg.radius * cfg.radius * cfg.height
        self.assertAlmostEqual(float(profile[0]), cfg.radius, places=8)
        self.assertAlmostEqual(float(profile[-1]), cfg.radius, places=8)
        self.assertLess(abs(profile_volume(profile, cfg.height) - expected_volume) / expected_volume, 5.0e-3)

    def test_hybrid_optimizer_smoke_returns_finite_best(self) -> None:
        cfg = self._small_cfg()
        result = run_hybrid_optimization(
            cfg,
            generations=1,
            population_size=4,
            elite_fraction=0.5,
            num_modes=3,
            local_iterations=0,
            seed=3,
        )
        self.assertEqual(result.best_profile.shape, (cfg.num_rings,))
        self.assertTrue(np.isfinite(result.best_profile).all())
        self.assertIn("score", result.best_metrics)
        self.assertGreaterEqual(result.candidate_count, 5)


if __name__ == "__main__":
    unittest.main()
