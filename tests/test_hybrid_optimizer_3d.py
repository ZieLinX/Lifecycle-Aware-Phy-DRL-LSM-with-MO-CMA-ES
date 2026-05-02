from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from config.cylinder_cfg import CylinderPhysicsCfg
from utils.hybrid_optimizer_3d import build_3d_basis, fields_from_coefficients, radius_field_volume, run_hybrid_3d_optimization
from utils.rated_condition import _evaluate_voltage_3d_batch
from utils.rated_condition import search_rated_condition_3d_batch


class Hybrid3DOptimizerTest(unittest.TestCase):
    def _small_cfg(self) -> CylinderPhysicsCfg:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_segments = 10
        cfg.num_rings = 8
        cfg.voltage_grid_points = 5
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 5
        cfg.thermal_max_iters = 40
        cfg.transient_max_time_s = 2.0
        cfg.transient_dt_s = 1.0
        return cfg

    def test_radius_field_projection_preserves_endcaps_and_volume(self) -> None:
        cfg = self._small_cfg()
        basis = build_3d_basis(cfg.num_rings, cfg.num_segments, axial_modes=2, circum_modes=1)
        coeff = np.zeros((1, basis.shape[0]), dtype=np.float64)
        coeff[0, 0] = 0.05
        coeff[0, 1] = -0.03
        field = fields_from_coefficients(cfg, coeff, basis)[0].detach().cpu().numpy()
        self.assertTrue(np.allclose(field[0], cfg.radius, atol=1.0e-10))
        self.assertTrue(np.allclose(field[-1], cfg.radius, atol=1.0e-10))
        expected = math.pi * cfg.radius * cfg.radius * cfg.height
        vol = radius_field_volume(field, cfg.height)
        self.assertLess(abs(vol - expected) / expected, 8.0e-3)

    def test_rated_condition_3d_returns_finite_metrics(self) -> None:
        cfg = self._small_cfg()
        r = np.full((cfg.num_rings, cfg.num_segments), cfg.radius, dtype=np.float32)
        # Make a small non-axisymmetric ripple away from the endcaps.
        if cfg.num_segments > 2 and cfg.num_rings > 3:
            r[2:-2, 1] *= 1.02
            r[2:-2, 3] *= 0.98
        field_t = torch.as_tensor(r, dtype=torch.float32)
        initial_volume = math.pi * cfg.radius * cfg.radius * cfg.height
        metrics = search_rated_condition_3d_batch(cfg, field_t, initial_volume)
        self.assertTrue(torch.isfinite(metrics["voltage_v"]).all().item())
        self.assertTrue(torch.isfinite(metrics["initial_net_band_power_w"]).all().item())
        # squeezed input returns scalar tensors (matching existing 2D APIs).
        self.assertEqual(tuple(metrics["feasible"].shape), ())

    def test_3d_physics_counts_slope_surface_area(self) -> None:
        cfg = self._small_cfg()
        cfg.shadow_slope_coeff = 0.0
        cfg.shadow_roughness_coeff = 0.0
        z = torch.linspace(-1.0, 1.0, cfg.num_rings, dtype=torch.float32)
        theta = torch.linspace(0.0, 2.0 * math.pi, cfg.num_segments + 1, dtype=torch.float32)[:-1]
        ripple = 0.05 * torch.sin(math.pi * (z + 1.0)).unsqueeze(1) * torch.cos(2.0 * theta).unsqueeze(0)
        field = torch.full((cfg.num_rings, cfg.num_segments), float(cfg.radius), dtype=torch.float32) * (1.0 + ripple)
        field[0, :] = float(cfg.radius)
        field[-1, :] = float(cfg.radius)
        initial_volume = math.pi * cfg.radius * cfg.radius * cfg.height
        metrics = _evaluate_voltage_3d_batch(cfg, field, initial_volume, voltage_v=1.0)
        self.assertGreater(metrics["surface_area_ratio_3d"].item(), 1.0)

    def test_hybrid_3d_smoke_returns_finite_best(self) -> None:
        cfg = self._small_cfg()
        result = run_hybrid_3d_optimization(
            cfg,
            generations=1,
            population_size=4,
            elite_fraction=0.5,
            axial_modes=2,
            circum_modes=1,
            seed=2,
        )
        self.assertEqual(result.best_field.shape, (cfg.num_rings, cfg.num_segments))
        self.assertTrue(np.isfinite(result.best_field).all())
        self.assertIn("score", result.best_metrics)
        self.assertGreaterEqual(result.candidate_count, 5)


if __name__ == "__main__":
    unittest.main()

