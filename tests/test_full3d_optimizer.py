from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from config.cylinder_cfg import CylinderPhysicsCfg
from utils.full3d_optimizer import (
    Full3DUNetGNNPolicy,
    _apply_strategy_displacement,
    build_baseline_full3d_geometry,
    evaluate_full3d_geometry,
    mesh_volume,
    project_full3d_geometry,
    run_full3d_optimization,
)


class Full3DOptimizerTest(unittest.TestCase):
    def _small_cfg(self) -> CylinderPhysicsCfg:
        cfg = CylinderPhysicsCfg()
        cfg.device = "cpu"
        cfg.num_segments = 16
        cfg.num_rings = 10
        cfg.full3d_cap_rings = 4
        cfg.full3d_fixed_voltage_v = 100.0
        cfg.full3d_volume_tolerance_ratio = 1.0e-5
        cfg.thermal_max_iters = 20
        return cfg

    def test_baseline_closed_mesh_preserves_volume_and_electrodes(self) -> None:
        cfg = self._small_cfg()
        geom = build_baseline_full3d_geometry(cfg)
        target = math.pi * cfg.radius * cfg.radius * cfg.height
        projected = project_full3d_geometry(cfg, geom, target)
        self.assertLess(abs(mesh_volume(projected) - target) / target, 1.0e-5)
        metrics = evaluate_full3d_geometry(cfg, projected)
        self.assertLessEqual(metrics["electrode_max_error_m"], cfg.full3d_electrode_tolerance_m)
        self.assertEqual(metrics["external_sphere_temperature_k"], 0.0)
        self.assertEqual(metrics["external_sphere_emissivity"], 1.0)

    def test_full3d_temperature_responds_to_fixed_voltage(self) -> None:
        cfg = self._small_cfg()
        geom = project_full3d_geometry(cfg, build_baseline_full3d_geometry(cfg), math.pi * cfg.radius * cfg.radius * cfg.height)
        cfg.full3d_fixed_voltage_v = 20.0
        low_voltage = evaluate_full3d_geometry(cfg, geom)
        cfg.full3d_fixed_voltage_v = 100.0
        high_voltage = evaluate_full3d_geometry(cfg, geom)
        self.assertLess(low_voltage["max_temperature_k"], high_voltage["max_temperature_k"])
        self.assertLess(low_voltage["temperature_violation_ratio"], high_voltage["temperature_violation_ratio"])
        self.assertAlmostEqual(low_voltage["voltage_v"], 20.0)
        self.assertAlmostEqual(high_voltage["voltage_v"], 100.0)

    def test_unet_gnn_policy_smoke(self) -> None:
        model = Full3DUNetGNNPolicy()
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_full3d_action_can_move_cap_faces(self) -> None:
        cfg = self._small_cfg()
        baseline = project_full3d_geometry(
            cfg,
            build_baseline_full3d_geometry(cfg),
            math.pi * cfg.radius * cfg.radius * cfg.height,
        )
        strategy = np.ones((1, 3, 8, 16, 16), dtype=np.float32)
        moved = _apply_strategy_displacement(
            cfg,
            baseline,
            torch.as_tensor(strategy),
            float(cfg.max_depth),
        )
        cap_ids = np.unique(
            np.concatenate(
                [
                    moved.lower_cap_indices[1:].reshape(-1),
                    moved.upper_cap_indices[1:].reshape(-1),
                ]
            )
        )
        cap_ids = cap_ids[cap_ids >= 0]
        displacement = np.linalg.norm(moved.vertices[cap_ids] - baseline.vertices[cap_ids], axis=1)
        self.assertGreater(float(np.max(displacement)), 1.0e-8)

    def test_full3d_optimizer_smoke(self) -> None:
        cfg = self._small_cfg()
        result = run_full3d_optimization(cfg, generations=1, population_size=2, seed=3)
        self.assertGreaterEqual(result.candidate_count, 3)
        self.assertIn("net_radiated_power_0k_sphere_w", result.best_metrics)
        self.assertIn("selection_reason_full3d", result.selection_diagnostics)
        self.assertEqual(result.selection_diagnostics["archive_candidate_count"], result.candidate_count)
        self.assertLessEqual(result.best_metrics["volume_change_ratio_3d"], 5.0e-5)
        self.assertTrue(result.best_metrics["top_bottom_faces_variable"])


if __name__ == "__main__":
    unittest.main()
