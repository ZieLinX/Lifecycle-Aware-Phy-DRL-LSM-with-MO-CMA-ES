from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from config.cylinder_cfg import CylinderPhysicsCfg
from utils.full3d_optimizer import (
    Full3DUNetGNNPolicy,
    _apply_strategy_displacement,
    _apply_full3d_topology_action,
    build_full3d_initial_shape_from_action,
    build_baseline_full3d_geometry,
    assign_pareto_metrics,
    evaluate_feature_scales,
    evaluate_full3d_geometry,
    evaluate_lifecycle,
    evaluate_visibility,
    full3d_action_dim,
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
        cfg.full3d_fixed_voltage_v = None
        cfg.full3d_volume_tolerance_ratio = 1.0e-5
        cfg.full3d_action_axial_modes = 3
        cfg.full3d_action_circum_modes = 2
        cfg.full3d_action_cap_radial_modes = 3
        cfg.full3d_action_strategy_channels = 4
        cfg.full3d_global_shape_steps = 2
        cfg.full3d_global_step_m = 7.0e-4
        cfg.full3d_global_max_radius_m = 5.0e-3
        cfg.full3d_neural_policy_amplitude_m = 3.0e-5
        cfg.voltage_grid_points = 9
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 7
        cfg.thermal_max_iters = 80
        cfg.full3d_lifecycle_steps = 2
        cfg.full3d_visibility_rays = 8
        cfg.full3d_visibility_patch_limit = 8
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
        self.assertEqual(metrics["thermal_radiation_sink_temperature_k"], cfg.ambient_temp)
        self.assertEqual(metrics["electrode_boundary_temperature_k"], cfg.ambient_temp)
        self.assertEqual(metrics["electrode_voltage_drop_v"], 0.0)
        self.assertAlmostEqual(metrics["tungsten_voltage_v"], metrics["voltage_v"])
        self.assertGreater(metrics["contact_end_face_area_m2"], 0.0)
        self.assertGreater(metrics["free_radiating_surface_area_m2"], 0.0)
        self.assertAlmostEqual(metrics["free_surface_thermal_balance_area_m2"], metrics["free_radiating_surface_area_m2"])
        self.assertLess(metrics["free_radiating_surface_area_m2"], metrics["surface_area_m2"])
        self.assertTrue(metrics["thermal_converged"])

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

    def test_full3d_default_searches_rated_voltage(self) -> None:
        cfg = self._small_cfg()
        geom = project_full3d_geometry(cfg, build_baseline_full3d_geometry(cfg), math.pi * cfg.radius * cfg.radius * cfg.height)
        metrics = evaluate_full3d_geometry(cfg, geom)
        self.assertEqual(metrics["voltage_search_mode"], "rated_search")
        self.assertGreaterEqual(metrics["voltage_v"], cfg.min_voltage)
        self.assertLessEqual(metrics["voltage_v"], cfg.max_voltage)
        self.assertGreater(metrics["voltage_search_evaluations"], 1)
        self.assertGreater(metrics["voltage_v"], 0.20)
        self.assertLess(metrics["voltage_v"], 0.50)
        self.assertTrue(metrics["thermal_converged"])

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

    def test_contact_area_changes_with_end_face_footprint(self) -> None:
        cfg = self._small_cfg()
        baseline = project_full3d_geometry(
            cfg,
            build_baseline_full3d_geometry(cfg),
            math.pi * cfg.radius * cfg.radius * cfg.height,
        )
        base_metrics = evaluate_full3d_geometry(cfg, baseline)
        vertices = baseline.vertices.copy()
        lower_ids = baseline.lower_electrode_indices
        upper_ids = baseline.upper_electrode_indices
        vertices[lower_ids, :2] *= 0.75
        vertices[upper_ids, :2] *= 0.75
        shrunk = project_full3d_geometry(cfg, baseline, math.pi * cfg.radius * cfg.radius * cfg.height)
        shrunk = type(baseline)(
            vertices=vertices,
            faces=baseline.faces,
            side_indices=baseline.side_indices,
            lower_cap_indices=baseline.lower_cap_indices,
            upper_cap_indices=baseline.upper_cap_indices,
            lower_electrode_indices=baseline.lower_electrode_indices,
            upper_electrode_indices=baseline.upper_electrode_indices,
            num_rings=baseline.num_rings,
            num_segments=baseline.num_segments,
            cap_rings=baseline.cap_rings,
        )
        shrunk = project_full3d_geometry(cfg, shrunk, math.pi * cfg.radius * cfg.radius * cfg.height)
        changed = evaluate_full3d_geometry(cfg, shrunk)
        self.assertLess(changed["electrode_contact_area_m2"], base_metrics["electrode_contact_area_m2"])
        self.assertGreater(changed["missing_electrode_contact_area_m2"], base_metrics["missing_electrode_contact_area_m2"])
        self.assertGreater(changed["mean_temperature_k"], base_metrics["mean_temperature_k"])

    def test_explicit_topology_action_space_moves_side_and_caps(self) -> None:
        cfg = self._small_cfg()
        baseline = project_full3d_geometry(
            cfg,
            build_baseline_full3d_geometry(cfg),
            math.pi * cfg.radius * cfg.radius * cfg.height,
        )
        action = np.zeros(full3d_action_dim(cfg), dtype=np.float64)
        action[0] = 1.0
        side_dim = (
            cfg.full3d_action_strategy_channels
            * cfg.full3d_action_axial_modes
            * (1 + 2 * cfg.full3d_action_circum_modes)
        )
        action[side_dim] = 1.0
        moved = _apply_full3d_topology_action(cfg, baseline, action, float(cfg.max_depth))
        moved = project_full3d_geometry(cfg, moved, math.pi * cfg.radius * cfg.radius * cfg.height)
        side_ids = baseline.side_indices[1:-1].reshape(-1)
        side_delta = np.linalg.norm(moved.vertices[side_ids, :2] - baseline.vertices[side_ids, :2], axis=1)
        cap_ids = np.unique(baseline.lower_cap_indices[1:].reshape(-1))
        cap_ids = cap_ids[cap_ids >= 0]
        cap_delta = np.linalg.norm(moved.vertices[cap_ids, :2] - baseline.vertices[cap_ids, :2], axis=1)
        self.assertGreater(float(np.max(side_delta)), 1.0e-8)
        self.assertGreater(float(np.max(cap_delta)), 1.0e-8)
        self.assertLessEqual(_electrode_error_for_test(cfg, moved), cfg.full3d_electrode_tolerance_m)

    def test_global_initial_shape_action_is_not_local_cylinder_perturbation(self) -> None:
        cfg = self._small_cfg()
        target = math.pi * cfg.radius * cfg.radius * cfg.height
        baseline = project_full3d_geometry(cfg, build_baseline_full3d_geometry(cfg), target)
        action = np.zeros(full3d_action_dim(cfg), dtype=np.float64)
        circum_terms = 1 + 2 * cfg.full3d_action_circum_modes
        side_channel_dim = cfg.full3d_action_axial_modes * circum_terms
        action[0 * side_channel_dim + 1 * circum_terms] = 2.0
        action[2 * side_channel_dim + 0 * circum_terms] = 1.3
        if circum_terms > 2:
            action[3 * side_channel_dim + 2] = -1.2
        geom = build_full3d_initial_shape_from_action(cfg, action, target_volume=target)
        self.assertLess(abs(mesh_volume(geom) - target) / target, 1.0e-5)
        side_ids = baseline.side_indices.reshape(-1)
        delta = np.linalg.norm(geom.vertices[side_ids, :2] - baseline.vertices[side_ids, :2], axis=1)
        self.assertGreater(float(np.max(delta)), 2.0e-4)
        metrics = evaluate_full3d_geometry(cfg, geom, evaluate_full3d_geometry(cfg, baseline))
        self.assertIn("energy_conversion_efficiency_0_3um", metrics)
        self.assertLessEqual(metrics["voltage_v"], cfg.max_voltage)

    def test_full3d_optimizer_smoke(self) -> None:
        cfg = self._small_cfg()
        result = run_full3d_optimization(cfg, generations=1, population_size=2, seed=3)
        self.assertGreaterEqual(result.candidate_count, 3)
        self.assertIn("net_radiated_power_0k_sphere_w", result.best_metrics)
        self.assertIn("selection_reason_full3d", result.selection_diagnostics)
        self.assertEqual(result.selection_diagnostics["archive_candidate_count"], result.candidate_count)
        self.assertEqual(result.selection_diagnostics["voltage_search_mode"], "rated_search")
        self.assertEqual(result.selection_diagnostics["action_dim_full3d"], full3d_action_dim(cfg))
        self.assertIn("action_space_full3d", result.selection_diagnostics)
        self.assertIn("optimization_target_full3d", result.selection_diagnostics)
        self.assertIn("energy_conversion_efficiency_0_3um", result.best_metrics)
        self.assertLessEqual(result.best_metrics["volume_change_ratio_3d"], 5.0e-5)
        self.assertTrue(result.best_metrics["top_bottom_faces_variable"])

    def test_feature_scale_detects_contact_footprint_change(self) -> None:
        cfg = self._small_cfg()
        target = math.pi * cfg.radius * cfg.radius * cfg.height
        baseline = project_full3d_geometry(cfg, build_baseline_full3d_geometry(cfg), target)
        base = evaluate_feature_scales(cfg, baseline)
        vertices = baseline.vertices.copy()
        vertices[baseline.lower_electrode_indices, :2] *= 0.55
        vertices[baseline.upper_electrode_indices, :2] *= 0.55
        shrunk = type(baseline)(
            vertices=vertices,
            faces=baseline.faces,
            side_indices=baseline.side_indices,
            lower_cap_indices=baseline.lower_cap_indices,
            upper_cap_indices=baseline.upper_cap_indices,
            lower_electrode_indices=baseline.lower_electrode_indices,
            upper_electrode_indices=baseline.upper_electrode_indices,
            num_rings=baseline.num_rings,
            num_segments=baseline.num_segments,
            cap_rings=baseline.cap_rings,
        )
        changed = evaluate_feature_scales(cfg, shrunk)
        self.assertLess(changed["feature_lower_contact_equiv_diameter_m"], base["feature_lower_contact_equiv_diameter_m"])
        self.assertLess(changed["feature_min_neck_scale_m"], base["feature_min_neck_scale_m"])

    def test_visibility_ray_cast_penalizes_inward_notch(self) -> None:
        cfg = self._small_cfg()
        cfg.full3d_visibility_rays = 12
        cfg.full3d_visibility_patch_limit = 12
        baseline = project_full3d_geometry(
            cfg,
            build_baseline_full3d_geometry(cfg),
            math.pi * cfg.radius * cfg.radius * cfg.height,
        )
        base_factor, _, _ = evaluate_visibility(cfg, baseline)
        vertices = baseline.vertices.copy()
        notch_ids = baseline.side_indices[3:7, :4].reshape(-1)
        vertices[notch_ids, :2] *= 0.40
        notched = type(baseline)(
            vertices=vertices,
            faces=baseline.faces,
            side_indices=baseline.side_indices,
            lower_cap_indices=baseline.lower_cap_indices,
            upper_cap_indices=baseline.upper_cap_indices,
            lower_electrode_indices=baseline.lower_electrode_indices,
            upper_electrode_indices=baseline.upper_electrode_indices,
            num_rings=baseline.num_rings,
            num_segments=baseline.num_segments,
            cap_rings=baseline.cap_rings,
        )
        notched_factor, _, diagnostics = evaluate_visibility(cfg, notched)
        self.assertGreater(len(diagnostics), 0)
        self.assertLessEqual(notched_factor, base_factor + 0.15)

    def test_lifecycle_outputs_trace_and_average_power(self) -> None:
        cfg = self._small_cfg()
        cfg.full3d_objective_mode = "sota"
        geom = project_full3d_geometry(
            cfg,
            build_baseline_full3d_geometry(cfg),
            math.pi * cfg.radius * cfg.radius * cfg.height,
        )
        baseline_metrics = evaluate_full3d_geometry(cfg, geom, include_lifecycle=False)
        lifecycle, trace = evaluate_lifecycle(cfg, geom, baseline_metrics)
        self.assertGreaterEqual(len(trace), 1)
        self.assertIn("lifecycle_avg_escape_0_3um_w", lifecycle)
        self.assertIn("feature_failure_reason", lifecycle)

    def test_pareto_sort_prefers_nondominated_feasible_items(self) -> None:
        rows = [
            {"constraint_feasible_3d": True, "P0_escape_0_3um_w": 1.0, "lifetime_s_full3d": 1.0, "lifecycle_avg_escape_0_3um_w": 1.0, "escape_visibility_factor": 0.5},
            {"constraint_feasible_3d": True, "P0_escape_0_3um_w": 2.0, "lifetime_s_full3d": 2.0, "lifecycle_avg_escape_0_3um_w": 2.0, "escape_visibility_factor": 0.8},
            {"constraint_feasible_3d": False, "P0_escape_0_3um_w": 3.0, "lifetime_s_full3d": 3.0, "lifecycle_avg_escape_0_3um_w": 3.0, "escape_visibility_factor": 0.9},
        ]
        assign_pareto_metrics(rows)
        self.assertEqual(rows[1]["pareto_rank"], 0)
        self.assertGreater(rows[0]["pareto_rank"], rows[1]["pareto_rank"])
        self.assertGreater(rows[2]["pareto_rank"], rows[1]["pareto_rank"])

    def test_mocma_optimizer_smoke_has_sota_fields(self) -> None:
        cfg = self._small_cfg()
        cfg.full3d_objective_mode = "sota"
        cfg.full3d_optimizer = "mo-cmaes"
        result = run_full3d_optimization(cfg, generations=1, population_size=2, seed=4, optimizer="mo-cmaes", objective_mode="sota")
        self.assertIn("P0_escape_0_3um_w", result.best_metrics)
        self.assertIn("lifecycle_avg_escape_0_3um_w", result.best_metrics)
        self.assertIn("pareto_hypervolume", result.selection_diagnostics)
        self.assertEqual(result.selection_diagnostics["optimizer_full3d"], "mo-cmaes")


def _electrode_error_for_test(cfg, geometry) -> float:
    points = geometry.vertices[np.concatenate([geometry.lower_electrode_indices, geometry.upper_electrode_indices])]
    target_z = np.concatenate(
        [
            np.full(len(geometry.lower_electrode_indices), -0.5 * cfg.height),
            np.full(len(geometry.upper_electrode_indices), 0.5 * cfg.height),
        ]
    )
    return float(np.max(np.abs(points[:, 2] - target_z)))


if __name__ == "__main__":
    unittest.main()
