from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from config.cylinder_cfg import make_training_cfg
from utils.full3d_optimizer import (
    export_full3d_animation,
    export_full3d_mesh,
    run_full3d_optimization,
    save_full3d_history_csv,
    save_full3d_summary,
    write_full3d_report,
)


def _configure_cfg(args):
    cfg = make_training_cfg()
    cfg.device = args.device
    cfg.thermal_max_iters = int(args.thermal_iters)
    cfg.freecad_timeout_s = float(args.freecad_timeout)
    cfg.max_voltage = float(args.max_voltage)
    cfg.full3d_use_neural_policy = bool(args.full3d_neural_policy)
    cfg.full3d_fixed_voltage_v = None if args.fixed_voltage is None else float(args.fixed_voltage)
    cfg.full3d_volume_tolerance_ratio = float(args.full3d_volume_tolerance)
    cfg.full3d_cap_rings = int(args.cap_rings)
    if args.action_axial_modes is not None:
        cfg.full3d_action_axial_modes = int(args.action_axial_modes)
    if args.action_circum_modes is not None:
        cfg.full3d_action_circum_modes = int(args.action_circum_modes)
    if args.action_cap_radial_modes is not None:
        cfg.full3d_action_cap_radial_modes = int(args.action_cap_radial_modes)
    if args.cem_initial_sigma is not None:
        cfg.full3d_cem_initial_sigma = float(args.cem_initial_sigma)
    if args.cem_elite_fraction is not None:
        cfg.full3d_cem_elite_fraction = float(args.cem_elite_fraction)
    if args.smoke:
        cfg.num_segments = 32
        cfg.num_rings = 24
        cfg.full3d_cap_rings = min(int(cfg.full3d_cap_rings), 5)
        cfg.voltage_grid_points = 5
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 5
        cfg.thermal_max_iters = min(int(args.thermal_iters), 240)
    return cfg


def _build_summary(args, cfg, result, export_info, anim_info) -> dict[str, object]:
    base_power = max(float(result.baseline_metrics.get("net_radiated_power_0k_sphere_w", 0.0)), 1.0e-9)
    final_power = float(result.best_metrics.get("net_radiated_power_0k_sphere_w", 0.0))
    return {
        "method": "full3d_unet_gnn_policy_closed_mesh_300k_sink",
        "backend": "full3d",
        "device": args.device,
        "candidate_count": int(result.candidate_count),
        "num_segments": int(cfg.num_segments),
        "num_rings": int(cfg.num_rings),
        "cap_rings": int(cfg.full3d_cap_rings),
        "top_bottom_faces_variable": True,
        "electrode_constraint": "two 5 mm circular electrode boundaries remain fixed in diameter and relative position",
        "volume_constraint": "pre-energization closed-mesh volume is projected to the initial cylinder volume",
        "thermal_boundary": "300 K fixed-temperature electrode ends plus 300 K free-surface radiative sink; end faces in contact with electrodes do not radiate or sublime",
        "external_sphere": "optical target is escaped 0-3 um free-surface radiation to a 0 K absorbing sphere; thermal balance uses 300 K free-surface radiation",
        "policy_model": "lightweight 3D U-Net encoder with graph-neighborhood smoothing head",
        "optimizer": "CEM over explicit low-order full3d topology actions, optionally seeded by the U-Net/GNN structured perturbation",
        "action_space_full3d": str(result.selection_diagnostics.get("action_space_full3d", "")),
        "action_dim_full3d": int(result.selection_diagnostics.get("action_dim_full3d", 0)),
        "cem_elite_fraction_full3d": float(result.selection_diagnostics.get("cem_elite_fraction_full3d", 0.0)),
        "voltage_search_mode": str(result.best_metrics.get("voltage_search_mode", result.selection_diagnostics.get("voltage_search_mode", ""))),
        "voltage_constraint": (
            f"fixed diagnostic voltage {float(cfg.full3d_fixed_voltage_v):.6g} V"
            if cfg.full3d_fixed_voltage_v is not None
            else f"rated voltage is searched under V <= {float(cfg.max_voltage):.6g} V; max voltage is not a fixed operating point"
        ),
        "fixed_voltage_v": None if cfg.full3d_fixed_voltage_v is None else float(cfg.full3d_fixed_voltage_v),
        "rated_voltage_upper_bound_v": float(cfg.max_voltage),
        "baseline_voltage_v": float(result.baseline_metrics.get("voltage_v", 0.0)),
        "final_voltage_v": float(result.best_metrics.get("voltage_v", 0.0)),
        "baseline_feasible_full3d": bool(result.baseline_metrics.get("constraint_feasible_3d", False)),
        "baseline_net_radiated_power_0k_sphere_w": base_power,
        "final_net_radiated_power_0k_sphere_w": final_power,
        "final_net_radiated_power_300k_environment_w": float(result.best_metrics.get("net_radiated_power_300k_environment_w", final_power)),
        "thermal_radiation_sink_temperature_k": float(result.best_metrics.get("thermal_radiation_sink_temperature_k", cfg.ambient_temp)),
        "electrode_boundary_temperature_k": float(result.best_metrics.get("electrode_boundary_temperature_k", cfg.ambient_temp)),
        "tungsten_voltage_v": float(result.best_metrics.get("tungsten_voltage_v", result.best_metrics.get("voltage_v", 0.0))),
        "electrode_voltage_drop_v": float(result.best_metrics.get("electrode_voltage_drop_v", 0.0)),
        "contact_end_face_area_m2": float(result.best_metrics.get("contact_end_face_area_m2", 0.0)),
        "free_radiating_surface_area_m2": float(result.best_metrics.get("free_radiating_surface_area_m2", 0.0)),
        "thermal_converged": bool(result.best_metrics.get("thermal_converged", False)),
        "power_ratio_full3d": final_power / base_power,
        "baseline_lifetime_s_full3d": float(result.baseline_metrics.get("lifetime_s", 0.0)),
        "final_lifetime_s_full3d": float(result.best_metrics.get("lifetime_s", 0.0)),
        "lifetime_ratio_full3d": float(result.best_metrics.get("lifetime_ratio_3d", 0.0)),
        "baseline_max_temperature_k": float(result.baseline_metrics.get("max_temperature_k", 0.0)),
        "volume_change_ratio_full3d": float(result.best_metrics.get("volume_change_ratio_3d", 0.0)),
        "electrode_max_error_m": float(result.best_metrics.get("electrode_max_error_m", 0.0)),
        "electrical_power_w": float(result.best_metrics.get("electrical_power_w", 0.0)),
        "full_spectrum_radiative_power_w": float(result.best_metrics.get("full_spectrum_radiative_power_w", 0.0)),
        "thermal_balance_residual_w": float(result.best_metrics.get("thermal_balance_residual_w", 0.0)),
        "effective_radiating_area_m2": float(result.best_metrics.get("effective_radiating_area_m2", 0.0)),
        "escape_view_factor_proxy": float(result.best_metrics.get("escape_view_factor_proxy", 0.0)),
        "surface_area_ratio": float(result.best_metrics.get("surface_area_ratio", 1.0)),
        "blackbody_band_fraction_0_3um": float(result.best_metrics.get("blackbody_band_fraction_0_3um", 0.0)),
        "max_temperature_k": float(result.best_metrics.get("max_temperature_k", 0.0)),
        "temperature_violation_ratio": float(result.best_metrics.get("temperature_violation_ratio", 0.0)),
        "feasible": bool(result.best_metrics.get("constraint_feasible_3d", False)),
        **result.selection_diagnostics,
        **export_info,
        **anim_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full closed-mesh 3D optimization for the tungsten cylinder task.")
    parser.add_argument("--output-dir", type=str, default="outputs/three_d_runs")
    parser.add_argument("--experiment-name", type=str, default="mcga_full3d")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--thermal-iters", type=int, default=640)
    parser.add_argument("--freecad-timeout", type=float, default=90.0)
    parser.add_argument("--max-voltage", type=float, default=100.0, help="Rated-voltage search upper bound in volts.")
    parser.add_argument("--fixed-voltage", type=float, default=None, help="Optional diagnostic voltage. Omit to search the rated voltage under --max-voltage.")
    parser.add_argument("--cap-rings", type=int, default=8)
    parser.add_argument("--full3d-volume-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--full3d-neural-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-axial-modes", type=int, default=None)
    parser.add_argument("--action-circum-modes", type=int, default=None)
    parser.add_argument("--action-cap-radial-modes", type=int, default=None)
    parser.add_argument("--cem-initial-sigma", type=float, default=None)
    parser.add_argument("--cem-elite-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-step", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    print(f"[3d] device: {torch.cuda.get_device_name(0) if str(args.device).startswith('cuda') else 'CPU'}", flush=True)
    if args.smoke:
        args.generations = min(int(args.generations), 2)
        args.population_size = min(int(args.population_size), 8)

    cfg = _configure_cfg(args)
    output_dir = Path(args.output_dir) / f"{args.experiment_name}_{datetime.now().strftime('%m-%d-%H-%M')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_full3d_optimization(
        cfg,
        generations=int(args.generations),
        population_size=int(args.population_size),
        seed=int(args.seed),
        use_neural_policy=bool(args.full3d_neural_policy),
    )
    export_info = export_full3d_mesh(
        result.best_geometry,
        cfg,
        output_dir=output_dir,
        output_name="optimized_full3d",
        export_step=not bool(args.no_step),
    )
    anim_info = export_full3d_animation(
        result.history_geometries,
        result.history_metrics,
        output_dir=output_dir,
        output_name="topology_evolution_full3d",
    )
    summary = _build_summary(args, cfg, result, export_info, anim_info)
    save_full3d_history_csv(output_dir, result.history_metrics)
    save_full3d_summary(output_dir, summary)
    report = write_full3d_report(output_dir, result, summary)
    print(f"[3d] artifacts saved to: {output_dir}", flush=True)
    print(f"[3d] report: {report}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
