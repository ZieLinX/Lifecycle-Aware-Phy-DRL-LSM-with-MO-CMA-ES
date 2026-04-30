from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config.cylinder_cfg import make_eval_cfg, make_training_cfg
from utils.animation import export_topology_evolution_animation
from utils.exporter import export_ring_profile_mesh
from utils.hybrid_optimizer import (
    HybridOptimizationResult,
    evaluate_profiles,
    profiles_from_coefficients,
    resample_profile,
    save_history_csv,
    save_summary_json,
    write_strategy_report,
    _extract_metrics,
    build_chebyshev_basis,
    run_hybrid_optimization,
)


def _configure_training(args):
    cfg = make_training_cfg()
    cfg.device = args.device
    cfg.max_steps = int(args.max_steps)
    cfg.thermal_max_iters = int(args.thermal_iters)
    cfg.shadow_slope_coeff = float(args.shadow_slope_coeff)
    cfg.shadow_roughness_coeff = float(args.shadow_roughness_coeff)
    if args.smoke:
        cfg.num_segments = 32
        cfg.num_rings = 24
        cfg.voltage_grid_points = 5
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 5
        cfg.thermal_max_iters = min(int(args.thermal_iters), 240)
        cfg.transient_max_time_s = 6.0
        cfg.transient_dt_s = 1.0
    return cfg


def _configure_eval(args):
    cfg = make_eval_cfg()
    cfg.device = args.device
    cfg.max_steps = int(args.max_steps)
    cfg.thermal_max_iters = int(args.thermal_iters)
    cfg.shadow_slope_coeff = float(args.shadow_slope_coeff)
    cfg.shadow_roughness_coeff = float(args.shadow_roughness_coeff)
    if args.smoke:
        cfg.num_segments = 48
        cfg.num_rings = 48
        cfg.voltage_grid_points = 5
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 5
        cfg.thermal_max_iters = min(int(args.thermal_iters), 240)
        cfg.transient_max_time_s = 6.0
        cfg.transient_dt_s = 1.0
    return cfg


def _select_final_profile(eval_cfg, profiles: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict, dict, int]:
    basis = build_chebyshev_basis(int(eval_cfg.num_rings), 1)
    baseline_profile_t = profiles_from_coefficients(eval_cfg, np.zeros((1, 1)), basis)
    baseline_metrics_t, baseline_transient_t, baseline_scores_t = evaluate_profiles(eval_cfg, baseline_profile_t)
    baseline_metrics = _extract_metrics(baseline_metrics_t, baseline_transient_t, baseline_scores_t, 0)

    resampled = [resample_profile(eval_cfg, profile, int(eval_cfg.num_rings)) for profile in profiles]
    profile_t = torch.as_tensor(np.stack(resampled, axis=0), dtype=torch.float32, device=baseline_profile_t.device)
    metrics_t, transient_t, scores_t = evaluate_profiles(eval_cfg, profile_t, baseline_metrics_t)
    life_ratio = metrics_t["lifetime_s"] / torch.clamp(baseline_metrics_t["lifetime_s"], min=1.0e-9)
    valid = metrics_t["feasible"] & (life_ratio >= float(eval_cfg.minimum_lifetime_ratio))
    if bool(torch.any(valid).item()):
        selection = torch.where(valid, scores_t, torch.full_like(scores_t, -float("inf")))
    else:
        selection = scores_t - 100.0 * torch.clamp(float(eval_cfg.minimum_lifetime_ratio) - life_ratio, min=0.0)
    best_idx = int(torch.argmax(selection).item())
    final_metrics = _extract_metrics(metrics_t, transient_t, scores_t, best_idx)
    final_metrics["lifetime_ratio"] = float(life_ratio[best_idx].item())
    final_metrics["constraint_feasible"] = bool(valid[best_idx].item())
    return baseline_profile_t[0].detach().cpu().numpy(), resampled[best_idx], baseline_metrics, final_metrics, best_idx


def _summary_from_metrics(args, eval_cfg, result: HybridOptimizationResult, baseline_eval: dict, final_eval: dict, artifact_info: dict) -> dict:
    return {
        "method": "physics_informed_hybrid_cem_local_search",
        "device": args.device,
        "smoke": bool(args.smoke),
        "candidate_count": int(result.candidate_count),
        "num_segments": int(eval_cfg.num_segments),
        "num_rings": int(eval_cfg.num_rings),
        "voltage_constraint": "rated voltage is searched under V <= 100 V; 100 V is not a fixed operating point",
        "time_selection": str(getattr(eval_cfg, "transient_time_selection", "search")),
        "baseline_voltage_v": float(baseline_eval.get("voltage_v", 0.0)),
        "final_voltage_v": float(final_eval.get("voltage_v", 0.0)),
        "baseline_initial_power_w": float(baseline_eval.get("initial_net_band_power_w", 0.0)),
        "final_initial_power_w": float(final_eval.get("initial_net_band_power_w", 0.0)),
        "baseline_average_power_w": float(baseline_eval.get("average_net_band_power_w", 0.0)),
        "final_average_power_w": float(final_eval.get("average_net_band_power_w", 0.0)),
        "baseline_lifetime_s": float(baseline_eval.get("lifetime_s", 0.0)),
        "final_lifetime_s": float(final_eval.get("lifetime_s", 0.0)),
        "initial_power_ratio": float(final_eval.get("initial_net_band_power_w", 0.0) / max(float(baseline_eval.get("initial_net_band_power_w", 0.0)), 1.0e-9)),
        "average_power_ratio": float(final_eval.get("average_net_band_power_w", 0.0) / max(float(baseline_eval.get("average_net_band_power_w", 0.0)), 1.0e-9)),
        "lifetime_ratio": float(final_eval.get("lifetime_s", 0.0) / max(float(baseline_eval.get("lifetime_s", 0.0)), 1.0e-9)),
        "baseline_optimal_transient_time_s": float(baseline_eval.get("optimal_transient_time_s", 0.0)),
        "final_optimal_transient_time_s": float(final_eval.get("optimal_transient_time_s", 0.0)),
        "final_transient_power_w": float(final_eval.get("transient_power_w", 0.0)),
        "final_transient_mean_power_w": float(final_eval.get("transient_mean_power_w", 0.0)),
        "final_max_temp_k": float(final_eval.get("max_temperature_k", 0.0)),
        "feature_change_ratio": float(final_eval.get("feature_change_ratio", 0.0)),
        "volume_change_ratio": float(final_eval.get("volume_change_ratio", 0.0)),
        "feasible": bool(final_eval.get("constraint_feasible", False)),
        **artifact_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics-informed hybrid optimizer for the tungsten cylinder task.")
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid_runs")
    parser.add_argument("--experiment-name", type=str, default="mcga_hybrid_sota")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--local-iterations", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--thermal-iters", type=int, default=640)
    parser.add_argument("--shadow-slope-coeff", type=float, default=0.01)
    parser.add_argument("--shadow-roughness-coeff", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-step", action="store_true", help="Skip STEP/STP conversion and write STL only.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if str(args.device).startswith("cuda"):
        print(f"[hybrid] optimization device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[hybrid] optimization device: CPU", flush=True)

    if args.smoke:
        args.generations = min(int(args.generations), 2)
        args.population_size = min(int(args.population_size), 8)
        args.num_modes = min(int(args.num_modes), 4)
        args.local_iterations = min(int(args.local_iterations), 1)

    train_cfg = _configure_training(args)
    eval_cfg = _configure_eval(args)
    stamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = Path(args.output_dir) / f"{args.experiment_name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_hybrid_optimization(
        train_cfg,
        generations=int(args.generations),
        population_size=int(args.population_size),
        elite_fraction=float(args.elite_fraction),
        num_modes=int(args.num_modes),
        local_iterations=int(args.local_iterations),
        seed=int(args.seed),
    )

    baseline_profile_eval, final_profile_eval, baseline_eval, final_eval, selected_archive_index = _select_final_profile(
        eval_cfg,
        result.archive_profiles,
    )
    history_profiles_eval = [resample_profile(eval_cfg, profile, int(eval_cfg.num_rings)) for profile in result.history_profiles]
    history_metrics = [dict(row) for row in result.history_metrics]
    if history_metrics:
        history_metrics[0].update(baseline_eval)
        history_metrics[-1].update(final_eval)
        history_metrics[-1]["selected_archive_index"] = selected_archive_index

    export_info = export_ring_profile_mesh(
        ring_radius=final_profile_eval,
        height=float(eval_cfg.height),
        num_segments=int(eval_cfg.num_segments),
        output_dir=str(output_dir),
        output_name="optimized_cylinder",
        export_step=not bool(args.no_step),
        freecad_cmd=getattr(eval_cfg, "freecad_cmd", ""),
    )
    animation_info = export_topology_evolution_animation(
        ring_radius_history=history_profiles_eval,
        metrics_history=history_metrics,
        height=float(eval_cfg.height),
        output_dir=str(output_dir),
        output_name="topology_evolution",
    )
    artifact_info = {
        "stl": export_info["stl"],
        "stp": export_info["stp"],
        "watertight": bool(export_info["watertight"]),
        "gif": animation_info["gif"],
        "mp4": animation_info["mp4"],
    }
    summary = _summary_from_metrics(args, eval_cfg, result, baseline_eval, final_eval, artifact_info)
    summary["selected_archive_index"] = int(selected_archive_index)
    save_history_csv(output_dir, history_metrics)
    save_summary_json(output_dir, summary)

    report_result = HybridOptimizationResult(
        best_profile=final_profile_eval,
        baseline_profile=baseline_profile_eval,
        best_metrics=final_eval,
        baseline_metrics=baseline_eval,
        history_profiles=history_profiles_eval,
        history_metrics=history_metrics,
        archive_profiles=[resample_profile(eval_cfg, profile, int(eval_cfg.num_rings)) for profile in result.archive_profiles],
        archive_metrics=result.archive_metrics,
        candidate_count=result.candidate_count,
    )
    report_path = write_strategy_report(output_dir, report_result, eval_cfg, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[hybrid] artifacts saved to: {output_dir}", flush=True)
    print(f"[hybrid] strategy report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
