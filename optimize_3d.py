from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config.cylinder_cfg import make_eval_cfg, make_training_cfg
from utils.hybrid_optimizer_3d import (
    Hybrid3DResult,
    evaluate_radius_fields,
    export_3d_evolution_animation,
    export_radius_field_mesh,
    resample_radius_field,
    run_hybrid_3d_optimization,
    save_3d_history_csv,
    save_3d_summary,
    write_3d_strategy_report,
    _extract_metrics,
)


def _configure_cfg(args, eval_mode: bool = False):
    cfg = make_eval_cfg() if eval_mode else make_training_cfg()
    cfg.device = args.device
    cfg.thermal_max_iters = int(args.thermal_iters)
    cfg.shadow_slope_coeff = float(args.shadow_slope_coeff)
    cfg.shadow_roughness_coeff = float(args.shadow_roughness_coeff)
    cfg.freecad_timeout_s = float(args.freecad_timeout)
    if args.smoke:
        cfg.num_segments = 32 if not eval_mode else 48
        cfg.num_rings = 24 if not eval_mode else 48
        cfg.voltage_grid_points = 5
        cfg.voltage_refine_levels = 1
        cfg.voltage_refine_points = 5
        cfg.thermal_max_iters = min(int(args.thermal_iters), 240)
        cfg.transient_max_time_s = 8.0
        cfg.transient_dt_s = 1.0
    return cfg


def _select_eval_candidate(eval_cfg, result: Hybrid3DResult):
    eval_fields = [
        resample_radius_field(eval_cfg, field, int(eval_cfg.num_rings), int(eval_cfg.num_segments))
        for field in result.archive_fields
    ]
    field_t = torch.as_tensor(np.stack(eval_fields, axis=0), dtype=torch.float32, device=torch.device(eval_cfg.device if torch.cuda.is_available() and str(eval_cfg.device).startswith("cuda") else "cpu"))
    baseline_t = field_t[:1]
    baseline_metrics_t, baseline_transient_t, baseline_score_t = evaluate_radius_fields(eval_cfg, baseline_t)
    metrics_t, transient_t, scores_t = evaluate_radius_fields(eval_cfg, field_t, baseline_metrics_t)
    life_ratio = metrics_t["adjusted_lifetime_ratio"]
    valid = metrics_t["constraint_feasible_3d"] & (life_ratio >= float(eval_cfg.minimum_lifetime_ratio))
    selection = torch.where(valid, scores_t, torch.full_like(scores_t, -float("inf")))
    if not bool(torch.any(valid).item()):
        selection = scores_t - 70.0 * torch.clamp(float(eval_cfg.minimum_lifetime_ratio) - life_ratio, min=0.0)
    idx = int(torch.argmax(selection).item())
    baseline_metrics = _extract_metrics(baseline_metrics_t, baseline_transient_t, baseline_score_t, 0)
    final_metrics = _extract_metrics(metrics_t, transient_t, scores_t, idx)
    return eval_fields[0], eval_fields[idx], baseline_metrics, final_metrics, idx


def _build_summary(args, eval_cfg, result, baseline, final, selected_idx, artifacts):
    base_power = max(float(baseline.get("adjusted_initial_power_w", baseline.get("initial_net_band_power_w", 0.0))), 1.0e-9)
    final_power = float(final.get("adjusted_initial_power_w", final.get("initial_net_band_power_w", 0.0)))
    base_avg = max(float(baseline.get("adjusted_average_power_w", baseline.get("average_net_band_power_w", 0.0))), 1.0e-9)
    final_avg = float(final.get("adjusted_average_power_w", final.get("average_net_band_power_w", 0.0)))
    return {
        "method": "3d_physics_informed_cem_fourier_surface_search",
        "device": args.device,
        "candidate_count": int(result.candidate_count),
        "selected_archive_index": int(selected_idx),
        "num_segments": int(eval_cfg.num_segments),
        "num_rings": int(eval_cfg.num_rings),
        "voltage_constraint": "rated voltage is searched under V <= 100 V; 100 V is not a fixed operating point",
        "baseline_voltage_v": float(baseline.get("voltage_v", 0.0)),
        "final_voltage_v": float(final.get("voltage_v", 0.0)),
        "baseline_initial_power_w_3d": base_power,
        "final_initial_power_w_3d": final_power,
        "initial_power_ratio_3d": final_power / base_power,
        "baseline_average_power_w_3d": base_avg,
        "final_average_power_w_3d": final_avg,
        "average_power_ratio_3d": final_avg / base_avg,
        "baseline_lifetime_s_3d": float(baseline.get("adjusted_lifetime_s", baseline.get("lifetime_s", 0.0))),
        "final_lifetime_s_3d": float(final.get("adjusted_lifetime_s", final.get("lifetime_s", 0.0))),
        "lifetime_ratio_3d": float(final.get("adjusted_lifetime_ratio", 0.0)),
        "feature_change_ratio_3d": float(final.get("feature_change_ratio_3d", 0.0)),
        "volume_change_ratio_3d": float(final.get("volume_change_ratio_3d", 0.0)),
        "surface_area_ratio": float(final.get("surface_area_ratio", 1.0)),
        "surface_view_factor": float(final.get("surface_view_factor", 1.0)),
        "surface_gain": float(final.get("surface_gain", 1.0)),
        "feasible": bool(final.get("constraint_feasible_3d", False)),
        **artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run true 3D surface optimization for the tungsten cylinder task.")
    parser.add_argument("--output-dir", type=str, default="outputs/three_d_runs")
    parser.add_argument("--experiment-name", type=str, default="mcga_3d_sota")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--axial-modes", type=int, default=4)
    parser.add_argument("--circum-modes", type=int, default=2)
    parser.add_argument("--thermal-iters", type=int, default=640)
    parser.add_argument("--shadow-slope-coeff", type=float, default=0.01)
    parser.add_argument("--shadow-roughness-coeff", type=float, default=0.01)
    parser.add_argument("--freecad-timeout", type=float, default=90.0, help="Seconds to wait for FreeCAD STEP export before falling back to STL only.")
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
        args.axial_modes = min(int(args.axial_modes), 3)
        args.circum_modes = min(int(args.circum_modes), 1)

    train_cfg = _configure_cfg(args, eval_mode=False)
    eval_cfg = _configure_cfg(args, eval_mode=True)
    output_dir = Path(args.output_dir) / f"{args.experiment_name}_{datetime.now().strftime('%m-%d-%H-%M')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_hybrid_3d_optimization(
        train_cfg,
        generations=int(args.generations),
        population_size=int(args.population_size),
        elite_fraction=float(args.elite_fraction),
        axial_modes=int(args.axial_modes),
        circum_modes=int(args.circum_modes),
        seed=int(args.seed),
    )
    baseline_eval_field, final_eval_field, baseline_metrics, final_metrics, selected_idx = _select_eval_candidate(eval_cfg, result)
    history_eval_fields = [
        resample_radius_field(eval_cfg, field, int(eval_cfg.num_rings), int(eval_cfg.num_segments))
        for field in result.history_fields
    ]
    history_metrics = [dict(item) for item in result.history_metrics]
    if history_metrics:
        history_metrics[0].update(baseline_metrics)
        history_metrics[-1].update(final_metrics)
        history_metrics[-1]["selected_archive_index"] = selected_idx

    export_info = export_radius_field_mesh(
        final_eval_field,
        eval_cfg,
        output_dir=str(output_dir),
        output_name="optimized_cylinder_3d",
        export_step=not bool(args.no_step),
    )
    anim_info = export_3d_evolution_animation(
        history_eval_fields,
        history_metrics,
        eval_cfg,
        output_dir=str(output_dir),
        output_name="topology_evolution_3d",
    )
    artifacts = {
        "stl": export_info["stl"],
        "stp": export_info["stp"],
        "watertight": bool(export_info["watertight"]),
        "gif": anim_info["gif"],
        "mp4": anim_info["mp4"],
    }
    summary = _build_summary(args, eval_cfg, result, baseline_metrics, final_metrics, selected_idx, artifacts)
    save_3d_history_csv(output_dir, history_metrics)
    save_3d_summary(output_dir, summary)
    report_result = Hybrid3DResult(
        best_field=final_eval_field,
        baseline_field=baseline_eval_field,
        best_metrics=final_metrics,
        baseline_metrics=baseline_metrics,
        history_fields=history_eval_fields,
        history_metrics=history_metrics,
        archive_fields=[],
        archive_metrics=[],
        candidate_count=result.candidate_count,
    )
    report = write_3d_strategy_report(output_dir, report_result, summary)
    print(f"[3d] artifacts saved to: {output_dir}", flush=True)
    print(f"[3d] report: {report}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
