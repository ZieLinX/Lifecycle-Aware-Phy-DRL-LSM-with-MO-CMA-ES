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
    cfg.hybrid3d_initial_sigma = float(args.initial_sigma)
    cfg.hybrid3d_min_sigma = float(args.min_sigma)
    cfg.hybrid3d_max_sigma = float(args.max_sigma)
    cfg.hybrid3d_max_log_delta = float(args.max_log_delta)
    cfg.hybrid3d_circum_penalty = float(args.circum_penalty)
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
    has_valid = bool(torch.any(valid).item())
    if not has_valid:
        selection = scores_t - 70.0 * torch.clamp(float(eval_cfg.minimum_lifetime_ratio) - life_ratio, min=0.0)
    idx = int(torch.argmax(selection).item())
    baseline_metrics = _extract_metrics(baseline_metrics_t, baseline_transient_t, baseline_score_t, 0)
    final_metrics = _extract_metrics(metrics_t, transient_t, scores_t, idx)
    diagnostics = _archive_diagnostics(metrics_t, scores_t, valid, idx, baseline_metrics, final_metrics)
    diagnostics["selection_reason_3d"] = _selection_reason(has_valid, idx, diagnostics)
    return eval_fields[0], eval_fields[idx], baseline_metrics, final_metrics, idx, diagnostics


def _metric_item(metrics, key: str, idx: int, default=None):
    value = metrics.get(key)
    if isinstance(value, torch.Tensor) and value.ndim == 1 and idx < value.numel():
        item = value[idx]
        return bool(item.item()) if item.dtype == torch.bool else float(item.item())
    return default


def _candidate_snapshot(metrics, scores, idx: int) -> dict:
    keys = [
        "adjusted_initial_power_w",
        "adjusted_average_power_w",
        "adjusted_lifetime_ratio",
        "initial_net_band_power_w",
        "average_net_band_power_w",
        "voltage_v",
        "max_temperature_k",
        "surface_area_ratio",
        "surface_view_factor",
        "surface_gain",
        "circum_nonuniformity",
        "feature_change_ratio_3d",
        "volume_change_ratio_3d",
        "constraint_feasible_3d",
    ]
    out = {"archive_index": int(idx), "score": float(scores[idx].item())}
    for key in keys:
        value = _metric_item(metrics, key, idx)
        if value is not None:
            out[key] = value
    return out


def _archive_diagnostics(metrics, scores, valid, selected_idx: int, baseline_metrics: dict, final_metrics: dict) -> dict:
    count = int(scores.numel())
    baseline_power = max(float(baseline_metrics.get("adjusted_initial_power_w", baseline_metrics.get("initial_net_band_power_w", 0.0))), 1.0e-9)
    feasible_count = int(torch.count_nonzero(valid).item())
    nonbaseline_mask = torch.ones_like(valid, dtype=torch.bool)
    if nonbaseline_mask.numel():
        nonbaseline_mask[0] = False
    feasible_nonbaseline = valid & nonbaseline_mask
    diag: dict[str, object] = {
        "archive_candidate_count": count,
        "archive_feasible_count": feasible_count,
        "archive_feasible_nonbaseline_count": int(torch.count_nonzero(feasible_nonbaseline).item()),
        "baseline_score_3d": float(baseline_metrics.get("score", scores[0].item() if count else 0.0)),
        "final_score_3d": float(final_metrics.get("score", 0.0)),
        "score_delta_vs_baseline_3d": float(final_metrics.get("score", 0.0) - baseline_metrics.get("score", 0.0)),
    }
    if count <= 1:
        return diag
    nonbaseline_scores = torch.where(nonbaseline_mask, scores, torch.full_like(scores, -float("inf")))
    best_nb_score_idx = int(torch.argmax(nonbaseline_scores).item())
    best_nb_score = _candidate_snapshot(metrics, scores, best_nb_score_idx)
    best_nb_score["initial_power_ratio_vs_baseline"] = float(best_nb_score.get("adjusted_initial_power_w", 0.0)) / baseline_power
    diag["best_nonbaseline_by_score"] = best_nb_score

    feasible_power = torch.where(
        feasible_nonbaseline,
        metrics["adjusted_initial_power_w"],
        torch.full_like(metrics["adjusted_initial_power_w"], -float("inf")),
    )
    if bool(torch.any(feasible_nonbaseline).item()):
        best_power_idx = int(torch.argmax(feasible_power).item())
        best_power = _candidate_snapshot(metrics, scores, best_power_idx)
        best_power["initial_power_ratio_vs_baseline"] = float(best_power.get("adjusted_initial_power_w", 0.0)) / baseline_power
        diag["best_feasible_nonbaseline_by_initial_power"] = best_power
    feature = metrics.get("feature_change_ratio_3d")
    if isinstance(feature, torch.Tensor) and feature.ndim == 1:
        diag["max_nonbaseline_feature_change_ratio_3d"] = float(torch.max(feature[1:]).item())
    area = metrics.get("surface_area_ratio")
    if isinstance(area, torch.Tensor) and area.ndim == 1:
        diag["max_nonbaseline_surface_area_ratio"] = float(torch.max(area[1:]).item())
    return diag


def _selection_reason(has_valid: bool, selected_idx: int, diagnostics: dict) -> str:
    if not has_valid:
        return "no archive candidate satisfied 3D feasibility and lifetime constraints; selected highest penalized score"
    if int(selected_idx) != 0:
        return "selected feasible archive candidate with highest fine-grid 3D score"
    best_power = diagnostics.get("best_feasible_nonbaseline_by_initial_power")
    best_score = diagnostics.get("best_nonbaseline_by_score")
    if isinstance(best_power, dict):
        power_ratio = float(best_power.get("initial_power_ratio_vs_baseline", 0.0))
        if power_ratio <= 1.0:
            return "baseline retained: no feasible non-baseline candidate improved fine-grid adjusted initial power"
    if isinstance(best_score, dict):
        delta = float(best_score.get("score", 0.0)) - float(diagnostics.get("baseline_score_3d", 0.0))
        if delta <= 0.0:
            return "baseline retained: all feasible non-baseline candidates scored below baseline"
    return "baseline retained by fine-grid score tie or score/lifetime tradeoff"


def _build_summary(args, eval_cfg, result, baseline, final, selected_idx, artifacts, diagnostics):
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
        **diagnostics,
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
    parser.add_argument("--initial-sigma", type=float, default=0.075, help="Initial CEM stddev for 3D Fourier/Chebyshev coefficients.")
    parser.add_argument("--min-sigma", type=float, default=0.010, help="Minimum CEM stddev for 3D coefficients.")
    parser.add_argument("--max-sigma", type=float, default=0.140, help="Maximum CEM stddev for 3D coefficients.")
    parser.add_argument("--max-log-delta", type=float, default=0.160, help="Clamp log radius perturbations to +/- this value before projection.")
    parser.add_argument("--circum-penalty", type=float, default=0.35, help="Score penalty weight for circumferential nonuniformity.")
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
    baseline_eval_field, final_eval_field, baseline_metrics, final_metrics, selected_idx, selection_diagnostics = _select_eval_candidate(eval_cfg, result)
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
    summary = _build_summary(args, eval_cfg, result, baseline_metrics, final_metrics, selected_idx, artifacts, selection_diagnostics)
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
