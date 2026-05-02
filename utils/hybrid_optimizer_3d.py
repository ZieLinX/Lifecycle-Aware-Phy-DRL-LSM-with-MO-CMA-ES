from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import math
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.exporter import export_mesh_files
from utils.hybrid_optimizer import axial_weights_np
from utils.rated_condition import search_rated_condition_3d_batch, simulate_transient_trajectory_3d, summarize_transient_selection


@dataclass(frozen=True)
class Hybrid3DResult:
    best_field: np.ndarray
    baseline_field: np.ndarray
    best_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    history_fields: List[np.ndarray]
    history_metrics: List[Dict[str, float]]
    archive_fields: List[np.ndarray]
    archive_metrics: List[Dict[str, float]]
    candidate_count: int


def radius_field_volume(radius_field: np.ndarray, height: float) -> float:
    field = np.asarray(radius_field, dtype=np.float64)
    weights = axial_weights_np(field.shape[0], height)
    cross_section_area = math.pi * np.mean(field * field, axis=1)
    return float(np.sum(cross_section_area * weights))


def points_from_radius_field(radius_field: np.ndarray, height: float) -> np.ndarray:
    field = np.asarray(radius_field, dtype=np.float64)
    num_rings, num_segments = field.shape
    z_values = np.linspace(-0.5 * float(height), 0.5 * float(height), num_rings)
    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False)
    points = []
    for ridx, z in enumerate(z_values):
        for sidx, angle in enumerate(theta):
            r = max(float(field[ridx, sidx]), 1.0e-9)
            points.append([r * math.cos(angle), r * math.sin(angle), z])
    return np.asarray(points, dtype=np.float64)


def build_3d_basis(num_rings: int, num_segments: int, axial_modes: int, circum_modes: int) -> np.ndarray:
    z = np.linspace(-1.0, 1.0, int(num_rings), dtype=np.float64)
    envelope = np.clip(1.0 - z * z, 0.0, 1.0)
    axial = []
    t_prev = np.ones_like(z)
    axial.append(envelope * t_prev)
    if int(axial_modes) > 1:
        t_cur = z.copy()
        axial.append(envelope * t_cur)
        for _ in range(2, int(axial_modes)):
            t_next = 2.0 * z * t_cur - t_prev
            axial.append(envelope * t_next)
            t_prev, t_cur = t_cur, t_next

    theta = np.linspace(0.0, 2.0 * math.pi, int(num_segments), endpoint=False)
    circum = [np.ones_like(theta)]
    for mode in range(1, int(circum_modes) + 1):
        circum.append(np.cos(mode * theta))
        circum.append(np.sin(mode * theta))

    basis = []
    for a in axial[: int(axial_modes)]:
        for c in circum:
            item = a[:, None] * c[None, :]
            item = item / max(float(np.max(np.abs(item))), 1.0e-12)
            basis.append(item)
    return np.stack(basis, axis=0)


def _project_radius_field(cfg, radius: torch.Tensor) -> torch.Tensor:
    if radius.ndim == 2:
        radius = radius.unsqueeze(0)
    r = torch.clamp(radius.float(), min=float(cfg.min_radius))
    r[:, 0, :] = float(cfg.radius)
    r[:, -1, :] = float(cfg.radius)
    dz = float(cfg.height) / max(int(r.shape[1]) - 1, 1)
    dtheta_arc = 2.0 * math.pi * float(cfg.radius) / max(int(r.shape[2]), 1)
    axial_delta = float(getattr(cfg, "feasibility_max_slope", 0.35)) * dz
    circum_delta = float(getattr(cfg, "feasibility_max_slope", 0.35)) * dtheta_arc
    weights = torch.as_tensor(axial_weights_np(r.shape[1], float(cfg.height)), dtype=torch.float32, device=r.device)
    target_volume = math.pi * float(cfg.radius) ** 2 * float(cfg.height)
    endpoint_volume = math.pi * float(cfg.radius) ** 2 * float((weights[0] + weights[-1]).item())
    target_interior = max(target_volume - endpoint_volume, 1.0e-12)

    for _ in range(6):
        r[:, 0, :] = float(cfg.radius)
        r[:, -1, :] = float(cfg.radius)
        for idx in range(1, r.shape[1] - 1):
            r[:, idx, :] = torch.minimum(r[:, idx, :], r[:, idx - 1, :] + axial_delta)
            r[:, idx, :] = torch.maximum(r[:, idx, :], r[:, idx - 1, :] - axial_delta)
        for idx in range(r.shape[1] - 2, 0, -1):
            r[:, idx, :] = torch.minimum(r[:, idx, :], r[:, idx + 1, :] + axial_delta)
            r[:, idx, :] = torch.maximum(r[:, idx, :], r[:, idx + 1, :] - axial_delta)
        for _circ in range(2):
            for sidx in range(1, r.shape[2]):
                r[:, :, sidx] = torch.minimum(r[:, :, sidx], r[:, :, sidx - 1] + circum_delta)
                r[:, :, sidx] = torch.maximum(r[:, :, sidx], r[:, :, sidx - 1] - circum_delta)
            for sidx in range(r.shape[2] - 2, -1, -1):
                r[:, :, sidx] = torch.minimum(r[:, :, sidx], r[:, :, (sidx + 1) % r.shape[2]] + circum_delta)
                r[:, :, sidx] = torch.maximum(r[:, :, sidx], r[:, :, (sidx + 1) % r.shape[2]] - circum_delta)
        r = torch.clamp(r, min=float(cfg.min_radius))
        if r.shape[1] > 2:
            area = math.pi * torch.mean(r[:, 1:-1, :].pow(2), dim=2)
            current_interior = torch.sum(area * weights[1:-1].unsqueeze(0), dim=1, keepdim=True)
            scale = torch.sqrt(torch.as_tensor(target_interior, dtype=torch.float32, device=r.device) / torch.clamp(current_interior, min=1.0e-12))
            r[:, 1:-1, :] = torch.clamp(r[:, 1:-1, :] * scale[:, :, None], min=float(cfg.min_radius))
    r[:, 0, :] = float(cfg.radius)
    r[:, -1, :] = float(cfg.radius)
    return r


def fields_from_coefficients(cfg, coefficients: np.ndarray, basis: np.ndarray) -> torch.Tensor:
    coeff = np.asarray(coefficients, dtype=np.float64)
    if coeff.ndim == 1:
        coeff = coeff.reshape(1, -1)
    log_delta = np.einsum("bk,krs->brs", coeff, np.asarray(basis, dtype=np.float64))
    log_delta = np.clip(log_delta, -0.10, 0.10)
    raw = float(cfg.radius) * np.exp(log_delta)
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    return _project_radius_field(cfg, torch.as_tensor(raw, dtype=torch.float32, device=device))


def resample_radius_field(cfg, radius_field: np.ndarray, num_rings: int, num_segments: int) -> np.ndarray:
    field = np.asarray(radius_field, dtype=np.float64)
    z0 = np.linspace(-1.0, 1.0, field.shape[0])
    z1 = np.linspace(-1.0, 1.0, int(num_rings))
    tmp = np.zeros((int(num_rings), field.shape[1]), dtype=np.float64)
    for sidx in range(field.shape[1]):
        tmp[:, sidx] = np.interp(z1, z0, field[:, sidx])
    theta0 = np.linspace(0.0, 2.0 * math.pi, field.shape[1], endpoint=False)
    theta_ext = np.concatenate([theta0, [2.0 * math.pi]])
    theta1 = np.linspace(0.0, 2.0 * math.pi, int(num_segments), endpoint=False)
    out = np.zeros((int(num_rings), int(num_segments)), dtype=np.float64)
    for ridx in range(int(num_rings)):
        row_ext = np.concatenate([tmp[ridx], [tmp[ridx, 0]]])
        out[ridx] = np.interp(theta1, theta_ext, row_ext)
    projected = _project_radius_field(cfg, torch.as_tensor(out, dtype=torch.float32)).detach().cpu().numpy()
    return projected[0]


def effective_ring_profile(radius_field: torch.Tensor) -> torch.Tensor:
    if radius_field.ndim == 2:
        radius_field = radius_field.unsqueeze(0)
    return torch.sqrt(torch.mean(torch.clamp(radius_field, min=1.0e-12).pow(2), dim=2))


def _surface_terms(cfg, radius_field: torch.Tensor) -> Dict[str, torch.Tensor]:
    if radius_field.ndim == 2:
        radius_field = radius_field.unsqueeze(0)
    r = torch.clamp(radius_field, min=float(cfg.min_radius))
    dz = float(cfg.height) / max(int(r.shape[1]) - 1, 1)
    dtheta = 2.0 * math.pi / max(int(r.shape[2]), 1)
    weights = torch.as_tensor(axial_weights_np(r.shape[1], float(cfg.height)), dtype=torch.float32, device=r.device)
    dr_dz = torch.gradient(r, spacing=dz, dim=1)[0] if r.shape[1] > 1 else torch.zeros_like(r)
    dr_dtheta = (torch.roll(r, shifts=-1, dims=2) - torch.roll(r, shifts=1, dims=2)) / max(2.0 * dtheta, 1.0e-12)
    density = torch.sqrt(r.pow(2) * (1.0 + dr_dz.pow(2)) + dr_dtheta.pow(2))
    area = torch.sum(density * weights.view(1, -1, 1) * dtheta, dim=(1, 2))
    cylinder_area = 2.0 * math.pi * float(cfg.radius) * float(cfg.height)
    area_ratio = area / max(cylinder_area, 1.0e-12)
    roughness = torch.std(r, dim=(1, 2)) / torch.clamp(torch.mean(r, dim=(1, 2)), min=1.0e-12)
    circum_nonuniformity = torch.mean(torch.std(r, dim=2), dim=1) / torch.clamp(torch.mean(r, dim=(1, 2)), min=1.0e-12)
    view_factor = torch.exp(-0.20 * roughness - 0.35 * circum_nonuniformity)
    surface_gain = torch.clamp(area_ratio * view_factor, min=0.35, max=1.50)
    return {
        "surface_area_ratio": area_ratio,
        "surface_view_factor": view_factor,
        "surface_gain": surface_gain,
        "surface_roughness": roughness,
        "circum_nonuniformity": circum_nonuniformity,
    }


def _field_feature_change(cfg, radius_field: torch.Tensor) -> torch.Tensor:
    ref_d = float(getattr(cfg, "feature_reference_diameter_m", 2.0 * cfg.radius))
    return torch.amax(torch.abs(2.0 * radius_field - ref_d) / max(ref_d, 1.0e-12), dim=(1, 2))


def evaluate_radius_fields(cfg, radius_fields: torch.Tensor, baseline: Dict[str, torch.Tensor] | None = None):
    if radius_fields.ndim == 2:
        radius_fields = radius_fields.unsqueeze(0)
    initial_volume = math.pi * float(cfg.radius) ** 2 * float(cfg.height)
    metrics = search_rated_condition_3d_batch(cfg, radius_fields, initial_volume)
    if baseline is None:
        baseline = metrics
    transient = simulate_transient_trajectory_3d(
        cfg=cfg,
        radius_field=radius_fields,
        voltage_schedule=metrics["voltage_v"] * float(getattr(cfg, "transient_default_voltage_ratio", 1.0)),
        t_max=float(getattr(cfg, "transient_max_time_s", 120.0)),
        dt=float(getattr(cfg, "transient_dt_s", 0.5)),
    )
    # summarize_transient_selection expects temperature shaped like (B, steps+1, R).
    # For 3D fields, reduce over theta by taking the per-ring peak temperature.
    transient_for_summary = dict(transient)
    if isinstance(transient_for_summary.get("temperature_k"), torch.Tensor) and transient_for_summary["temperature_k"].ndim == 4:
        transient_for_summary["temperature_k"] = torch.amax(transient_for_summary["temperature_k"], dim=3)
    transient_summary = summarize_transient_selection(
        cfg,
        transient_for_summary,
        baseline_power_w=torch.clamp(baseline["initial_net_band_power_w"], min=1.0e-9),
    )
    surface = _surface_terms(cfg, radius_fields)
    adjusted_initial = metrics["initial_net_band_power_w"] * surface["surface_gain"]
    adjusted_average = metrics["average_net_band_power_w"] * surface["surface_gain"]
    adjusted_transient = transient_summary["transient_power_w"] * surface["surface_gain"]
    adjusted_lifetime = metrics["lifetime_s"] / torch.clamp(surface["surface_area_ratio"], min=1.0)
    baseline_initial = torch.clamp(baseline["initial_net_band_power_w"], min=1.0e-9)
    baseline_average = torch.clamp(baseline["average_net_band_power_w"], min=1.0e-9)
    baseline_life = torch.clamp(baseline["lifetime_s"], min=1.0e-9)
    initial_ratio = adjusted_initial / baseline_initial
    average_ratio = adjusted_average / baseline_average
    lifetime_ratio = adjusted_lifetime / baseline_life
    transient_ratio = adjusted_transient / baseline_initial
    feature_change = metrics.get("feature_change_ratio_3d", _field_feature_change(cfg, radius_fields))
    volume = torch.as_tensor(
        [radius_field_volume(field.detach().cpu().numpy(), float(cfg.height)) for field in radius_fields],
        dtype=torch.float32,
        device=radius_fields.device,
    )
    volume_change = torch.abs(volume - initial_volume) / max(initial_volume, 1.0e-12)
    valid = (
        metrics["feasible"]
        & (feature_change < float(cfg.feature_fail_ratio))
        & (volume_change <= float(cfg.volume_tolerance_ratio))
        & (lifetime_ratio >= float(cfg.minimum_lifetime_ratio))
    )
    score = (
        1.20 * initial_ratio
        + 0.95 * average_ratio
        + 0.30 * transient_ratio
        + 0.10 * metrics["temperature_uniformity"]
        + 0.05 * torch.clamp((lifetime_ratio - float(cfg.minimum_lifetime_ratio)) / 0.70, 0.0, 1.0)
        - 70.0 * torch.clamp(float(cfg.minimum_lifetime_ratio) - lifetime_ratio, min=0.0)
        - 80.0 * torch.clamp(feature_change - float(cfg.feature_fail_ratio), min=0.0)
        - 60.0 * volume_change
        - 3.0 * surface["circum_nonuniformity"]
    )
    score = torch.where(valid, score, score - 25.0)
    metrics = dict(metrics)
    metrics.update(surface)
    metrics.update(
        {
            "adjusted_initial_power_w": adjusted_initial,
            "adjusted_average_power_w": adjusted_average,
            "adjusted_transient_power_w": adjusted_transient,
            "adjusted_lifetime_s": adjusted_lifetime,
            "adjusted_lifetime_ratio": lifetime_ratio,
            "feature_change_ratio_3d": feature_change,
            "volume_change_ratio_3d": volume_change,
            "constraint_feasible_3d": valid,
        }
    )
    return metrics, transient_summary, score


def _extract_metrics(metrics: Dict[str, torch.Tensor], transient: Dict[str, torch.Tensor], score: torch.Tensor, idx: int) -> Dict[str, float]:
    out: Dict[str, float] = {"score": float(score[idx].item())}
    for source in (metrics, transient):
        for key, value in source.items():
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                item = value[idx]
                out[key] = bool(item.item()) if item.dtype == torch.bool else float(item.item())
    return out


def _seed_coefficients(num_coeffs: int, axial_modes: int, circum_modes: int) -> list[np.ndarray]:
    seeds = [np.zeros((num_coeffs,), dtype=np.float64)]
    if num_coeffs:
        for amp in (-0.01, -0.015, 0.01):
            seed = np.zeros((num_coeffs,), dtype=np.float64)
            seed[0] = amp
            seeds.append(seed)
    if circum_modes >= 1 and axial_modes >= 1:
        for idx in (1, 2):
            if idx < num_coeffs:
                for amp in (-0.012, 0.012):
                    seed = np.zeros((num_coeffs,), dtype=np.float64)
                    seed[idx] = amp
                    seeds.append(seed)
    return seeds


def run_hybrid_3d_optimization(
    cfg,
    generations: int = 4,
    population_size: int = 16,
    elite_fraction: float = 0.25,
    axial_modes: int = 4,
    circum_modes: int = 2,
    seed: int = 42,
) -> Hybrid3DResult:
    rng = np.random.default_rng(int(seed))
    basis = build_3d_basis(cfg.num_rings, cfg.num_segments, axial_modes, circum_modes)
    num_coeffs = int(basis.shape[0])
    baseline_field_t = fields_from_coefficients(cfg, np.zeros((1, num_coeffs)), basis)
    baseline_metrics_t, baseline_transient_t, baseline_score_t = evaluate_radius_fields(cfg, baseline_field_t)
    baseline_metrics = _extract_metrics(baseline_metrics_t, baseline_transient_t, baseline_score_t, 0)
    history_fields = [baseline_field_t[0].detach().cpu().numpy().copy()]
    history_metrics = [dict(baseline_metrics, step=0, generation=0)]
    archive_fields = [history_fields[0].copy()]
    archive_metrics = [dict(history_metrics[0], archive_index=0)]
    best_field = history_fields[0].copy()
    best_metrics = dict(history_metrics[0])
    best_score = float(baseline_score_t[0].item())
    best_coeff = np.zeros((num_coeffs,), dtype=np.float64)
    mean = np.zeros_like(best_coeff)
    sigma = np.full_like(mean, 0.035)
    seeds = _seed_coefficients(num_coeffs, axial_modes, circum_modes)
    elite_count = max(2, int(round(float(population_size) * float(elite_fraction))))
    candidate_count = 1

    for generation in range(1, int(generations) + 1):
        samples = rng.normal(mean, sigma, size=(max(int(population_size), len(seeds)), num_coeffs))
        population = np.vstack(seeds + [row for row in samples])[: int(population_size)] if generation == 1 else samples[: int(population_size)]
        fields = fields_from_coefficients(cfg, population, basis)
        metrics, transient, scores = evaluate_radius_fields(cfg, fields, baseline_metrics_t)
        candidate_count += int(population.shape[0])
        for idx in range(int(population.shape[0])):
            archive_fields.append(fields[idx].detach().cpu().numpy().copy())
            archive_metrics.append(dict(_extract_metrics(metrics, transient, scores, idx), archive_index=len(archive_fields) - 1, generation=generation))
        order = torch.argsort(scores, descending=True).detach().cpu().numpy()
        elites = population[order[:elite_count]]
        elite_scores = scores[order[:elite_count]].detach().cpu().numpy()
        weights = np.exp((elite_scores - np.max(elite_scores)) / max(float(np.std(elite_scores)), 1.0e-6))
        weights = weights / np.sum(weights)
        mean = np.sum(elites * weights[:, None], axis=0)
        sigma = np.clip(np.sqrt(np.sum(weights[:, None] * (elites - mean) ** 2, axis=0)) * 0.85 + sigma * 0.15, 0.004, 0.055)
        idx = int(order[0])
        if float(scores[idx].item()) > best_score:
            best_score = float(scores[idx].item())
            best_coeff = population[idx].copy()
            best_field = fields[idx].detach().cpu().numpy().copy()
            best_metrics = _extract_metrics(metrics, transient, scores, idx)
        history_fields.append(best_field.copy())
        history_metrics.append(dict(best_metrics, step=len(history_fields) - 1, generation=generation))

    return Hybrid3DResult(
        best_field=best_field,
        baseline_field=history_fields[0],
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        history_fields=history_fields,
        history_metrics=history_metrics,
        archive_fields=archive_fields,
        archive_metrics=archive_metrics,
        candidate_count=candidate_count,
    )


def export_radius_field_mesh(radius_field: np.ndarray, cfg, output_dir: str, output_name: str = "optimized_cylinder_3d", export_step: bool = True):
    points = points_from_radius_field(radius_field, float(cfg.height))
    return export_mesh_files(
        points=points,
        num_segments=int(radius_field.shape[1]),
        num_rings=int(radius_field.shape[0]),
        output_dir=output_dir,
        output_name=output_name,
        export_step=export_step,
        freecad_cmd=getattr(cfg, "freecad_cmd", ""),
        freecad_timeout_s=float(getattr(cfg, "freecad_timeout_s", 90.0)),
    )


def export_3d_evolution_animation(fields: list[np.ndarray], metrics_history: list[Dict[str, float]], cfg, output_dir: str, output_name: str = "topology_evolution_3d", fps: int = 3) -> Dict[str, str | None]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frames = []
    theta = np.linspace(0.0, 2.0 * math.pi, int(cfg.num_segments), endpoint=False)
    z = np.linspace(-0.5 * float(cfg.height), 0.5 * float(cfg.height), int(cfg.num_rings)) * 1.0e3
    theta_g, z_g = np.meshgrid(theta, z)
    for idx, field in enumerate(fields):
        r_mm = np.asarray(field) * 1.0e3
        x = r_mm * np.cos(theta_g)
        y = r_mm * np.sin(theta_g)
        fig = plt.figure(figsize=(9, 4.8), dpi=120)
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot_surface(x, z_g, y, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        ax.set_xlabel("x mm")
        ax.set_ylabel("z mm")
        ax.set_zlabel("y mm")
        ax.set_box_aspect((1, 2.2, 1))
        ax.view_init(elev=18, azim=-62)
        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow((r_mm - float(cfg.radius) * 1.0e3), aspect="auto", cmap="coolwarm", origin="lower", extent=[0, 360, z[0], z[-1]])
        ax2.set_xlabel("theta deg")
        ax2.set_ylabel("z mm")
        ax2.set_title("radial change (mm)")
        fig.colorbar(im, ax=ax2, shrink=0.75)
        m = metrics_history[min(idx, len(metrics_history) - 1)] if metrics_history else {}
        fig.suptitle(
            f"step={idx}  P3D={m.get('adjusted_initial_power_w', 0.0):.2f}W  "
            f"life={m.get('adjusted_lifetime_ratio', 1.0):.3f}  feasible={m.get('constraint_feasible_3d', True)}",
            fontsize=10,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    gif_path = output_path / f"{output_name}.gif"
    mp4_path = output_path / f"{output_name}.mp4"
    imageio.mimsave(gif_path, frames, duration=max(1.0 / fps, 0.1), loop=0)
    mp4_written = None
    try:
        with imageio.get_writer(str(mp4_path), fps=fps, codec="libx264", quality=7, macro_block_size=16) as writer:
            for frame in frames:
                writer.append_data(frame)
        mp4_written = str(mp4_path)
    except Exception:
        mp4_written = None
    return {"gif": str(gif_path), "mp4": mp4_written}


def save_3d_history_csv(output_dir: str | Path, history: List[Dict[str, float]]) -> str:
    path = Path(output_dir) / "optimization_history_3d.csv"
    if not history:
        path.write_text("", encoding="utf-8")
        return str(path)
    keys = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    return str(path)


def save_3d_summary(output_dir: str | Path, summary: Dict[str, object]) -> str:
    path = Path(output_dir) / "run_summary_3d.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def write_3d_strategy_report(output_dir: str | Path, result: Hybrid3DResult, summary: Dict[str, object]) -> str:
    path = Path(output_dir) / "design_strategy_report_3d.md"
    lines = [
        "# 三维混合物理优化策略报告",
        "",
        "## 方案定位",
        "",
        "- 本路线直接优化三维表面半径场 `r(z, theta)`，不是二维剖面旋转。",
        "- 设计变量采用轴向 Chebyshev 基和周向 Fourier 基，输出为非轴对称三维 mesh。",
        "- `100V` 仍只作为上限；每个三维候选先搜索可行额定电压，再搜索瞬态窗口内最佳采样时间。",
        "",
        "## 细网格结果",
        "",
        f"- 三维有效初始功率提升：`{summary.get('initial_power_ratio_3d', 0.0):.4f}x`",
        f"- 三维有效平均功率提升：`{summary.get('average_power_ratio_3d', 0.0):.4f}x`",
        f"- 寿命比例：`{summary.get('lifetime_ratio_3d', 0.0):.4f}`",
        f"- 体积偏差：`{summary.get('volume_change_ratio_3d', 0.0):.6g}`",
        f"- 三维特征尺度变化：`{summary.get('feature_change_ratio_3d', 0.0):.6g}`",
        "",
        "## 设计启发",
        "",
        "- 相比大幅 2D 收颈，三维优化倾向于使用很浅的周向起伏和轴向微调，增加有效表面积，同时避免局部蒸发寿命快速下降。",
        "- 可行设计的关键不是把电压推到 100V，而是在当前几何下找到不过温的额定电压。",
        "- 三维 archive 重评估能过滤掉训练粗网格里看似高功率但寿命不足的候选。",
        "",
        "## 产物",
        "",
        f"- STL：`{summary.get('stl')}`",
        f"- STP：`{summary.get('stp')}`",
        f"- 动画 GIF：`{summary.get('gif')}`",
        f"- 动画 MP4：`{summary.get('mp4')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
