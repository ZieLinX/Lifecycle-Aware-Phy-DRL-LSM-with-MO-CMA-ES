from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from utils.feasibility import project_connected_profile_batch
from utils.rated_condition import (
    search_rated_condition_batch,
    simulate_transient_trajectory,
    summarize_transient_selection,
)


@dataclass(frozen=True)
class HybridOptimizationResult:
    best_profile: np.ndarray
    baseline_profile: np.ndarray
    best_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    history_profiles: List[np.ndarray]
    history_metrics: List[Dict[str, float]]
    archive_profiles: List[np.ndarray]
    archive_metrics: List[Dict[str, float]]
    candidate_count: int


def axial_weights_np(num_rings: int, height: float) -> np.ndarray:
    weights = np.full((int(num_rings),), float(height) / max(int(num_rings) - 1, 1), dtype=np.float64)
    if int(num_rings) > 1:
        weights[0] *= 0.5
        weights[-1] *= 0.5
    return weights


def profile_volume(profile: np.ndarray, height: float) -> float:
    radius = np.asarray(profile, dtype=np.float64).reshape(-1)
    weights = axial_weights_np(radius.size, height)
    return float(np.sum(math.pi * radius * radius * weights))


def build_chebyshev_basis(num_rings: int, num_modes: int) -> np.ndarray:
    z = np.linspace(-1.0, 1.0, int(num_rings), dtype=np.float64)
    envelope = np.clip(1.0 - z * z, 0.0, 1.0)
    basis = []
    t_prev = np.ones_like(z)
    if int(num_modes) >= 1:
        basis.append(envelope * t_prev)
    if int(num_modes) >= 2:
        t_cur = z.copy()
        basis.append(envelope * t_cur)
        for _ in range(2, int(num_modes)):
            t_next = 2.0 * z * t_cur - t_prev
            basis.append(envelope * t_next)
            t_prev, t_cur = t_cur, t_next
    out = np.stack(basis, axis=0) if basis else np.zeros((0, int(num_rings)), dtype=np.float64)
    norm = np.max(np.abs(out), axis=1, keepdims=True)
    return out / np.clip(norm, 1.0e-12, None)


def _project_volume_and_connectivity(cfg, profiles: torch.Tensor) -> torch.Tensor:
    if profiles.ndim == 1:
        profiles = profiles.unsqueeze(0)
    radius = torch.clamp(profiles.float(), min=float(cfg.min_radius))
    radius[:, 0] = float(cfg.radius)
    radius[:, -1] = float(cfg.radius)

    max_step = float(getattr(cfg, "feasibility_area_ratio_max", 5.0)) ** 0.5
    weights = torch.as_tensor(
        axial_weights_np(radius.shape[1], float(cfg.height)),
        dtype=torch.float32,
        device=radius.device,
    ).unsqueeze(0)
    target_volume = math.pi * float(cfg.radius) * float(cfg.radius) * float(cfg.height)
    endpoint_volume = math.pi * float(cfg.radius) ** 2 * float(weights[:, [0, -1]].sum().item())
    target_interior = max(target_volume - endpoint_volume, 1.0e-12)
    dz = float(cfg.height) / max(int(radius.shape[1]) - 1, 1)
    slope_delta = float(getattr(cfg, "feasibility_max_slope", 0.35)) * dz

    def limit_slope(r: torch.Tensor) -> torch.Tensor:
        if r.shape[1] <= 1:
            return r
        out = r.clone()
        for _ in range(3):
            out[:, 0] = float(cfg.radius)
            out[:, -1] = float(cfg.radius)
            for idx in range(1, out.shape[1] - 1):
                out[:, idx] = torch.minimum(out[:, idx], out[:, idx - 1] + slope_delta)
                out[:, idx] = torch.maximum(out[:, idx], out[:, idx - 1] - slope_delta)
            for idx in range(out.shape[1] - 2, 0, -1):
                out[:, idx] = torch.minimum(out[:, idx], out[:, idx + 1] + slope_delta)
                out[:, idx] = torch.maximum(out[:, idx], out[:, idx + 1] - slope_delta)
        return torch.clamp(out, min=float(cfg.min_radius))

    for _ in range(5):
        radius = project_connected_profile_batch(
            radius,
            min_radius=float(cfg.min_radius),
            max_step_ratio=max_step,
            fix_endpoints=bool(getattr(cfg, "keep_electrode_rings_fixed", True)),
        )
        radius = limit_slope(radius)
        radius[:, 0] = float(cfg.radius)
        radius[:, -1] = float(cfg.radius)
        if radius.shape[1] <= 2:
            continue
        interior = radius[:, 1:-1]
        interior_weights = weights[:, 1:-1]
        current_interior = torch.sum(math.pi * interior.pow(2) * interior_weights, dim=1, keepdim=True)
        scale = torch.sqrt(torch.as_tensor(target_interior, dtype=torch.float32, device=radius.device) / torch.clamp(current_interior, min=1.0e-12))
        radius[:, 1:-1] = torch.clamp(interior * scale, min=float(cfg.min_radius))

    radius = limit_slope(radius)
    radius[:, 0] = float(cfg.radius)
    radius[:, -1] = float(cfg.radius)
    return radius


def profiles_from_coefficients(cfg, coefficients: np.ndarray, basis: np.ndarray) -> torch.Tensor:
    coeff = np.asarray(coefficients, dtype=np.float64)
    if coeff.ndim == 1:
        coeff = coeff.reshape(1, -1)
    log_delta = coeff @ np.asarray(basis, dtype=np.float64)
    log_delta = np.clip(log_delta, -0.35, 0.35)
    raw = float(cfg.radius) * np.exp(log_delta)
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    return _project_volume_and_connectivity(cfg, torch.as_tensor(raw, dtype=torch.float32, device=device))


def resample_profile(cfg, profile: np.ndarray, num_rings: int) -> np.ndarray:
    source = np.asarray(profile, dtype=np.float64).reshape(-1)
    z0 = np.linspace(-1.0, 1.0, source.size)
    z1 = np.linspace(-1.0, 1.0, int(num_rings))
    sampled = np.interp(z1, z0, source)
    projected = _project_volume_and_connectivity(
        cfg,
        torch.as_tensor(sampled, dtype=torch.float32).unsqueeze(0),
    )[0]
    return projected.detach().cpu().numpy()


def _extract_metrics(metrics: Dict[str, torch.Tensor], transient: Dict[str, torch.Tensor], scores: torch.Tensor, idx: int) -> Dict[str, float]:
    out: Dict[str, float] = {"score": float(scores[idx].item())}
    scalar_keys = [
        "voltage_v",
        "current_a",
        "resistance_ohm",
        "mean_temperature_k",
        "max_temperature_k",
        "initial_net_band_power_w",
        "average_net_band_power_w",
        "band_efficiency",
        "lifetime_s",
        "mass_loss_rate_kg_s",
        "view_factor_proxy",
        "temperature_uniformity",
        "min_equivalent_diameter_mm",
        "feature_change_ratio",
        "volume_change_ratio",
        "smoothness_penalty",
        "feasibility_penalty",
        "thermo_mech_penalty",
        "min_neck_diameter_mm",
        "max_radius_slope",
        "max_axial_stress_pa",
        "rated_utility",
        "thermal_residual_k",
    ]
    for key in scalar_keys:
        value = metrics.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = float(value[idx].item())
    out["thermal_converged"] = bool(metrics["thermal_converged"][idx].item())
    out["feasible"] = bool(metrics["feasible"][idx].item())
    for key, value in transient.items():
        if isinstance(value, torch.Tensor) and value.ndim == 1:
            out[key] = float(value[idx].item())
    return out


def _score_metrics(cfg, metrics: Dict[str, torch.Tensor], transient: Dict[str, torch.Tensor], baseline: Dict[str, torch.Tensor]) -> torch.Tensor:
    initial_ratio = metrics["initial_net_band_power_w"] / torch.clamp(baseline["initial_net_band_power_w"], min=1.0e-9)
    average_ratio = metrics["average_net_band_power_w"] / torch.clamp(baseline["average_net_band_power_w"], min=1.0e-9)
    lifetime_ratio = metrics["lifetime_s"] / torch.clamp(baseline["lifetime_s"], min=1.0e-9)
    min_life = float(cfg.minimum_lifetime_ratio)
    lifetime_margin = torch.clamp((lifetime_ratio - min_life) / max(1.0 - min_life, 1.0e-6), min=0.0, max=1.0)
    score = (
        float(cfg.reward_weight_initial_power) * initial_ratio
        + float(cfg.reward_weight_average_power) * average_ratio
        + 0.05 * lifetime_margin
        + float(cfg.reward_weight_uniformity) * metrics["temperature_uniformity"]
        + float(cfg.reward_weight_efficiency) * metrics["band_efficiency"]
        + float(cfg.reward_weight_transient_power) * transient["transient_power_ratio"]
        + 0.20 * transient["transient_mean_power_ratio"]
        - float(cfg.reward_penalty_feasibility) * metrics["feasibility_penalty"]
        - float(cfg.reward_weight_thermomech) * metrics["thermo_mech_penalty"]
        - float(cfg.reward_penalty_mass_loss) * torch.clamp(metrics["mass_loss_rate_kg_s"] - float(cfg.max_mass_loss_rate), min=0.0)
        - float(cfg.reward_penalty_temp_violation) * torch.clamp(metrics["max_temperature_k"] - float(cfg.max_temp), min=0.0).pow(2)
        - float(cfg.reward_penalty_feature_violation) * torch.clamp(metrics["feature_change_ratio"] - float(cfg.feature_fail_ratio), min=0.0)
        - float(cfg.reward_penalty_volume_change) * metrics["volume_change_ratio"]
    )
    life_violation = torch.clamp(min_life - lifetime_ratio, min=0.0)
    score = score - 100.0 * life_violation
    score = torch.where(metrics["feasible"], score, score - 100.0)
    return score


def evaluate_profiles(cfg, profiles: torch.Tensor, baseline: Dict[str, torch.Tensor] | None = None) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    if profiles.ndim == 1:
        profiles = profiles.unsqueeze(0)
    initial_volume = math.pi * float(cfg.radius) * float(cfg.radius) * float(cfg.height)
    metrics = search_rated_condition_batch(cfg, profiles, initial_volume)
    if baseline is None:
        baseline = metrics
    transient = simulate_transient_trajectory(
        cfg=cfg,
        ring_radius=profiles,
        voltage_schedule=metrics["voltage_v"] * float(getattr(cfg, "transient_default_voltage_ratio", 1.0)),
        t_max=float(getattr(cfg, "transient_max_time_s", 120.0)),
        dt=float(getattr(cfg, "transient_dt_s", 0.5)),
    )
    transient_summary = summarize_transient_selection(
        cfg,
        transient,
        baseline_power_w=torch.clamp(baseline["initial_net_band_power_w"], min=1.0e-9),
    )
    scores = _score_metrics(cfg, metrics, transient_summary, baseline)
    return metrics, transient_summary, scores


def _initial_seed_coefficients(num_modes: int) -> list[np.ndarray]:
    seeds = [np.zeros((num_modes,), dtype=np.float64)]
    if num_modes >= 1:
        tiny_waist = np.zeros((num_modes,), dtype=np.float64)
        tiny_waist[0] = -0.01
        seeds.append(tiny_waist)
        tiny_bulge = np.zeros((num_modes,), dtype=np.float64)
        tiny_bulge[0] = 0.01
        seeds.append(tiny_bulge)
        small_waist = np.zeros((num_modes,), dtype=np.float64)
        small_waist[0] = -0.015
        seeds.append(small_waist)
        mild_waist = np.zeros((num_modes,), dtype=np.float64)
        mild_waist[0] = -0.03
        seeds.append(mild_waist)
        mild_bulge = np.zeros((num_modes,), dtype=np.float64)
        mild_bulge[0] = 0.03
        seeds.append(mild_bulge)
        medium_waist = np.zeros((num_modes,), dtype=np.float64)
        medium_waist[0] = -0.06
        seeds.append(medium_waist)
        medium_bulge = np.zeros((num_modes,), dtype=np.float64)
        medium_bulge[0] = 0.06
        seeds.append(medium_bulge)
        center_waist = np.zeros((num_modes,), dtype=np.float64)
        center_waist[0] = -0.10
        seeds.append(center_waist)
        center_bulge = np.zeros((num_modes,), dtype=np.float64)
        center_bulge[0] = 0.08
        seeds.append(center_bulge)
    if num_modes >= 3:
        mild_saddle = np.zeros((num_modes,), dtype=np.float64)
        mild_saddle[2] = -0.04
        seeds.append(mild_saddle)
        mild_inverse_saddle = np.zeros((num_modes,), dtype=np.float64)
        mild_inverse_saddle[2] = 0.04
        seeds.append(mild_inverse_saddle)
        saddle = np.zeros((num_modes,), dtype=np.float64)
        saddle[0] = -0.04
        saddle[2] = 0.12
        seeds.append(saddle)
        inverse_saddle = np.zeros((num_modes,), dtype=np.float64)
        inverse_saddle[0] = 0.05
        inverse_saddle[2] = -0.10
        seeds.append(inverse_saddle)
    return seeds


def _merge_population(samples: np.ndarray, seeds: Iterable[np.ndarray], population_size: int) -> np.ndarray:
    merged = list(seeds)
    merged.extend(np.asarray(samples, dtype=np.float64))
    return np.stack(merged[: int(population_size)], axis=0)


def run_hybrid_optimization(
    cfg,
    generations: int = 8,
    population_size: int = 18,
    elite_fraction: float = 0.25,
    num_modes: int = 8,
    local_iterations: int = 2,
    seed: int = 42,
) -> HybridOptimizationResult:
    rng = np.random.default_rng(int(seed))
    basis = build_chebyshev_basis(int(cfg.num_rings), int(num_modes))
    baseline_profile_t = profiles_from_coefficients(cfg, np.zeros((1, int(num_modes))), basis)
    baseline_metrics_t, baseline_transient_t, baseline_scores_t = evaluate_profiles(cfg, baseline_profile_t)
    baseline_metrics = _extract_metrics(baseline_metrics_t, baseline_transient_t, baseline_scores_t, 0)

    mean = np.zeros((int(num_modes),), dtype=np.float64)
    sigma = np.full_like(mean, 0.12)
    elite_count = max(2, int(round(float(population_size) * float(elite_fraction))))
    history_profiles: List[np.ndarray] = [baseline_profile_t[0].detach().cpu().numpy().copy()]
    history_metrics: List[Dict[str, float]] = [dict(baseline_metrics, step=0, generation=0)]
    archive_profiles: List[np.ndarray] = [history_profiles[0].copy()]
    archive_metrics: List[Dict[str, float]] = [dict(history_metrics[0], archive_index=0)]
    best_coeff = mean.copy()
    best_score = float(baseline_scores_t[0].item())
    best_profile = history_profiles[0].copy()
    best_metrics = dict(history_metrics[0])
    candidate_count = 1

    seed_coeffs = _initial_seed_coefficients(int(num_modes))
    for generation in range(1, int(generations) + 1):
        samples = rng.normal(loc=mean, scale=sigma, size=(max(int(population_size), len(seed_coeffs)), int(num_modes)))
        if generation == 1:
            population = _merge_population(samples, seed_coeffs, int(population_size))
        else:
            population = samples[: int(population_size)]

        profiles = profiles_from_coefficients(cfg, population, basis)
        metrics, transient, scores = evaluate_profiles(cfg, profiles, baseline_metrics_t)
        candidate_count += int(population.shape[0])
        for archive_idx in range(int(population.shape[0])):
            archive_profiles.append(profiles[archive_idx].detach().cpu().numpy().copy())
            archive_metrics.append(
                dict(
                    _extract_metrics(metrics, transient, scores, archive_idx),
                    archive_index=len(archive_profiles) - 1,
                    generation=generation,
                )
            )
        order = torch.argsort(scores, descending=True).detach().cpu().numpy()
        elites = population[order[:elite_count]]
        elite_scores = scores[order[:elite_count]].detach().cpu().numpy()
        weights = np.exp((elite_scores - np.max(elite_scores)) / max(np.std(elite_scores), 1.0e-6))
        weights = weights / np.sum(weights)
        mean = np.sum(elites * weights[:, None], axis=0)
        variance = np.sum(weights[:, None] * (elites - mean) ** 2, axis=0)
        sigma = np.clip(np.sqrt(variance) * 0.80 + sigma * 0.20, 0.015, 0.18)

        gen_best_idx = int(order[0])
        gen_best_score = float(scores[gen_best_idx].item())
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_coeff = population[gen_best_idx].copy()
            best_profile = profiles[gen_best_idx].detach().cpu().numpy().copy()
            best_metrics = _extract_metrics(metrics, transient, scores, gen_best_idx)
        history_profiles.append(best_profile.copy())
        history_metrics.append(dict(best_metrics, step=len(history_profiles) - 1, generation=generation))

    step = np.full((int(num_modes),), 0.045, dtype=np.float64)
    for local_iter in range(int(local_iterations)):
        improved = False
        for dim in range(int(num_modes)):
            trials = []
            for direction in (-1.0, 1.0):
                coeff = best_coeff.copy()
                coeff[dim] += direction * step[dim]
                trials.append(coeff)
            population = np.stack(trials, axis=0)
            profiles = profiles_from_coefficients(cfg, population, basis)
            metrics, transient, scores = evaluate_profiles(cfg, profiles, baseline_metrics_t)
            candidate_count += int(population.shape[0])
            for archive_idx in range(int(population.shape[0])):
                archive_profiles.append(profiles[archive_idx].detach().cpu().numpy().copy())
                archive_metrics.append(
                    dict(
                        _extract_metrics(metrics, transient, scores, archive_idx),
                        archive_index=len(archive_profiles) - 1,
                        generation=int(generations),
                        local_iter=local_iter + 1,
                    )
                )
            idx = int(torch.argmax(scores).item())
            trial_score = float(scores[idx].item())
            if trial_score > best_score:
                best_score = trial_score
                best_coeff = population[idx].copy()
                best_profile = profiles[idx].detach().cpu().numpy().copy()
                best_metrics = _extract_metrics(metrics, transient, scores, idx)
                history_profiles.append(best_profile.copy())
                history_metrics.append(dict(best_metrics, step=len(history_profiles) - 1, generation=int(generations), local_iter=local_iter + 1))
                improved = True
        step *= 0.55 if improved else 0.40

    return HybridOptimizationResult(
        best_profile=best_profile,
        baseline_profile=history_profiles[0],
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        history_profiles=history_profiles,
        history_metrics=history_metrics,
        archive_profiles=archive_profiles,
        archive_metrics=archive_metrics,
        candidate_count=candidate_count,
    )


def write_strategy_report(output_dir: str | Path, result: HybridOptimizationResult, cfg, summary: Dict[str, object]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "design_strategy_report.md"

    z_mm = np.linspace(-0.5 * float(cfg.height), 0.5 * float(cfg.height), result.best_profile.size) * 1.0e3
    delta_mm = (result.best_profile - result.baseline_profile) * 1.0e3
    threshold = 0.003
    zones = []
    active = np.where(np.abs(delta_mm) >= threshold)[0]
    if active.size:
        splits = np.where(np.diff(active) > 1)[0] + 1
        for group in np.split(active, splits):
            sign = "增材/外鼓" if float(np.mean(delta_mm[group])) > 0.0 else "减材/收颈"
            zones.append(
                f"- {sign}: z={z_mm[group[0]]:.2f} 到 {z_mm[group[-1]]:.2f} mm, "
                f"平均半径变化 {float(np.mean(delta_mm[group])):.3f} mm"
            )
    else:
        zones.append("- 几何变化很小，当前最优仍接近基准圆柱。")

    base = result.baseline_metrics
    best = result.best_metrics
    lines = [
        "# 混合物理优化设计策略报告",
        "",
        "## 建模口径",
        "",
        "- `100V` 只作为额定工况搜索上限，不作为固定工作电压，也不作为奖励目标。",
        "- 每个候选几何先搜索 `V <= 100V` 的额定工况，再在瞬态窗口内搜索最优采样时间。",
        "- 几何采用轴对称 ring profile 表示，电极端面半径固定，材料总体积通过投影保持守恒。",
        "",
        "## 指标对比",
        "",
        "| 指标 | 初始圆柱 | 优化设计 |",
        "| --- | ---: | ---: |",
        f"| 额定电压 V* (V) | {base.get('voltage_v', 0.0):.3f} | {best.get('voltage_v', 0.0):.3f} |",
        f"| 0-3um 初始净功率 (W) | {base.get('initial_net_band_power_w', 0.0):.6f} | {best.get('initial_net_band_power_w', 0.0):.6f} |",
        f"| 生命周期平均净功率 (W) | {base.get('average_net_band_power_w', 0.0):.6f} | {best.get('average_net_band_power_w', 0.0):.6f} |",
        f"| 寿命 (s) | {base.get('lifetime_s', 0.0):.3f} | {best.get('lifetime_s', 0.0):.3f} |",
        f"| 最高温度 (K) | {base.get('max_temperature_k', 0.0):.3f} | {best.get('max_temperature_k', 0.0):.3f} |",
        f"| 最优瞬态时间 (s) | {base.get('optimal_transient_time_s', 0.0):.3f} | {best.get('optimal_transient_time_s', 0.0):.3f} |",
        f"| 体积偏差 | {base.get('volume_change_ratio', 0.0):.6g} | {best.get('volume_change_ratio', 0.0):.6g} |",
        f"| 特征尺度变化 | {base.get('feature_change_ratio', 0.0):.6g} | {best.get('feature_change_ratio', 0.0):.6g} |",
        "",
        "## 几何变化摘要",
        "",
        *zones,
        "",
        "## 给人类设计者的启发",
        "",
        "- 优先把电压视为由几何诱导的额定工况结果，而不是预设常数；几何改变电阻后，最佳电压会随之漂移。",
        "- 优化目标不应只看稳态末端功率。升温阶段的最佳采样时间可能提前出现，过长工作会快速放大蒸发和温度惩罚。",
        "- 材料重分布要同时看辐射面积、局部电阻发热和蒸发寿命：可解释的设计通常表现为局部收颈提高电阻/温度，同时在相邻区域补偿体积并控制坡度。",
        "- 如果最优形态接近圆柱，说明当前硬约束和代理物理更偏向保守设计；后续应重点校准视因子、3D 遮挡和制造约束，而不是盲目增加策略网络规模。",
        "",
        "## 产物",
        "",
        f"- 候选评估数量：{int(summary.get('candidate_count', result.candidate_count))}",
        f"- STL：`{summary.get('stl')}`",
        f"- STP：`{summary.get('stp')}`",
        f"- 动画 GIF：`{summary.get('gif')}`",
        f"- 动画 MP4：`{summary.get('mp4')}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def save_history_csv(output_dir: str | Path, history: List[Dict[str, float]]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "optimization_history.csv"
    if not history:
        path.write_text("", encoding="utf-8")
        return str(path)
    keys = sorted({key for row in history for key in row.keys()})
    lines = [",".join(keys)]
    for row in history:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, bool):
                values.append("1" if value else "0")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def save_summary_json(output_dir: str | Path, summary: Dict[str, object]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
