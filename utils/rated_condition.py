from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Dict

import numpy as np
import torch
from utils.feasibility import evaluate_feasibility
from utils.thermo_mech import evaluate_thermo_mech


@dataclass(frozen=True)
class RatedConditionMetrics:
    voltage_v: float
    current_a: float
    resistance_ohm: float
    mean_temperature_k: float
    max_temperature_k: float
    initial_net_band_power_w: float
    average_net_band_power_w: float
    band_efficiency: float
    lifetime_s: float
    mass_loss_rate_kg_s: float
    view_factor_proxy: float
    temperature_uniformity: float
    min_equivalent_diameter_mm: float
    feature_change_ratio: float
    volume_change_ratio: float
    smoothness_penalty: float
    feasibility_penalty: float
    thermo_mech_penalty: float
    min_neck_diameter_mm: float
    max_radius_slope: float
    max_axial_stress_pa: float
    feasible: bool
    rated_utility: float
    thermal_iterations: int
    thermal_residual_k: float
    thermal_converged: bool
    ring_temperature_k: torch.Tensor
    ring_recession_rate_m_s: torch.Tensor


def _material_properties(cfg, temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = torch.clamp(temperature - 300.0, min=0.0)
    cp = cfg.cp_ref + cfg.cp_temp_coeff * delta
    k = torch.clamp(cfg.k_ref + cfg.k_temp_coeff * delta, min=20.0)
    rho_elec = cfg.rho_elec_ref * (1.0 + cfg.rho_elec_temp_coeff * delta)
    return cp, k, rho_elec


def _evaporation_flux_kg_m2_s(cfg, temperature: torch.Tensor) -> torch.Tensor:
    y_g_cm2_s = cfg.evap_A * torch.exp(cfg.evap_B / torch.clamp(temperature, min=1.0))
    return y_g_cm2_s * 10.0


def _laplacian_1d(values: torch.Tensor) -> torch.Tensor:
    lap = torch.zeros_like(values)
    if values.shape[-1] == 1:
        return lap
    lap[..., 0] = values[..., 1] - values[..., 0]
    lap[..., -1] = values[..., -2] - values[..., -1]
    if values.shape[-1] > 2:
        lap[..., 1:-1] = values[..., :-2] - 2.0 * values[..., 1:-1] + values[..., 2:]
    return lap


def _axial_weights(num_rings: int, dz: float, device: torch.device) -> torch.Tensor:
    weights = torch.full((num_rings,), float(dz), dtype=torch.float32, device=device)
    if num_rings > 1:
        weights[0] *= 0.5
        weights[-1] *= 0.5
    return weights


@lru_cache(maxsize=1024)
def _blackbody_band_fraction_cached(temp_k_rounded: float, upper_um: float) -> float:
    temp_k = max(float(temp_k_rounded), 100.0)
    wavelengths_um = np.linspace(0.1, 20.0, 2400)
    wavelengths_m = wavelengths_um * 1.0e-6
    c2 = 1.438776877e-2
    exponent = np.clip(c2 / (wavelengths_m * temp_k), 1.0e-9, 700.0)
    spectral_shape = 1.0 / (np.power(wavelengths_m, 5) * (np.exp(exponent) - 1.0))
    total = float(np.trapezoid(spectral_shape, wavelengths_um))
    band_mask = wavelengths_um <= upper_um
    in_band = float(np.trapezoid(spectral_shape[band_mask], wavelengths_um[band_mask]))
    return in_band / max(total, 1.0e-12)


def blackbody_band_fraction(temp_k: float, upper_um: float) -> float:
    return _blackbody_band_fraction_cached(round(float(temp_k), 1), float(upper_um))


@lru_cache(maxsize=32)
def _band_fraction_lookup_table(
    min_temp: float,
    max_temp: float,
    lut_size: int,
    upper_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    temps = np.linspace(float(min_temp), float(max_temp), int(lut_size), dtype=np.float64)
    fractions = np.asarray(
        [blackbody_band_fraction(temp_k=float(temp), upper_um=upper_um) for temp in temps],
        dtype=np.float32,
    )
    return temps.astype(np.float32), fractions


def _band_fraction_tensor(cfg, temperature: torch.Tensor, upper_um: float) -> torch.Tensor:
    lut_temp, lut_fraction = _band_fraction_lookup_table(
        min_temp=float(cfg.band_fraction_min_temp),
        max_temp=float(cfg.band_fraction_max_temp),
        lut_size=int(cfg.band_fraction_lut_size),
        upper_um=float(upper_um),
    )
    lut_temp_t = torch.as_tensor(lut_temp, dtype=torch.float32, device=temperature.device)
    lut_fraction_t = torch.as_tensor(lut_fraction, dtype=torch.float32, device=temperature.device)
    clamped = torch.clamp(temperature, min=float(lut_temp[0]), max=float(lut_temp[-1]))
    upper_idx = torch.bucketize(clamped, lut_temp_t).clamp(1, lut_temp_t.numel() - 1)
    lower_idx = upper_idx - 1
    x0 = lut_temp_t[lower_idx]
    x1 = lut_temp_t[upper_idx]
    y0 = lut_fraction_t[lower_idx]
    y1 = lut_fraction_t[upper_idx]
    weight = (clamped - x0) / torch.clamp(x1 - x0, min=1.0e-6)
    return y0 + weight * (y1 - y0)


def ring_radii_from_points(points: torch.Tensor, ring_index: torch.Tensor, num_rings: int) -> torch.Tensor:
    radial = torch.norm(points[:, :2], dim=1)
    ring_radius = torch.zeros(num_rings, dtype=torch.float32, device=points.device)
    for ridx in range(num_rings):
        mask = ring_index == ridx
        if torch.any(mask):
            ring_radius[ridx] = torch.mean(radial[mask])
    return ring_radius


def _feature_change_ratio(cfg, ring_radius: torch.Tensor) -> float:
    if getattr(cfg, "feature_scale_mode", "equivalent_diameter") == "equivalent_diameter":
        ref_d = max(float(getattr(cfg, "feature_reference_diameter_m", 2.0 * cfg.radius)), 1.0e-12)
        eq_d = 2.0 * ring_radius
        return float(torch.max(torch.abs(eq_d - ref_d) / ref_d).item())
    return float(torch.max(torch.abs(ring_radius - cfg.radius) / max(cfg.radius, 1.0e-12)).item())


def _apply_thermal_boundary_mode(cfg, temperature: torch.Tensor) -> torch.Tensor:
    mode = getattr(cfg, "thermal_boundary_mode", "fixed_room_temp")
    if temperature.shape[-1] <= 1:
        return temperature
    bounded = temperature.clone()
    if mode == "fixed_room_temp":
        bounded[..., 0] = float(cfg.ambient_temp)
        bounded[..., -1] = float(cfg.ambient_temp)
        return bounded
    if mode == "infinite_room_temp":
        bounded[..., 0] = 0.5 * (bounded[..., 0] + float(cfg.ambient_temp))
        bounded[..., -1] = 0.5 * (bounded[..., -1] + float(cfg.ambient_temp))
        return bounded
    raise ValueError(f"Unsupported thermal_boundary_mode: {mode}")


def _to_batch_ring_radius(ring_radius: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if ring_radius.ndim == 1:
        return ring_radius.unsqueeze(0), True
    if ring_radius.ndim == 2:
        return ring_radius, False
    raise ValueError(f"ring_radius must have shape (R,) or (B,R), got {tuple(ring_radius.shape)}")


def _to_batch_volume(initial_volume, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(initial_volume, torch.Tensor):
        volume = initial_volume.to(device=device, dtype=torch.float32).flatten()
        if volume.numel() == 1:
            return volume.repeat(batch_size)
        if volume.numel() != batch_size:
            raise ValueError(f"initial_volume size {volume.numel()} does not match batch size {batch_size}")
        return volume
    return torch.full((batch_size,), float(initial_volume), dtype=torch.float32, device=device)


def _expand_initial_temperature(initial_temperature, batch_size: int, num_rings: int, device: torch.device) -> torch.Tensor | None:
    if initial_temperature is None:
        return None
    temperature = initial_temperature.to(device=device, dtype=torch.float32)
    if temperature.ndim == 1:
        return temperature.unsqueeze(0).repeat(batch_size, 1)
    if temperature.ndim == 2 and temperature.shape == (batch_size, num_rings):
        return temperature.clone()
    if temperature.ndim == 2 and temperature.shape[0] == 1 and temperature.shape[1] == num_rings:
        return temperature.repeat(batch_size, 1)
    raise ValueError(
        f"initial_temperature must have shape (R,) or (B,R); got {tuple(temperature.shape)} for batch {batch_size}"
    )


def _evaluate_voltage_batch(
    cfg,
    ring_radius: torch.Tensor,
    initial_volume,
    voltage_v,
    initial_temperature: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor | float | bool]:
    ring_radius_batch, _ = _to_batch_ring_radius(ring_radius)
    device = ring_radius_batch.device
    batch_size, num_rings = ring_radius_batch.shape
    initial_volume_batch = _to_batch_volume(initial_volume, batch_size, device)
    voltage_batch = torch.as_tensor(voltage_v, dtype=torch.float32, device=device).flatten()
    if voltage_batch.numel() == 1:
        voltage_batch = voltage_batch.repeat(batch_size)
    if voltage_batch.numel() != batch_size:
        raise ValueError(f"voltage_v size {voltage_batch.numel()} does not match batch size {batch_size}")

    dz = cfg.height / max(num_rings - 1, 1)
    axial_weight = _axial_weights(num_rings, dz, device=device).unsqueeze(0)
    clamped_radius = torch.clamp(ring_radius_batch, min=cfg.min_radius)
    area = math.pi * clamped_radius.pow(2)
    surface = 2.0 * math.pi * clamped_radius * axial_weight
    volume = torch.sum(area * axial_weight, dim=1)
    volume_change_ratio = torch.abs(volume - initial_volume_batch) / torch.clamp(initial_volume_batch, min=1.0e-12)
    feature_change_ratio = torch.as_tensor(
        [_feature_change_ratio(cfg, ring_radius_batch[idx]) for idx in range(batch_size)],
        dtype=torch.float32,
        device=device,
    )
    roughness = torch.std(ring_radius_batch, dim=1) / torch.clamp(torch.mean(ring_radius_batch, dim=1), min=1.0e-12)
    feasibility_reports = [evaluate_feasibility(cfg, ring_radius_batch[idx]) for idx in range(batch_size)]
    feasibility_penalty = torch.as_tensor(
        [report.soft_penalty for report in feasibility_reports],
        dtype=torch.float32,
        device=device,
    )
    min_neck_diameter_mm = torch.as_tensor(
        [report.min_diameter_m * 1.0e3 for report in feasibility_reports],
        dtype=torch.float32,
        device=device,
    )
    max_radius_slope = torch.as_tensor(
        [report.max_slope for report in feasibility_reports],
        dtype=torch.float32,
        device=device,
    )

    if num_rings > 1:
        grad = torch.gradient(ring_radius_batch, spacing=dz, dim=1)[0] / max(cfg.radius, 1.0e-12)
    else:
        grad = torch.zeros_like(ring_radius_batch)
    global_shadow = torch.clamp(
        1.0 - float(cfg.shadow_roughness_coeff) * roughness,
        min=float(cfg.min_view_factor),
        max=1.0,
    )
    view_factor = torch.clamp(
        global_shadow.unsqueeze(1) - float(cfg.shadow_slope_coeff) * torch.abs(grad),
        min=float(cfg.min_view_factor),
        max=1.0,
    )

    expanded_temperature = _expand_initial_temperature(initial_temperature, batch_size, num_rings, device)
    if expanded_temperature is None:
        temperature = torch.full((batch_size, num_rings), float(cfg.ambient_temp), dtype=torch.float32, device=device)
    else:
        temperature = torch.clamp(expanded_temperature, min=float(cfg.ambient_temp)).clone()
    current = torch.zeros(batch_size, dtype=torch.float32, device=device)
    total_resistance = torch.full((batch_size,), float(cfg.external_series_resistance), dtype=torch.float32, device=device)
    thermal_residual = torch.full((batch_size,), float("inf"), dtype=torch.float32, device=device)
    thermal_iterations = torch.zeros(batch_size, dtype=torch.int64, device=device)
    thermal_converged = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for iter_idx in range(cfg.thermal_max_iters):
        cp, k, rho_elec = _material_properties(cfg, temperature)
        segment_resistance = rho_elec * axial_weight / torch.clamp(area, min=1.0e-12)
        total_resistance = torch.sum(segment_resistance, dim=1) + float(cfg.external_series_resistance)
        current = torch.minimum(voltage_batch / torch.clamp(total_resistance, min=float(cfg.min_resistance)), torch.full_like(total_resistance, float(cfg.max_current)))

        joule_power = current.unsqueeze(1).pow(2) * segment_resistance
        alpha = k / (float(cfg.density) * cp)
        evap_flux = _evaporation_flux_kg_m2_s(cfg, temperature)
        band_fraction = _band_fraction_tensor(cfg, temperature, cfg.in_band_upper_um)
        effective_total_emissivity = (
            float(cfg.band_emissivity) * band_fraction + float(cfg.out_of_band_emissivity) * (1.0 - band_fraction)
        )
        radiative_power = (
            effective_total_emissivity
            * float(cfg.stefan_boltzmann)
            * float(cfg.radiative_cooling_scale)
            * view_factor
            * surface
            * torch.clamp(temperature.pow(4) - float(cfg.ambient_temp) ** 4, min=0.0)
        )
        convective_power = float(cfg.convective_cooling_coeff) * surface * torch.clamp(temperature - float(cfg.ambient_temp), min=0.0)
        evaporative_power = evap_flux * surface * float(cfg.latent_heat_evap)
        thermal_mass = float(cfg.density) * area * axial_weight * cp
        dtemp = float(cfg.thermal_pseudo_dt) * (
            joule_power / torch.clamp(thermal_mass, min=1.0e-12)
            + alpha * _laplacian_1d(temperature) / max(dz * dz, 1.0e-12)
            - (radiative_power + convective_power + evaporative_power) / torch.clamp(thermal_mass, min=1.0e-12)
        )
        updated = torch.clamp(
            temperature + float(cfg.thermal_relaxation) * dtemp,
            min=float(cfg.ambient_temp),
            max=float(cfg.max_temp) * 1.25,
        )
        updated = _apply_thermal_boundary_mode(cfg, updated)
        residual = torch.max(torch.abs(updated - temperature), dim=1).values
        newly_converged = residual < float(cfg.thermal_tol_k)
        thermal_iterations = torch.where(
            thermal_converged,
            thermal_iterations,
            torch.full_like(thermal_iterations, iter_idx + 1),
        )
        thermal_residual = torch.where(thermal_converged, thermal_residual, residual)
        thermal_converged = thermal_converged | newly_converged
        temperature = updated
        if bool(torch.all(thermal_converged).item()):
            break

    temperature = _apply_thermal_boundary_mode(cfg, temperature)
    cp, k, rho_elec = _material_properties(cfg, temperature)
    segment_resistance = rho_elec * axial_weight / torch.clamp(area, min=1.0e-12)
    total_resistance = torch.sum(segment_resistance, dim=1) + float(cfg.external_series_resistance)
    current = torch.minimum(voltage_batch / torch.clamp(total_resistance, min=float(cfg.min_resistance)), torch.full_like(total_resistance, float(cfg.max_current)))
    total_power = voltage_batch * current

    band_fraction = _band_fraction_tensor(cfg, temperature, cfg.in_band_upper_um)
    net_band_power = torch.sum(
        float(cfg.band_emissivity)
        * float(cfg.stefan_boltzmann)
        * float(cfg.radiative_cooling_scale)
        * view_factor
        * surface
        * torch.clamp(temperature.pow(4) - float(cfg.ambient_temp) ** 4, min=0.0)
        * band_fraction,
        dim=1,
    )

    evap_flux = _evaporation_flux_kg_m2_s(cfg, temperature)
    mass_loss_rate = torch.sum(evap_flux * surface, dim=1)
    recession_rate = evap_flux / max(float(cfg.density), 1.0e-12)
    lifetime_budget = float(cfg.feature_fail_ratio) * clamped_radius
    lifetime_s = torch.min(lifetime_budget / torch.clamp(recession_rate, min=1.0e-12), dim=1).values
    uniformity = 1.0 / (
        1.0 + torch.std(temperature, dim=1) / torch.clamp(torch.mean(temperature, dim=1) - float(cfg.ambient_temp), min=1.0)
    )
    lifecycle_factor = torch.clamp(lifetime_s / (lifetime_s + float(cfg.lifecycle_reference_s)), min=0.0, max=1.0)
    smoothness_factor = torch.clamp(1.0 - 0.20 * roughness - 0.30 * feature_change_ratio, min=0.35, max=1.0)
    average_band_power = net_band_power * lifecycle_factor * smoothness_factor
    thermo_reports = [evaluate_thermo_mech(cfg, temperature[idx]) for idx in range(batch_size)]
    thermo_mech_penalty = torch.as_tensor(
        [report.soft_penalty for report in thermo_reports],
        dtype=torch.float32,
        device=device,
    )
    max_axial_stress_pa = torch.as_tensor(
        [report.max_axial_stress_pa for report in thermo_reports],
        dtype=torch.float32,
        device=device,
    )

    max_temperature = torch.max(temperature, dim=1).values
    feasible = (
        (max_temperature <= float(cfg.max_temp))
        & (feature_change_ratio < float(cfg.feature_fail_ratio))
        & (volume_change_ratio <= float(cfg.volume_tolerance_ratio))
        & (feasibility_penalty <= 1.0e-12)
    )
    rated_utility = (
        float(cfg.rated_weight_initial_power) * net_band_power
        + float(cfg.rated_weight_average_power) * average_band_power
        + float(cfg.rated_weight_uniformity) * uniformity
        - float(cfg.rated_penalty_mass_loss) * torch.clamp(mass_loss_rate - float(cfg.max_mass_loss_rate), min=0.0)
        - float(cfg.rated_penalty_temp_violation) * torch.clamp(max_temperature - float(cfg.max_temp), min=0.0).pow(2)
        - float(cfg.rated_penalty_feature_violation) * torch.clamp(feature_change_ratio - float(cfg.feature_fail_ratio), min=0.0)
        - float(cfg.rated_penalty_volume_change) * volume_change_ratio
        - float(getattr(cfg, "rated_penalty_feasibility", 25.0)) * feasibility_penalty
    )
    rated_utility = torch.where(feasible, rated_utility, rated_utility - 1.0e3)

    return {
        "voltage_v": voltage_batch,
        "current_a": current,
        "resistance_ohm": total_resistance,
        "mean_temperature_k": torch.mean(temperature, dim=1),
        "max_temperature_k": max_temperature,
        "initial_net_band_power_w": net_band_power,
        "average_net_band_power_w": average_band_power,
        "band_efficiency": net_band_power / torch.clamp(total_power, min=1.0e-12),
        "lifetime_s": lifetime_s,
        "mass_loss_rate_kg_s": mass_loss_rate,
        "view_factor_proxy": torch.mean(view_factor, dim=1),
        "temperature_uniformity": uniformity,
        "min_equivalent_diameter_mm": 2.0 * torch.min(ring_radius_batch, dim=1).values * 1.0e3,
        "feature_change_ratio": feature_change_ratio,
        "volume_change_ratio": volume_change_ratio,
        "smoothness_penalty": roughness,
        "feasibility_penalty": feasibility_penalty,
        "thermo_mech_penalty": thermo_mech_penalty,
        "min_neck_diameter_mm": min_neck_diameter_mm,
        "max_radius_slope": max_radius_slope,
        "max_axial_stress_pa": max_axial_stress_pa,
        "feasible": feasible,
        "rated_utility": rated_utility,
        "thermal_iterations": thermal_iterations,
        "thermal_residual_k": thermal_residual,
        "thermal_converged": thermal_converged,
        "ring_temperature_k": temperature,
        "ring_recession_rate_m_s": recession_rate,
    }


def _evaluate_voltage(
    cfg,
    ring_radius: torch.Tensor,
    initial_volume: float,
    voltage_v: float,
    initial_temperature: torch.Tensor | None = None,
) -> RatedConditionMetrics:
    batch = _evaluate_voltage_batch(
        cfg=cfg,
        ring_radius=ring_radius,
        initial_volume=initial_volume,
        voltage_v=voltage_v,
        initial_temperature=initial_temperature,
    )
    return RatedConditionMetrics(
        voltage_v=float(batch["voltage_v"][0].item()),
        current_a=float(batch["current_a"][0].item()),
        resistance_ohm=float(batch["resistance_ohm"][0].item()),
        mean_temperature_k=float(batch["mean_temperature_k"][0].item()),
        max_temperature_k=float(batch["max_temperature_k"][0].item()),
        initial_net_band_power_w=float(batch["initial_net_band_power_w"][0].item()),
        average_net_band_power_w=float(batch["average_net_band_power_w"][0].item()),
        band_efficiency=float(batch["band_efficiency"][0].item()),
        lifetime_s=float(batch["lifetime_s"][0].item()),
        mass_loss_rate_kg_s=float(batch["mass_loss_rate_kg_s"][0].item()),
        view_factor_proxy=float(batch["view_factor_proxy"][0].item()),
        temperature_uniformity=float(batch["temperature_uniformity"][0].item()),
        min_equivalent_diameter_mm=float(batch["min_equivalent_diameter_mm"][0].item()),
        feature_change_ratio=float(batch["feature_change_ratio"][0].item()),
        volume_change_ratio=float(batch["volume_change_ratio"][0].item()),
        smoothness_penalty=float(batch["smoothness_penalty"][0].item()),
        feasibility_penalty=float(batch["feasibility_penalty"][0].item()),
        thermo_mech_penalty=float(batch["thermo_mech_penalty"][0].item()),
        min_neck_diameter_mm=float(batch["min_neck_diameter_mm"][0].item()),
        max_radius_slope=float(batch["max_radius_slope"][0].item()),
        max_axial_stress_pa=float(batch["max_axial_stress_pa"][0].item()),
        feasible=bool(batch["feasible"][0].item()),
        rated_utility=float(batch["rated_utility"][0].item()),
        thermal_iterations=int(batch["thermal_iterations"][0].item()),
        thermal_residual_k=float(batch["thermal_residual_k"][0].item()),
        thermal_converged=bool(batch["thermal_converged"][0].item()),
        ring_temperature_k=batch["ring_temperature_k"][0].clone(),
        ring_recession_rate_m_s=batch["ring_recession_rate_m_s"][0].clone(),
    )


def _voltage_grid(min_voltage: float, max_voltage: float, num_points: int) -> np.ndarray:
    if int(num_points) <= 1:
        return np.asarray([float(max_voltage)], dtype=np.float64)
    return np.linspace(float(min_voltage), float(max_voltage), int(num_points), dtype=np.float64)


def search_rated_condition(cfg, ring_radius: torch.Tensor, initial_volume: float) -> RatedConditionMetrics:
    evaluated: Dict[float, RatedConditionMetrics] = {}

    def evaluate(voltage_v: float, initial_temperature: torch.Tensor | None = None) -> RatedConditionMetrics:
        key = round(float(voltage_v), 6)
        if key not in evaluated:
            evaluated[key] = _evaluate_voltage(
                cfg,
                ring_radius,
                initial_volume,
                float(voltage_v),
                initial_temperature=initial_temperature,
            )
        return evaluated[key]

    best: RatedConditionMetrics | None = None
    coarse_grid = _voltage_grid(cfg.min_voltage, cfg.max_voltage, cfg.voltage_grid_points)
    warm_temperature = None
    for voltage_v in coarse_grid:
        metrics = evaluate(float(voltage_v), initial_temperature=warm_temperature)
        warm_temperature = metrics.ring_temperature_k.detach()
        if best is None or metrics.rated_utility > best.rated_utility:
            best = metrics

    if best is None:
        raise RuntimeError("Rated-condition search failed to evaluate any voltage.")

    span = max(
        (float(cfg.max_voltage) - float(cfg.min_voltage)) * float(cfg.voltage_focus_ratio),
        float(cfg.max_voltage - cfg.min_voltage) / max(float(cfg.voltage_grid_points - 1), 1.0),
    )
    for _ in range(int(cfg.voltage_refine_levels)):
        lo = max(float(cfg.min_voltage), best.voltage_v - span)
        hi = min(float(cfg.max_voltage), best.voltage_v + span)
        warm_temperature = best.ring_temperature_k.detach()
        local_grid = sorted(_voltage_grid(lo, hi, cfg.voltage_refine_points), key=lambda v: abs(float(v) - best.voltage_v))
        for voltage_v in local_grid:
            metrics = evaluate(float(voltage_v), initial_temperature=warm_temperature)
            warm_temperature = metrics.ring_temperature_k.detach()
            if metrics.rated_utility > best.rated_utility:
                best = metrics
        span *= 0.35

    if best is None:
        raise RuntimeError("Rated-condition search failed to evaluate any voltage.")
    return best


def search_rated_condition_batch(cfg, ring_radius: torch.Tensor, initial_volume) -> Dict[str, torch.Tensor]:
    ring_radius_batch, squeezed = _to_batch_ring_radius(ring_radius)
    batch_size = int(ring_radius_batch.shape[0])
    evaluated: Dict[float, Dict[str, torch.Tensor | float | bool]] = {}

    def evaluate(voltage_v: float, initial_temperature: torch.Tensor | None = None) -> Dict[str, torch.Tensor | float | bool]:
        key = round(float(voltage_v), 6)
        if key not in evaluated:
            evaluated[key] = _evaluate_voltage_batch(
                cfg=cfg,
                ring_radius=ring_radius_batch,
                initial_volume=initial_volume,
                voltage_v=torch.full((batch_size,), float(voltage_v), dtype=torch.float32, device=ring_radius_batch.device),
                initial_temperature=initial_temperature,
            )
        return evaluated[key]

    coarse_grid = _voltage_grid(cfg.min_voltage, cfg.max_voltage, cfg.voltage_grid_points)
    warm_temperature = None
    best_metrics = None
    best_utility = None
    for voltage_v in coarse_grid:
        metrics = evaluate(float(voltage_v), initial_temperature=warm_temperature)
        warm_temperature = metrics["ring_temperature_k"].detach()
        utility = metrics["rated_utility"]
        if best_metrics is None:
            best_metrics = metrics
            best_utility = utility
            continue
        better = utility > best_utility
        if bool(torch.any(better).item()):
            best_metrics = {
                key: (
                    torch.where(better.unsqueeze(1), value, best_metrics[key])
                    if isinstance(value, torch.Tensor) and value.ndim == 2
                    else torch.where(better, value, best_metrics[key])
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in metrics.items()
            }
            best_utility = torch.where(better, utility, best_utility)

    if best_metrics is None or best_utility is None:
        raise RuntimeError("Rated-condition batch search failed to evaluate any voltage.")

    span = max(
        (float(cfg.max_voltage) - float(cfg.min_voltage)) * float(cfg.voltage_focus_ratio),
        float(cfg.max_voltage - cfg.min_voltage) / max(float(cfg.voltage_grid_points - 1), 1.0),
    )
    for _ in range(int(cfg.voltage_refine_levels)):
        best_voltage = best_metrics["voltage_v"]
        warm_temperature = best_metrics["ring_temperature_k"].detach()
        trial_voltages = []
        for idx in range(batch_size):
            lo = max(float(cfg.min_voltage), float(best_voltage[idx].item()) - span)
            hi = min(float(cfg.max_voltage), float(best_voltage[idx].item()) + span)
            trial_voltages.append(_voltage_grid(lo, hi, cfg.voltage_refine_points))
        candidate_stack = np.stack(trial_voltages, axis=0)
        for local_idx in range(candidate_stack.shape[1]):
            local_voltage = torch.as_tensor(candidate_stack[:, local_idx], dtype=torch.float32, device=ring_radius_batch.device)
            metrics = _evaluate_voltage_batch(
                cfg=cfg,
                ring_radius=ring_radius_batch,
                initial_volume=initial_volume,
                voltage_v=local_voltage,
                initial_temperature=warm_temperature,
            )
            warm_temperature = metrics["ring_temperature_k"].detach()
            utility = metrics["rated_utility"]
            better = utility > best_utility
            if bool(torch.any(better).item()):
                best_metrics = {
                    key: (
                        torch.where(better.unsqueeze(1), value, best_metrics[key])
                        if isinstance(value, torch.Tensor) and value.ndim == 2
                        else torch.where(better, value, best_metrics[key])
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in metrics.items()
                }
                best_utility = torch.where(better, utility, best_utility)
        span *= 0.35

    if squeezed:
        return {key: value[0] if isinstance(value, torch.Tensor) else value for key, value in best_metrics.items()}
    return best_metrics


def _prepare_voltage_schedule(
    voltage_schedule,
    batch_size: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    schedule = torch.as_tensor(voltage_schedule, dtype=torch.float32, device=device)
    if schedule.ndim == 0:
        return schedule.repeat(batch_size, num_steps)
    if schedule.ndim == 1:
        if schedule.numel() == num_steps:
            return schedule.unsqueeze(0).repeat(batch_size, 1)
        if schedule.numel() == batch_size:
            return schedule.unsqueeze(1).repeat(1, num_steps)
    if schedule.ndim == 2 and schedule.shape == (batch_size, num_steps):
        return schedule
    raise ValueError(
        f"voltage_schedule must be scalar, (steps,), (batch,), or (batch, steps); got {tuple(schedule.shape)}"
    )


def simulate_transient_trajectory(
    cfg,
    ring_radius: torch.Tensor,
    voltage_schedule,
    t_max: float | None = None,
    dt: float | None = None,
    initial_temperature: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    ring_radius_batch, squeezed = _to_batch_ring_radius(ring_radius)
    device = ring_radius_batch.device
    batch_size, num_rings = ring_radius_batch.shape
    dt_s = float(dt if dt is not None else cfg.transient_dt_s)
    total_time = float(t_max if t_max is not None else cfg.transient_max_time_s)
    num_steps = max(int(math.ceil(total_time / max(dt_s, 1.0e-6))), 1)
    schedule = _prepare_voltage_schedule(voltage_schedule, batch_size, num_steps, device)

    dz = cfg.height / max(num_rings - 1, 1)
    axial_weight = _axial_weights(num_rings, dz, device=device).unsqueeze(0)
    clamped_radius = torch.clamp(ring_radius_batch, min=cfg.min_radius)
    area = math.pi * clamped_radius.pow(2)
    surface = 2.0 * math.pi * clamped_radius * axial_weight
    if initial_temperature is None:
        temperature = torch.full((batch_size, num_rings), float(cfg.ambient_temp), dtype=torch.float32, device=device)
    else:
        temperature = _expand_initial_temperature(initial_temperature, batch_size, num_rings, device)
        temperature = torch.clamp(temperature, min=float(cfg.ambient_temp))
    temperature = _apply_thermal_boundary_mode(cfg, temperature)

    temperature_history = [temperature.clone()]
    band_power_history = []
    total_power_history = []
    mass_loss_history = [torch.zeros(batch_size, dtype=torch.float32, device=device)]
    current_history = []

    if num_rings > 1:
        grad = torch.gradient(ring_radius_batch, spacing=dz, dim=1)[0] / max(cfg.radius, 1.0e-12)
    else:
        grad = torch.zeros_like(ring_radius_batch)
    roughness = torch.std(ring_radius_batch, dim=1) / torch.clamp(torch.mean(ring_radius_batch, dim=1), min=1.0e-12)
    global_shadow = torch.clamp(
        1.0 - float(cfg.shadow_roughness_coeff) * roughness,
        min=float(cfg.min_view_factor),
        max=1.0,
    )
    view_factor = torch.clamp(
        global_shadow.unsqueeze(1) - float(cfg.shadow_slope_coeff) * torch.abs(grad),
        min=float(cfg.min_view_factor),
        max=1.0,
    )

    cumulative_mass = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for step_idx in range(num_steps):
        voltage = schedule[:, step_idx]
        cp, k, rho_elec = _material_properties(cfg, temperature)
        segment_resistance = rho_elec * axial_weight / torch.clamp(area, min=1.0e-12)
        total_resistance = torch.sum(segment_resistance, dim=1) + float(cfg.external_series_resistance)
        current = torch.minimum(voltage / torch.clamp(total_resistance, min=float(cfg.min_resistance)), torch.full_like(total_resistance, float(cfg.max_current)))
        joule_power = current.unsqueeze(1).pow(2) * segment_resistance
        alpha = k / (float(cfg.density) * cp)
        evap_flux = _evaporation_flux_kg_m2_s(cfg, temperature)
        band_fraction = _band_fraction_tensor(cfg, temperature, cfg.in_band_upper_um)
        effective_total_emissivity = (
            float(cfg.band_emissivity) * band_fraction + float(cfg.out_of_band_emissivity) * (1.0 - band_fraction)
        )
        radiative_power = (
            effective_total_emissivity
            * float(cfg.stefan_boltzmann)
            * float(cfg.radiative_cooling_scale)
            * view_factor
            * surface
            * torch.clamp(temperature.pow(4) - float(cfg.ambient_temp) ** 4, min=0.0)
        )
        convective_power = float(cfg.convective_cooling_coeff) * surface * torch.clamp(temperature - float(cfg.ambient_temp), min=0.0)
        evaporative_power = evap_flux * surface * float(cfg.latent_heat_evap)
        thermal_mass = float(cfg.density) * area * axial_weight * cp
        dtemp = dt_s * (
            joule_power / torch.clamp(thermal_mass, min=1.0e-12)
            + alpha * _laplacian_1d(temperature) / max(dz * dz, 1.0e-12)
            - (radiative_power + convective_power + evaporative_power) / torch.clamp(thermal_mass, min=1.0e-12)
        )
        temperature = torch.clamp(
            temperature + dtemp,
            min=float(cfg.ambient_temp),
            max=float(cfg.max_temp) * 1.25,
        )
        temperature = _apply_thermal_boundary_mode(cfg, temperature)

        band_power = torch.sum(
            float(cfg.band_emissivity)
            * float(cfg.stefan_boltzmann)
            * float(cfg.radiative_cooling_scale)
            * view_factor
            * surface
            * torch.clamp(temperature.pow(4) - float(cfg.ambient_temp) ** 4, min=0.0)
            * band_fraction,
            dim=1,
        )
        total_power = voltage * current
        cumulative_mass = cumulative_mass + torch.sum(evap_flux * surface, dim=1) * dt_s

        temperature_history.append(temperature.clone())
        band_power_history.append(band_power)
        total_power_history.append(total_power)
        mass_loss_history.append(cumulative_mass.clone())
        current_history.append(current)

    result = {
        "time_s": torch.linspace(0.0, num_steps * dt_s, num_steps + 1, dtype=torch.float32, device=device),
        "temperature_k": torch.stack(temperature_history, dim=1),
        "band_power_w": torch.stack(band_power_history, dim=1) if band_power_history else torch.zeros((batch_size, 0), dtype=torch.float32, device=device),
        "total_power_w": torch.stack(total_power_history, dim=1) if total_power_history else torch.zeros((batch_size, 0), dtype=torch.float32, device=device),
        "mass_loss_kg": torch.stack(mass_loss_history, dim=1),
        "current_a": torch.stack(current_history, dim=1) if current_history else torch.zeros((batch_size, 0), dtype=torch.float32, device=device),
        "dt_s": torch.tensor(dt_s, dtype=torch.float32, device=device),
    }
    if squeezed:
        return {
            key: value[0] if isinstance(value, torch.Tensor) and value.ndim > 1 and key != "time_s" else value
            for key, value in result.items()
        }
    return result
