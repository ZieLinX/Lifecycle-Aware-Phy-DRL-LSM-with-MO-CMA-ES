from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Dict

import numpy as np
import torch


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
    if values.numel() == 1:
        return lap
    lap[0] = values[1] - values[0]
    lap[-1] = values[-2] - values[-1]
    if values.numel() > 2:
        lap[1:-1] = values[:-2] - 2.0 * values[1:-1] + values[2:]
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


def _evaluate_voltage(
    cfg,
    ring_radius: torch.Tensor,
    initial_volume: float,
    voltage_v: float,
    initial_temperature: torch.Tensor | None = None,
) -> RatedConditionMetrics:
    device = ring_radius.device
    num_rings = int(ring_radius.shape[0])
    dz = cfg.height / max(num_rings - 1, 1)
    axial_weight = _axial_weights(num_rings, dz, device=device)
    area = math.pi * torch.clamp(ring_radius, min=cfg.min_radius).pow(2)
    surface = 2.0 * math.pi * torch.clamp(ring_radius, min=cfg.min_radius) * axial_weight
    volume = float(torch.sum(area * axial_weight).item())
    volume_change_ratio = abs(volume - initial_volume) / max(initial_volume, 1.0e-12)
    feature_change_ratio = float(torch.max(torch.abs(ring_radius - cfg.radius) / max(cfg.radius, 1.0e-12)).item())
    roughness = float(torch.std(ring_radius).item() / max(float(torch.mean(ring_radius).item()), 1.0e-12))

    if num_rings > 1:
        grad = torch.gradient(ring_radius, spacing=dz)[0] / max(cfg.radius, 1.0e-12)
    else:
        grad = torch.zeros_like(ring_radius)
    global_shadow = float(np.clip(1.0 - cfg.shadow_roughness_coeff * roughness, cfg.min_view_factor, 1.0))
    view_factor = torch.clamp(
        torch.full_like(ring_radius, global_shadow) - cfg.shadow_slope_coeff * torch.abs(grad),
        min=cfg.min_view_factor,
        max=1.0,
    )

    if initial_temperature is None:
        temperature = torch.full((num_rings,), float(cfg.ambient_temp), dtype=torch.float32, device=device)
    else:
        temperature = torch.clamp(initial_temperature.to(device=device, dtype=torch.float32), min=cfg.ambient_temp).clone()
    current = 0.0
    total_resistance = float(cfg.external_series_resistance)
    thermal_residual = float("inf")
    thermal_iterations = 0
    thermal_converged = False
    for iter_idx in range(cfg.thermal_max_iters):
        cp, k, rho_elec = _material_properties(cfg, temperature)
        segment_resistance = rho_elec * axial_weight / torch.clamp(area, min=1.0e-12)
        total_resistance = float(torch.sum(segment_resistance).item() + cfg.external_series_resistance)
        current = min(float(voltage_v) / max(total_resistance, cfg.min_resistance), cfg.max_current)

        joule_power = (current**2) * segment_resistance
        alpha = k / (cfg.density * cp)
        evap_flux = _evaporation_flux_kg_m2_s(cfg, temperature)
        band_fraction = _band_fraction_tensor(cfg, temperature, cfg.in_band_upper_um)
        effective_total_emissivity = (
            cfg.band_emissivity * band_fraction + cfg.out_of_band_emissivity * (1.0 - band_fraction)
        )
        radiative_power = (
            effective_total_emissivity
            * cfg.stefan_boltzmann
            * cfg.radiative_cooling_scale
            * view_factor
            * surface
            * torch.clamp(temperature.pow(4) - cfg.ambient_temp**4, min=0.0)
        )
        convective_power = cfg.convective_cooling_coeff * surface * torch.clamp(temperature - cfg.ambient_temp, min=0.0)
        evaporative_power = evap_flux * surface * cfg.latent_heat_evap
        thermal_mass = cfg.density * area * axial_weight * cp
        dtemp = cfg.thermal_pseudo_dt * (
            joule_power / torch.clamp(thermal_mass, min=1.0e-12)
            + alpha * _laplacian_1d(temperature) / max(dz * dz, 1.0e-12)
            - (radiative_power + convective_power + evaporative_power) / torch.clamp(thermal_mass, min=1.0e-12)
        )
        updated = torch.clamp(
            temperature + cfg.thermal_relaxation * dtemp,
            min=cfg.ambient_temp,
            max=cfg.max_temp * 1.25,
        )
        thermal_iterations = iter_idx + 1
        thermal_residual = float(torch.max(torch.abs(updated - temperature)).item())
        if thermal_residual < cfg.thermal_tol_k:
            temperature = updated
            thermal_converged = True
            break
        temperature = updated

    cp, k, rho_elec = _material_properties(cfg, temperature)
    segment_resistance = rho_elec * axial_weight / torch.clamp(area, min=1.0e-12)
    total_resistance = float(torch.sum(segment_resistance).item() + cfg.external_series_resistance)
    current = min(float(voltage_v) / max(total_resistance, cfg.min_resistance), cfg.max_current)
    total_power = float(voltage_v) * current

    band_fraction = _band_fraction_tensor(cfg, temperature, cfg.in_band_upper_um)
    net_band_power = torch.sum(
        cfg.band_emissivity
        * cfg.stefan_boltzmann
        * cfg.radiative_cooling_scale
        * view_factor
        * surface
        * torch.clamp(temperature.pow(4) - cfg.ambient_temp**4, min=0.0)
        * band_fraction
    )

    evap_flux = _evaporation_flux_kg_m2_s(cfg, temperature)
    mass_loss_rate = float(torch.sum(evap_flux * surface).item())
    recession_rate = evap_flux / max(cfg.density, 1.0e-12)
    lifetime_budget = cfg.feature_fail_ratio * torch.clamp(ring_radius, min=cfg.min_radius)
    lifetime_s = float(torch.min(lifetime_budget / torch.clamp(recession_rate, min=1.0e-12)).item())
    uniformity = 1.0 / (
        1.0 + float(torch.std(temperature).item()) / max(float(torch.mean(temperature).item() - cfg.ambient_temp), 1.0)
    )
    lifecycle_factor = float(np.clip(lifetime_s / (lifetime_s + cfg.lifecycle_reference_s), 0.0, 1.0))
    smoothness_factor = float(np.clip(1.0 - 0.20 * roughness - 0.30 * feature_change_ratio, 0.35, 1.0))
    average_band_power = float(net_band_power.item()) * lifecycle_factor * smoothness_factor

    max_temperature = float(torch.max(temperature).item())
    feasible = (
        max_temperature <= cfg.max_temp
        and feature_change_ratio < cfg.feature_fail_ratio
        and volume_change_ratio <= cfg.volume_tolerance_ratio
    )
    rated_utility = (
        cfg.rated_weight_initial_power * float(net_band_power.item())
        + cfg.rated_weight_average_power * average_band_power
        + cfg.rated_weight_uniformity * uniformity
        - cfg.rated_penalty_mass_loss * max(mass_loss_rate - cfg.max_mass_loss_rate, 0.0)
        - cfg.rated_penalty_temp_violation * max(max_temperature - cfg.max_temp, 0.0) ** 2
        - cfg.rated_penalty_feature_violation * max(feature_change_ratio - cfg.feature_fail_ratio, 0.0)
        - cfg.rated_penalty_volume_change * volume_change_ratio
    )
    if not feasible:
        rated_utility -= 1.0e3

    return RatedConditionMetrics(
        voltage_v=float(voltage_v),
        current_a=float(current),
        resistance_ohm=float(total_resistance),
        mean_temperature_k=float(torch.mean(temperature).item()),
        max_temperature_k=max_temperature,
        initial_net_band_power_w=float(net_band_power.item()),
        average_net_band_power_w=average_band_power,
        band_efficiency=float(net_band_power.item()) / max(total_power, 1.0e-12),
        lifetime_s=lifetime_s,
        mass_loss_rate_kg_s=mass_loss_rate,
        view_factor_proxy=float(torch.mean(view_factor).item()),
        temperature_uniformity=uniformity,
        min_equivalent_diameter_mm=float(2.0 * torch.min(ring_radius).item() * 1.0e3),
        feature_change_ratio=feature_change_ratio,
        volume_change_ratio=volume_change_ratio,
        smoothness_penalty=roughness,
        feasible=feasible,
        rated_utility=rated_utility,
        thermal_iterations=thermal_iterations,
        thermal_residual_k=thermal_residual,
        thermal_converged=thermal_converged,
        ring_temperature_k=temperature,
        ring_recession_rate_m_s=recession_rate,
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
