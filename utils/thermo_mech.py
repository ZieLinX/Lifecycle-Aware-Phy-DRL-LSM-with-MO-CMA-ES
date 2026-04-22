from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThermoMechanicalReport:
    max_axial_stress_pa: float
    mean_axial_stress_pa: float
    stress_violation_ratio: float
    soft_penalty: float


def evaluate_thermo_mech(cfg, ring_temperature_k: torch.Tensor) -> ThermoMechanicalReport:
    if not bool(getattr(cfg, "enable_thermomech", True)):
        return ThermoMechanicalReport(
            max_axial_stress_pa=0.0,
            mean_axial_stress_pa=0.0,
            stress_violation_ratio=0.0,
            soft_penalty=0.0,
        )

    delta_t = torch.clamp(ring_temperature_k - float(getattr(cfg, "thermo_reference_temp_k", cfg.ambient_temp)), min=0.0)
    axial_stress = (
        float(getattr(cfg, "thermo_youngs_modulus_pa", 4.0e11))
        * float(getattr(cfg, "thermo_expansion_coeff_per_k", 4.5e-6))
        * delta_t
    )
    max_stress = float(torch.max(axial_stress).item())
    mean_stress = float(torch.mean(axial_stress).item())
    soft_yield = max(float(getattr(cfg, "thermo_soft_yield_pa", 2.0e8)), 1.0e-12)
    violation_ratio = max(max_stress / soft_yield - 1.0, 0.0)
    soft_penalty = float(torch.log1p(torch.tensor(violation_ratio * violation_ratio)).item())
    return ThermoMechanicalReport(
        max_axial_stress_pa=max_stress,
        mean_axial_stress_pa=mean_stress,
        stress_violation_ratio=violation_ratio,
        soft_penalty=float(soft_penalty),
    )
