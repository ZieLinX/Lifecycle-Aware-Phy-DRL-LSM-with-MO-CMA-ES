from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    min_diameter_m: float
    max_slope: float
    worst_area_ratio: float
    neck_violation: float
    slope_violation: float
    area_ratio_violation: float
    soft_penalty: float


def evaluate_feasibility(cfg, ring_radius: torch.Tensor) -> FeasibilityReport:
    radius = torch.clamp(ring_radius, min=1.0e-9)
    diameter = 2.0 * radius
    min_diameter = float(torch.min(diameter).item())
    neck_violation = max(float(getattr(cfg, "min_neck_diameter_m", 2.0 * cfg.min_radius)) - min_diameter, 0.0)

    if radius.numel() > 1:
        dz = float(cfg.height) / max(int(radius.numel()) - 1, 1)
        slope = torch.abs(torch.diff(radius) / max(dz, 1.0e-12))
        max_slope = float(torch.max(slope).item())
        slope_violation = max(max_slope - float(getattr(cfg, "feasibility_max_slope", 0.35)), 0.0)
        area = math.pi * radius.pow(2)
        area_ratio = area[1:] / torch.clamp(area[:-1], min=1.0e-12)
        area_ratio_clamped = torch.maximum(area_ratio, 1.0 / torch.clamp(area_ratio, min=1.0e-12))
        worst_area_ratio = float(torch.max(area_ratio_clamped).item())
        area_ratio_violation = max(
            worst_area_ratio - float(getattr(cfg, "feasibility_area_ratio_max", 5.0)),
            0.0,
        )
        min_ratio = float(torch.min(area_ratio).item())
        area_ratio_violation = max(
            area_ratio_violation,
            max(float(getattr(cfg, "feasibility_area_ratio_min", 0.20)) - min_ratio, 0.0),
        )
    else:
        max_slope = 0.0
        slope_violation = 0.0
        worst_area_ratio = 1.0
        area_ratio_violation = 0.0

    soft_penalty = (
        neck_violation / max(float(getattr(cfg, "min_neck_diameter_m", 1.0e-3)), 1.0e-12)
        + slope_violation / max(float(getattr(cfg, "feasibility_max_slope", 0.35)), 1.0e-12)
        + area_ratio_violation / max(float(getattr(cfg, "feasibility_area_ratio_max", 5.0)), 1.0e-12)
    )
    feasible = soft_penalty <= 1.0e-12
    return FeasibilityReport(
        feasible=feasible,
        min_diameter_m=min_diameter,
        max_slope=max_slope,
        worst_area_ratio=worst_area_ratio,
        neck_violation=neck_violation,
        slope_violation=slope_violation,
        area_ratio_violation=area_ratio_violation,
        soft_penalty=float(soft_penalty),
    )
