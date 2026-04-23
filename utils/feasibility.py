from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import numpy as np


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


def project_connected_profile(
    ring_radius: torch.Tensor,
    min_radius: float,
    max_step_ratio: float = 3.0,
    fix_endpoints: bool = True,
) -> torch.Tensor:
    """
    Project ring_radius onto the space of physically connected profiles.

    The projection is a two-pass monotone slope-limited propagation:
      - Forward pass:  r[i] = min(r[i], r[i-1] * max_step_ratio)
      - Backward pass: r[i] = min(r[i], r[i+1] * max_step_ratio)

    This guarantees no adjacent radius ratio exceeds `max_step_ratio`, making
    it geometrically impossible for any cross-section to diverge far enough
    from its neighbours to form a disconnected floating volume.

    Parameters
    ----------
    ring_radius:     1-D tensor of ring radii.
    min_radius:      Hard lower bound applied after projection.
    max_step_ratio:  Maximum ratio r[i] / r[i-1] allowed between adjacent rings.
    fix_endpoints:   If True, keep the first and last ring unchanged (electrode rings).

    Returns
    -------
    Projected and clamped ring_radius tensor.
    """
    if ring_radius.numel() <= 1:
        return torch.clamp(ring_radius, min=float(min_radius))

    r = ring_radius.clone().float()
    r = torch.clamp(r, min=float(min_radius))

    start = 1 if fix_endpoints else 0
    end = r.numel() - 1 if fix_endpoints else r.numel()
    cap = float(max_step_ratio)

    # Forward pass: prevent a ring from being too large relative to its predecessor.
    for i in range(start, end):
        r[i] = torch.minimum(r[i], r[i - 1] * cap).clamp_min(float(min_radius))

    # Backward pass: prevent a ring from being too large relative to its successor.
    for i in range(end - 1, start - 1, -1):
        r[i] = torch.minimum(r[i], r[i + 1] * cap).clamp_min(float(min_radius))

    return r


def project_connected_profile_batch(
    ring_radius: torch.Tensor,
    min_radius: float,
    max_step_ratio: float = 3.0,
    fix_endpoints: bool = True,
) -> torch.Tensor:
    """Batched version of project_connected_profile. Shape: (B, R)."""
    if ring_radius.ndim == 1:
        return project_connected_profile(ring_radius, min_radius, max_step_ratio, fix_endpoints)
    if ring_radius.shape[1] <= 1:
        return torch.clamp(ring_radius, min=float(min_radius))
    r = ring_radius.clone().float()
    r = torch.clamp(r, min=float(min_radius))
    R = r.shape[1]
    start = 1 if fix_endpoints else 0
    end = R - 1 if fix_endpoints else R
    cap = float(max_step_ratio)
    for i in range(start, end):
        upper = r[:, i - 1] * cap
        r[:, i] = torch.minimum(r[:, i], upper).clamp_min(float(min_radius))
    for i in range(end - 1, start - 1, -1):
        upper = r[:, i + 1] * cap
        r[:, i] = torch.minimum(r[:, i], upper).clamp_min(float(min_radius))
    return r


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
