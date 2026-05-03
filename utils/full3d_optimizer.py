from __future__ import annotations

from dataclasses import dataclass
import csv
from functools import lru_cache
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
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from utils.exporter import _run_freecad_stl_to_step


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None)
    if callable(integrate):
        return float(integrate(y, x))
    integrate = getattr(np, "trapz", None)
    if callable(integrate):
        return float(integrate(y, x))
    if y.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


@lru_cache(maxsize=1024)
def _blackbody_band_fraction_cached(temp_k_rounded: float, upper_um: float) -> float:
    temp_k = max(float(temp_k_rounded), 100.0)
    wavelengths_um = np.linspace(0.1, 20.0, 2400)
    wavelengths_m = wavelengths_um * 1.0e-6
    c2 = 1.438776877e-2
    exponent = np.clip(c2 / (wavelengths_m * temp_k), 1.0e-9, 700.0)
    spectral_shape = 1.0 / (np.power(wavelengths_m, 5) * (np.exp(exponent) - 1.0))
    total = _trapezoid_integral(spectral_shape, wavelengths_um)
    band_mask = wavelengths_um <= upper_um
    in_band = _trapezoid_integral(spectral_shape[band_mask], wavelengths_um[band_mask])
    return in_band / max(total, 1.0e-12)


def blackbody_band_fraction(temp_k: float, upper_um: float) -> float:
    return _blackbody_band_fraction_cached(round(float(temp_k), 1), float(upper_um))


@dataclass(frozen=True)
class Full3DGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    side_indices: np.ndarray
    lower_cap_indices: np.ndarray
    upper_cap_indices: np.ndarray
    lower_electrode_indices: np.ndarray
    upper_electrode_indices: np.ndarray
    num_rings: int
    num_segments: int
    cap_rings: int


@dataclass(frozen=True)
class Full3DResult:
    best_geometry: Full3DGeometry
    baseline_geometry: Full3DGeometry
    best_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    history_geometries: List[Full3DGeometry]
    history_metrics: List[Dict[str, float]]
    candidate_count: int
    archive_metrics: List[Dict[str, float]]
    selection_diagnostics: Dict[str, object]


class Full3DUNetGNNPolicy(nn.Module):
    """Small 3D U-Net style encoder plus graph smoothing head for strategy fields.

    The optimizer uses the untrained network as a structured generator. It is
    deliberately lightweight so CPU smoke tests and RTX4090 runs share the same
    code path without requiring extra sparse-convolution dependencies.
    """

    def __init__(self, in_channels: int = 4, hidden_channels: int = 12, out_channels: int = 3):
        super().__init__()
        self.enc1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1)
        self.dec = nn.ConvTranspose3d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.out = nn.Conv3d(hidden_channels * 2, out_channels, kernel_size=1)
        self.gnn_mix = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        x1 = F.silu(self.enc1(volume))
        x2 = F.silu(self.enc2(x1))
        x2 = F.silu(self.mid(x2))
        up = self.dec(x2)
        if up.shape[-3:] != x1.shape[-3:]:
            up = F.interpolate(up, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        logits = self.out(torch.cat([x1, up], dim=1))
        smooth = (
            torch.roll(logits, 1, dims=2)
            + torch.roll(logits, -1, dims=2)
            + torch.roll(logits, 1, dims=3)
            + torch.roll(logits, -1, dims=3)
            + torch.roll(logits, 1, dims=4)
            + torch.roll(logits, -1, dims=4)
        ) / 6.0
        mix = torch.clamp(self.gnn_mix, 0.0, 1.0)
        return torch.sigmoid((1.0 - mix) * logits + mix * smooth)


def _idx_grid(rows: int, cols: int) -> np.ndarray:
    return np.full((int(rows), int(cols)), -1, dtype=np.int64)


def build_baseline_full3d_geometry(cfg) -> Full3DGeometry:
    num_rings = int(cfg.num_rings)
    num_segments = int(cfg.num_segments)
    cap_rings = int(getattr(cfg, "full3d_cap_rings", 8))
    radius = float(cfg.radius)
    height = float(cfg.height)
    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False, dtype=np.float64)
    side_idx = _idx_grid(num_rings, num_segments)
    lower_idx = _idx_grid(cap_rings + 1, num_segments)
    upper_idx = _idx_grid(cap_rings + 1, num_segments)
    vertices: list[list[float]] = []

    def add_vertex(x: float, y: float, z: float) -> int:
        vertices.append([float(x), float(y), float(z)])
        return len(vertices) - 1

    z_values = np.linspace(-0.5 * height, 0.5 * height, num_rings, dtype=np.float64)
    for ridx, z in enumerate(z_values):
        for sidx, angle in enumerate(theta):
            side_idx[ridx, sidx] = add_vertex(radius * math.cos(angle), radius * math.sin(angle), z)

    # cap ring 0 is the tungsten end-face outer boundary and reuses the side end ring.
    # It is not fixed to the 5 mm electrode disk; contact is computed as the
    # footprint overlap with the fixed electrode disk.
    lower_idx[0, :] = side_idx[0, :]
    upper_idx[0, :] = side_idx[-1, :]
    for k in range(1, cap_rings):
        ring_radius = radius * (1.0 - float(k) / float(cap_rings))
        for sidx, angle in enumerate(theta):
            lower_idx[k, sidx] = add_vertex(ring_radius * math.cos(angle), ring_radius * math.sin(angle), -0.5 * height)
            upper_idx[k, sidx] = add_vertex(ring_radius * math.cos(angle), ring_radius * math.sin(angle), 0.5 * height)
    lower_center = add_vertex(0.0, 0.0, -0.5 * height)
    upper_center = add_vertex(0.0, 0.0, 0.5 * height)
    lower_idx[cap_rings, :] = lower_center
    upper_idx[cap_rings, :] = upper_center

    faces: list[list[int]] = []
    for ridx in range(num_rings - 1):
        for sidx in range(num_segments):
            sn = (sidx + 1) % num_segments
            faces.append([int(side_idx[ridx, sidx]), int(side_idx[ridx + 1, sidx]), int(side_idx[ridx, sn])])
            faces.append([int(side_idx[ridx, sn]), int(side_idx[ridx + 1, sidx]), int(side_idx[ridx + 1, sn])])
    for grid, flip in ((lower_idx, True), (upper_idx, False)):
        for k in range(cap_rings):
            for sidx in range(num_segments):
                sn = (sidx + 1) % num_segments
                if k == cap_rings - 1:
                    tri = [int(grid[k, sidx]), int(grid[k + 1, sidx]), int(grid[k, sn])]
                    faces.append(tri[::-1] if flip else tri)
                else:
                    f1 = [int(grid[k, sidx]), int(grid[k + 1, sidx]), int(grid[k, sn])]
                    f2 = [int(grid[k, sn]), int(grid[k + 1, sidx]), int(grid[k + 1, sn])]
                    faces.append(f1[::-1] if flip else f1)
                    faces.append(f2[::-1] if flip else f2)

    return Full3DGeometry(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        side_indices=side_idx,
        lower_cap_indices=lower_idx,
        upper_cap_indices=upper_idx,
        lower_electrode_indices=side_idx[0].copy(),
        upper_electrode_indices=side_idx[-1].copy(),
        num_rings=num_rings,
        num_segments=num_segments,
        cap_rings=cap_rings,
    )


def _clone_geometry(geometry: Full3DGeometry, vertices: np.ndarray) -> Full3DGeometry:
    return Full3DGeometry(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=geometry.faces.copy(),
        side_indices=geometry.side_indices.copy(),
        lower_cap_indices=geometry.lower_cap_indices.copy(),
        upper_cap_indices=geometry.upper_cap_indices.copy(),
        lower_electrode_indices=geometry.lower_electrode_indices.copy(),
        upper_electrode_indices=geometry.upper_electrode_indices.copy(),
        num_rings=geometry.num_rings,
        num_segments=geometry.num_segments,
        cap_rings=geometry.cap_rings,
    )


def mesh_volume(geometry: Full3DGeometry) -> float:
    mesh = trimesh.Trimesh(vertices=geometry.vertices, faces=geometry.faces, process=False)
    return float(abs(mesh.volume))


def _mesh_area_normals_centers(geometry: Full3DGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = geometry.vertices
    tri = v[geometry.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(2.0 * area[:, None], 1.0e-12)
    centers = np.mean(tri, axis=1)
    mesh_center = np.mean(v, axis=0)
    outward = np.sum(normals * (centers - mesh_center), axis=1) < 0.0
    normals[outward] *= -1.0
    return area, normals, centers


def _free_vertex_mask(geometry: Full3DGeometry) -> np.ndarray:
    return np.ones(geometry.vertices.shape[0], dtype=bool)


def _electrode_error(geometry: Full3DGeometry, cfg) -> float:
    height = float(cfg.height)
    ids = np.unique(
        np.concatenate(
            [
                geometry.side_indices[0].reshape(-1),
                geometry.side_indices[-1].reshape(-1),
                geometry.lower_cap_indices.reshape(-1),
                geometry.upper_cap_indices.reshape(-1),
            ]
        )
    )
    ids = ids[ids >= 0]
    points = geometry.vertices[ids]
    lower = points[:, 2] < 0.0
    lower_error = np.max(np.abs(points[lower, 2] + 0.5 * height)) if np.any(lower) else 0.0
    upper_error = np.max(np.abs(points[~lower, 2] - 0.5 * height)) if np.any(~lower) else 0.0
    return float(max(lower_error, upper_error))


def _enforce_axial_length(cfg, geometry: Full3DGeometry, vertices: np.ndarray) -> None:
    height = float(cfg.height)
    z_values = np.linspace(-0.5 * height, 0.5 * height, int(geometry.num_rings), dtype=np.float64)
    for ridx, z in enumerate(z_values):
        ids = geometry.side_indices[ridx]
        ids = ids[ids >= 0]
        vertices[ids, 2] = float(z)
    lower_ids = geometry.lower_cap_indices.reshape(-1)
    lower_ids = lower_ids[lower_ids >= 0]
    upper_ids = geometry.upper_cap_indices.reshape(-1)
    upper_ids = upper_ids[upper_ids >= 0]
    vertices[lower_ids, 2] = -0.5 * height
    vertices[upper_ids, 2] = 0.5 * height


def _end_contact_metrics(geometry: Full3DGeometry, cfg) -> Dict[str, float]:
    electrode_radius = float(cfg.radius)
    electrode_area = math.pi * electrode_radius * electrode_radius

    def one_end(ids: np.ndarray) -> tuple[float, float, float, float]:
        points = geometry.vertices[ids]
        footprint_area = _polygon_area_xy(points[:, :2])
        radial = np.linalg.norm(points[:, :2], axis=1)
        scale = np.minimum(1.0, electrode_radius / np.maximum(radial, 1.0e-12))
        clipped = points[:, :2] * scale[:, None]
        contact_area = min(_polygon_area_xy(clipped), footprint_area, electrode_area)
        noncontact = max(footprint_area - contact_area, 0.0)
        missing = max(electrode_area - contact_area, 0.0)
        return float(footprint_area), float(contact_area), float(noncontact), float(missing)

    lower_footprint, lower_contact, lower_noncontact, lower_missing = one_end(geometry.lower_electrode_indices)
    upper_footprint, upper_contact, upper_noncontact, upper_missing = one_end(geometry.upper_electrode_indices)
    total_footprint = lower_footprint + upper_footprint
    total_contact = lower_contact + upper_contact
    return {
        "electrode_disk_area_m2": float(electrode_area),
        "lower_end_face_area_m2": lower_footprint,
        "upper_end_face_area_m2": upper_footprint,
        "end_face_area_m2": float(total_footprint),
        "lower_electrode_contact_area_m2": lower_contact,
        "upper_electrode_contact_area_m2": upper_contact,
        "electrode_contact_area_m2": float(total_contact),
        "lower_noncontact_end_face_area_m2": lower_noncontact,
        "upper_noncontact_end_face_area_m2": upper_noncontact,
        "noncontact_end_face_area_m2": float(lower_noncontact + upper_noncontact),
        "lower_missing_electrode_area_m2": lower_missing,
        "upper_missing_electrode_area_m2": upper_missing,
        "missing_electrode_contact_area_m2": float(lower_missing + upper_missing),
        "lower_contact_fraction_of_tungsten_end": float(lower_contact / max(lower_footprint, 1.0e-18)),
        "upper_contact_fraction_of_tungsten_end": float(upper_contact / max(upper_footprint, 1.0e-18)),
    }


def project_full3d_geometry(cfg, geometry: Full3DGeometry, target_volume: float) -> Full3DGeometry:
    vertices = geometry.vertices.copy()
    free = _free_vertex_mask(geometry)
    _enforce_axial_length(cfg, geometry, vertices)
    for _ in range(6):
        trial = _clone_geometry(geometry, vertices)
        current_volume = mesh_volume(trial)
        scale = math.sqrt(float(target_volume) / max(current_volume, 1.0e-18))
        vertices[free, :2] *= scale
        _enforce_axial_length(cfg, geometry, vertices)
    return _clone_geometry(geometry, vertices)


def _geometry_to_policy_volume(cfg, geometry: Full3DGeometry, device: torch.device) -> torch.Tensor:
    side = geometry.vertices[geometry.side_indices]
    side_radius = np.linalg.norm(side[:, :, :2], axis=2) / max(float(cfg.radius), 1.0e-12)
    side_z = side[:, :, 2] / max(0.5 * float(cfg.height), 1.0e-12)
    lower = geometry.vertices[geometry.lower_cap_indices[:-1]]
    upper = geometry.vertices[geometry.upper_cap_indices[:-1]]
    cap_lower_z = lower[:, :, 2] / max(0.5 * float(cfg.height), 1.0e-12)
    cap_upper_z = upper[:, :, 2] / max(0.5 * float(cfg.height), 1.0e-12)
    cap_lower_r = np.linalg.norm(lower[:, :, :2], axis=2) / max(float(cfg.radius), 1.0e-12)
    cap_upper_r = np.linalg.norm(upper[:, :, :2], axis=2) / max(float(cfg.radius), 1.0e-12)
    rows = max(side_radius.shape[0], cap_lower_r.shape[0] * 2)
    cols = side_radius.shape[1]
    depth = 8
    vol = np.zeros((4, depth, rows, cols), dtype=np.float32)
    resized = np.vstack([side_radius, cap_lower_r, cap_upper_r])
    resized_z = np.vstack([side_z, cap_lower_z, cap_upper_z])
    for cidx, arr in enumerate((resized, resized_z, np.abs(resized - 1.0), np.ones_like(resized))):
        tmp = torch.as_tensor(arr[None, None], dtype=torch.float32)
        tmp = F.interpolate(tmp, size=(rows, cols), mode="bilinear", align_corners=False).numpy()[0, 0]
        for didx in range(depth):
            vol[cidx, didx] = tmp
    return torch.as_tensor(vol[None], dtype=torch.float32, device=device)


def _apply_strategy_displacement(cfg, geometry: Full3DGeometry, strategy: torch.Tensor, amplitude: float) -> Full3DGeometry:
    vertices = geometry.vertices.copy()
    baseline = build_baseline_full3d_geometry(cfg)
    free = _free_vertex_mask(geometry)
    area, normals, centers = _mesh_area_normals_centers(geometry)
    vertex_speed = np.zeros(vertices.shape[0], dtype=np.float64)
    vertex_weight = np.zeros(vertices.shape[0], dtype=np.float64)
    strat = strategy.detach().cpu().numpy()[0]
    drive = float(np.mean(strat[0]) - 0.70 * np.mean(strat[1]) + 0.35 * np.mean(strat[2]))
    local = np.sin(np.linspace(0.0, 2.0 * math.pi, vertices.shape[0], endpoint=False))
    face_drive = drive + 0.25 * local[geometry.faces].mean(axis=1)
    for fidx, face in enumerate(geometry.faces):
        for vid in face:
            vertex_speed[int(vid)] += face_drive[fidx] * area[fidx]
            vertex_weight[int(vid)] += area[fidx]
    vertex_speed = vertex_speed / np.maximum(vertex_weight, 1.0e-12)
    # Convert face-normal tendencies to a vertex displacement direction using the
    # radial vector. This keeps the mesh connected and avoids arbitrary shearing.
    dirs = vertices.copy()
    dirs[:, 2] *= 0.35
    norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norm, 1.0e-12)
    vertices[free] += float(amplitude) * vertex_speed[free, None] * dirs[free]
    cap_amp = float(getattr(cfg, "full3d_cap_max_displacement_m", 4.0e-4))
    for grid, sign in ((geometry.lower_cap_indices[1:], -1.0), (geometry.upper_cap_indices[1:], 1.0)):
        ids = np.unique(grid.reshape(-1))
        ids = ids[ids >= 0]
        ids = ids[free[ids]]
        base_z = baseline.vertices[ids, 2]
        vertices[ids, 2] = np.clip(vertices[ids, 2], base_z - cap_amp, base_z + cap_amp)
        vertices[ids, 2] += sign * 0.25 * float(amplitude) * vertex_speed[ids]
    return _clone_geometry(geometry, vertices)


def _full3d_action_layout(cfg) -> Dict[str, int]:
    axial_modes = max(int(getattr(cfg, "full3d_action_axial_modes", 5)), 1)
    circum_modes = max(int(getattr(cfg, "full3d_action_circum_modes", 3)), 0)
    cap_modes = max(int(getattr(cfg, "full3d_action_cap_radial_modes", 4)), 1)
    strategy_channels = max(int(getattr(cfg, "full3d_action_strategy_channels", 4)), 1)
    circum_terms = 1 + 2 * circum_modes
    side_channel_dim = axial_modes * circum_terms
    side_dim = strategy_channels * side_channel_dim
    cap_dim = 2 * cap_modes * circum_terms
    return {
        "axial_modes": axial_modes,
        "circum_modes": circum_modes,
        "cap_modes": cap_modes,
        "strategy_channels": strategy_channels,
        "circum_terms": circum_terms,
        "side_channel_dim": side_channel_dim,
        "side_dim": side_dim,
        "cap_dim": cap_dim,
        "action_dim": side_dim + cap_dim,
    }


def full3d_action_dim(cfg) -> int:
    """Number of global strategy-field topology coefficients used by full3d CEM."""

    return int(_full3d_action_layout(cfg)["action_dim"])


def _chebyshev_basis(values: np.ndarray, modes: int) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
    modes = max(int(modes), 1)
    basis = np.ones((values.size, modes), dtype=np.float64)
    if modes > 1:
        basis[:, 1] = values
    for midx in range(2, modes):
        basis[:, midx] = 2.0 * values * basis[:, midx - 1] - basis[:, midx - 2]
    return basis


def _circum_basis(num_segments: int, circum_modes: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, int(num_segments), endpoint=False, dtype=np.float64)
    cols = [np.ones_like(theta)]
    for mode in range(1, int(circum_modes) + 1):
        cols.append(np.cos(float(mode) * theta))
        cols.append(np.sin(float(mode) * theta))
    return np.stack(cols, axis=1)


def _basis_field(primary_basis: np.ndarray, coeff: np.ndarray, circum_basis: np.ndarray) -> np.ndarray:
    field = primary_basis @ coeff @ circum_basis.T
    return np.tanh(field)


def _full3d_physics_strategy_fields(
    cfg,
    geometry: Full3DGeometry,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    layout = _full3d_action_layout(cfg)
    action_dim = int(layout["action_dim"])
    coeffs = np.zeros(action_dim, dtype=np.float64)
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    coeffs[: min(raw.size, action_dim)] = raw[:action_dim]
    coeffs = np.clip(coeffs, -3.0, 3.0)

    axial_modes = int(layout["axial_modes"])
    circum_modes = int(layout["circum_modes"])
    circum_terms = int(layout["circum_terms"])
    cap_modes = int(layout["cap_modes"])
    strategy_channels = int(layout["strategy_channels"])
    side_channel_dim = int(layout["side_channel_dim"])
    side_dim = int(layout["side_dim"])

    side_coeff = coeffs[:side_dim].reshape(strategy_channels, axial_modes, circum_terms)
    side = geometry.vertices[geometry.side_indices]
    z = np.mean(side[:, :, 2], axis=1)
    z_norm = 2.0 * (z - np.min(z)) / max(float(np.max(z) - np.min(z)), 1.0e-12) - 1.0
    axial_basis = _chebyshev_basis(z_norm, axial_modes)
    circum = _circum_basis(geometry.num_segments, circum_modes)
    axial_fade = 0.35 + 0.65 * np.sin(math.pi * np.clip((z_norm + 1.0) * 0.5, 0.0, 1.0))
    strategy = np.empty((strategy_channels, geometry.num_rings, geometry.num_segments), dtype=np.float64)
    for channel in range(strategy_channels):
        strategy[channel] = _basis_field(axial_basis, side_coeff[channel], circum) * axial_fade[:, None]
    cap_coeff = coeffs[side_dim:].reshape(2, cap_modes, circum_terms)
    return strategy, cap_coeff, coeffs[: side_dim].reshape(strategy_channels, side_channel_dim), layout


def _side_physics_sensitivities(cfg, geometry: Full3DGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = geometry.vertices[geometry.side_indices]
    radius = np.maximum(np.linalg.norm(side[:, :, :2], axis=2), 1.0e-9)
    z = np.mean(side[:, :, 2], axis=1)
    z_norm = (z - np.min(z)) / max(float(np.max(z) - np.min(z)), 1.0e-12)
    center_hot = np.sin(math.pi * np.clip(z_norm, 0.0, 1.0))[:, None]
    radius_norm = radius / max(float(cfg.radius), 1.0e-12)
    rad = center_hot * np.power(np.maximum(radius_norm, 0.05), 0.25)
    evap = np.square(center_hot) / np.maximum(radius_norm, 0.15)
    cur = center_hot / np.maximum(np.square(radius_norm), 0.08)

    def normalize(field: np.ndarray) -> np.ndarray:
        centered = field - float(np.mean(field))
        scale = float(np.max(np.abs(centered)))
        return centered / max(scale, 1.0e-12)

    return normalize(rad), normalize(evap), normalize(cur)


def _volume_preserve_surface_speed(speed: np.ndarray, geometry: Full3DGeometry) -> np.ndarray:
    side = geometry.vertices[geometry.side_indices]
    z = np.mean(side[:, :, 2], axis=1)
    radius = np.linalg.norm(side[:, :, :2], axis=2)
    if len(z) > 1:
        dz = np.gradient(z)
    else:
        dz = np.ones_like(z)
    weights = np.maximum(radius * np.abs(dz)[:, None], 1.0e-18)
    lagrange = float(np.sum(speed * weights) / max(np.sum(weights), 1.0e-18))
    return speed - lagrange


def _radial_bounds_for_side(cfg, geometry: Full3DGeometry) -> tuple[np.ndarray, np.ndarray]:
    side = geometry.vertices[geometry.side_indices]
    z = np.mean(side[:, :, 2], axis=1)
    z_norm = (z - np.min(z)) / max(float(np.max(z) - np.min(z)), 1.0e-12)
    center = np.sin(math.pi * np.clip(z_norm, 0.0, 1.0))[:, None]
    end_relief = 1.0 - 0.65 * center
    min_radius = max(float(getattr(cfg, "min_radius", 8.0e-4)), 1.0e-5)
    max_radius = max(float(getattr(cfg, "full3d_global_max_radius_m", 5.0e-3)), float(cfg.radius))
    min_field = np.full((geometry.num_rings, geometry.num_segments), min_radius, dtype=np.float64)
    max_by_ring = max_radius * end_relief + float(cfg.radius) * (1.0 - end_relief)
    max_field = np.broadcast_to(max_by_ring, (geometry.num_rings, geometry.num_segments)).copy()
    return min_field, max_field


def _apply_full3d_topology_action(
    cfg,
    geometry: Full3DGeometry,
    action: np.ndarray,
    amplitude: float,
) -> Full3DGeometry:
    """Apply a full initial-shape topology action.

    The action is a compact strategy-field parameterization. It emits spatial
    weights over radiation, evaporation and current-density sensitivities; a
    volume-preserving operator subtracts a Lagrange multiplier from the raw
    normal speed before the closed mesh is projected to the target volume.
    """

    strategy, cap_coeff, _side_coeff, layout = _full3d_physics_strategy_fields(cfg, geometry, action)
    circum_modes = int(layout["circum_modes"])
    cap_modes = int(layout["cap_modes"])
    circum = _circum_basis(geometry.num_segments, circum_modes)

    vertices = geometry.vertices.copy()
    free = _free_vertex_mask(geometry)
    rad_s, evap_s, cur_s = _side_physics_sensitivities(cfg, geometry)
    alpha_rad = 0.5 * (strategy[0] + 1.0)
    alpha_evap = 0.5 * (strategy[1] + 1.0) if strategy.shape[0] > 1 else 0.5
    alpha_cur = 0.5 * (strategy[2] + 1.0) if strategy.shape[0] > 2 else 0.5
    direct = strategy[3] if strategy.shape[0] > 3 else 0.0
    raw_speed = alpha_rad * rad_s - alpha_evap * evap_s + alpha_cur * cur_s + 0.35 * direct
    speed = _volume_preserve_surface_speed(raw_speed, geometry)
    side_ids = geometry.side_indices.reshape(-1)
    side_disp = speed.reshape(-1)
    side_valid = side_ids >= 0
    side_ids = side_ids[side_valid]
    side_disp = side_disp[side_valid]
    side_ids = side_ids[free[side_ids]]
    side_disp = side_disp[free[geometry.side_indices.reshape(-1)[side_valid]]]
    radial = vertices[side_ids, :2]
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_dir = radial / np.maximum(radial_norm, 1.0e-12)
    vertices[side_ids, :2] += float(amplitude) * side_disp[:, None] * radial_dir
    min_radius, max_radius = _radial_bounds_for_side(cfg, geometry)
    all_side_ids = geometry.side_indices.reshape(-1)
    valid_side = all_side_ids >= 0
    all_side_ids = all_side_ids[valid_side]
    target_min = min_radius.reshape(-1)[valid_side]
    target_max = max_radius.reshape(-1)[valid_side]
    radial = vertices[all_side_ids, :2]
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    target_radius = np.clip(radial_norm[:, 0], target_min, target_max)
    vertices[all_side_ids, :2] = radial / np.maximum(radial_norm, 1.0e-12) * target_radius[:, None]

    for cap_idx, grid in enumerate((geometry.lower_cap_indices[1:], geometry.upper_cap_indices[1:])):
        cap_vertices = vertices[grid]
        radial_fraction = np.linalg.norm(cap_vertices[:, :, :2], axis=2) / max(float(cfg.radius), 1.0e-12)
        inward = np.clip(1.0 - radial_fraction, 0.0, 1.0)
        cap_basis = _chebyshev_basis(2.0 * inward[:, 0] - 1.0, cap_modes)
        cap_fade = np.sin(0.5 * math.pi * inward)
        cap_field = _basis_field(cap_basis, cap_coeff[cap_idx], circum) * cap_fade
        cap_ids = grid.reshape(-1)
        cap_disp = float(amplitude) * 0.85 * cap_field.reshape(-1)
        cap_valid = (cap_ids >= 0) & free[np.maximum(cap_ids, 0)]
        if np.any(cap_valid):
            target_ids = cap_ids[cap_valid]
            target_disp = cap_disp[cap_valid]
            radial = vertices[target_ids, :2]
            radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
            radial_dir = radial / np.maximum(radial_norm, 1.0e-12)
            vertices[target_ids, :2] += target_disp[:, None] * radial_dir

    return _clone_geometry(geometry, vertices)


def build_full3d_initial_shape_from_action(
    cfg,
    action: np.ndarray,
    target_volume: float | None = None,
    use_neural_policy: bool = False,
    policy: Full3DUNetGNNPolicy | None = None,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> Full3DGeometry:
    """Generate one complete pre-energization initial shape from a global action.

    Unlike a local hill-climb step, this starts from the specified 5 mm x 15 mm
    cylinder only as material inventory and parameterization support. The final
    candidate is the optimized initial geometry with equal material volume.
    """

    target = (
        math.pi * float(cfg.radius) ** 2 * float(cfg.height)
        if target_volume is None
        else float(target_volume)
    )
    geom = build_baseline_full3d_geometry(cfg)
    steps = max(int(getattr(cfg, "full3d_global_shape_steps", 4)), 1)
    step_size = float(getattr(cfg, "full3d_global_step_m", 6.0e-4))
    decay = float(getattr(cfg, "full3d_global_step_decay", 0.96))
    local_rng = rng if rng is not None else np.random.default_rng(0)
    for step_idx in range(steps):
        geom = _apply_full3d_topology_action(cfg, geom, action, step_size * (decay ** step_idx))
        if bool(use_neural_policy) and policy is not None:
            policy_device = device if device is not None else torch.device("cpu")
            volume = _geometry_to_policy_volume(cfg, geom, policy_device)
            with torch.no_grad():
                strategy = policy(volume)
            neural_amp = float(getattr(cfg, "full3d_neural_policy_amplitude_m", 7.5e-5))
            geom = _apply_strategy_displacement(
                cfg,
                geom,
                strategy,
                neural_amp * float(local_rng.uniform(0.35, 1.0)),
            )
        geom = project_full3d_geometry(cfg, geom, target)
    return project_full3d_geometry(cfg, geom, target)


def _seed_full3d_actions(cfg) -> list[np.ndarray]:
    layout = _full3d_action_layout(cfg)
    action_dim = int(layout["action_dim"])
    axial_modes = int(layout["axial_modes"])
    circum_terms = int(layout["circum_terms"])
    side_channel_dim = int(layout["side_channel_dim"])
    side_dim = int(layout["side_dim"])
    seeds = [np.zeros(action_dim, dtype=np.float64)]

    def add_side(channel: int, mode: int, term: int, value: float) -> None:
        action = np.zeros(action_dim, dtype=np.float64)
        if mode < axial_modes and term < circum_terms:
            idx = channel * side_channel_dim + mode * circum_terms + term
            if idx < side_dim:
                action[idx] = float(value)
                seeds.append(action)

    for channel in range(min(int(layout["strategy_channels"]), 4)):
        add_side(channel, 0, 0, 0.9 if channel != 1 else -0.9)
        add_side(channel, 0, 0, -0.9 if channel != 1 else 0.9)
        add_side(channel, 1, 0, 0.7)
        add_side(channel, 1, 0, -0.7)
    if circum_terms > 1:
        for channel in range(min(int(layout["strategy_channels"]), 4)):
            action = np.zeros(action_dim, dtype=np.float64)
            idx = channel * side_channel_dim + 0 * circum_terms + 1
            if idx < side_dim:
                action[idx] = 0.65
            idx = channel * side_channel_dim + 0 * circum_terms + 2
            if idx < side_dim:
                action[idx] = -0.65
            seeds.append(action)

    cap_start = side_dim
    for cap_idx in range(2):
        action = np.zeros(action_dim, dtype=np.float64)
        action[cap_start + cap_idx * int(layout["cap_modes"]) * circum_terms] = 0.6
        seeds.append(action)
        action = np.zeros(action_dim, dtype=np.float64)
        action[cap_start + cap_idx * int(layout["cap_modes"]) * circum_terms] = -0.6
        seeds.append(action)
    return seeds


def _polygon_area_xy(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _axial_resistance_shape_factor(geometry: Full3DGeometry, cfg) -> float:
    side = geometry.vertices[geometry.side_indices]
    ring_z = np.mean(side[:, :, 2], axis=1)
    ring_area = np.asarray([_polygon_area_xy(points[:, :2]) for points in side], dtype=np.float64)
    order = np.argsort(ring_z)
    z = ring_z[order]
    area = np.maximum(ring_area[order], 1.0e-12)
    dz = np.diff(z)
    valid = dz > 1.0e-10
    if not np.any(valid):
        return float(cfg.height) / max(math.pi * float(cfg.radius) ** 2, 1.0e-12)
    inv_area = 0.5 * (1.0 / area[:-1] + 1.0 / area[1:])
    shape_factor = float(np.sum(dz[valid] * inv_area[valid]))
    baseline = float(cfg.height) / max(math.pi * float(cfg.radius) ** 2, 1.0e-12)
    return max(shape_factor, 0.05 * baseline)


def _side_face_count(geometry: Full3DGeometry) -> int:
    return int(max(geometry.num_rings - 1, 0) * int(geometry.num_segments) * 2)


def _side_surface_lumped_by_ring(
    geometry: Full3DGeometry,
    area: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    side_faces = int(min(_side_face_count(geometry), len(geometry.faces)))
    free_area = np.zeros(int(geometry.num_rings), dtype=np.float64)
    effective_area = np.zeros(int(geometry.num_rings), dtype=np.float64)
    if side_faces <= 0:
        return free_area, effective_area, 0.0
    mesh_center = np.mean(geometry.vertices, axis=0)
    center_to_face = centers[:side_faces] - mesh_center
    escape = np.clip(
        np.sum(normals[:side_faces] * center_to_face, axis=1)
        / np.maximum(np.linalg.norm(center_to_face, axis=1), 1.0e-12),
        0.0,
        1.0,
    )
    face_idx = 0
    for ridx in range(int(geometry.num_rings) - 1):
        for _ in range(int(geometry.num_segments)):
            for _tri in range(2):
                patch_area = float(area[face_idx])
                escaped = patch_area * float(escape[face_idx])
                free_area[ridx] += 0.5 * patch_area
                free_area[ridx + 1] += 0.5 * patch_area
                effective_area[ridx] += 0.5 * escaped
                effective_area[ridx + 1] += 0.5 * escaped
                face_idx += 1
    side_total_area = float(np.sum(area[:side_faces]))
    return free_area, effective_area, side_total_area


def _full3d_axial_profile(geometry: Full3DGeometry) -> tuple[np.ndarray, np.ndarray]:
    side = geometry.vertices[geometry.side_indices]
    z = np.mean(side[:, :, 2], axis=1).astype(np.float64)
    cross_section_area = np.asarray([_polygon_area_xy(points[:, :2]) for points in side], dtype=np.float64)
    return z, np.maximum(cross_section_area, 1.0e-12)


def _band_fraction_for_balance(cfg, temperature: float) -> float:
    quantized = round(max(float(temperature), 100.0) / 25.0) * 25.0
    return float(blackbody_band_fraction(float(quantized), float(cfg.in_band_upper_um)))


def _total_radiative_emissivity(cfg, temperature: float) -> float:
    band_fraction = _band_fraction_for_balance(cfg, temperature)
    return float(cfg.band_emissivity) * band_fraction + float(cfg.out_of_band_emissivity) * (1.0 - band_fraction)


def _material_properties_np(cfg, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.maximum(np.asarray(temperature, dtype=np.float64) - 300.0, 0.0)
    k = np.maximum(float(cfg.k_ref) + float(cfg.k_temp_coeff) * delta, 20.0)
    rho_elec = float(cfg.rho_elec_ref) * (1.0 + float(cfg.rho_elec_temp_coeff) * delta)
    return k, rho_elec


def _evaporation_flux_np(cfg, temperature: np.ndarray) -> np.ndarray:
    temp = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0)
    return float(cfg.evap_A) * np.exp(float(cfg.evap_B) / temp) * 10.0


def _radiation_sink_temperature(cfg) -> float:
    return float(getattr(cfg, "full3d_thermal_sink_temperature_k", float(cfg.ambient_temp)))


def _radiative_loss_by_node(cfg, temperature: np.ndarray, effective_area: np.ndarray) -> np.ndarray:
    sink_temp = _radiation_sink_temperature(cfg)
    emissivity = np.asarray([_total_radiative_emissivity(cfg, temp) for temp in temperature], dtype=np.float64)
    return (
        emissivity
        * float(cfg.stefan_boltzmann)
        * float(getattr(cfg, "radiative_cooling_scale", 1.0))
        * effective_area
        * np.maximum(np.power(temperature, 4) - sink_temp ** 4, 0.0)
    )


def _radiative_loss_slope_by_node(cfg, temperature: np.ndarray, effective_area: np.ndarray) -> np.ndarray:
    emissivity = np.asarray([_total_radiative_emissivity(cfg, temp) for temp in temperature], dtype=np.float64)
    return (
        4.0
        * emissivity
        * float(cfg.stefan_boltzmann)
        * float(getattr(cfg, "radiative_cooling_scale", 1.0))
        * effective_area
        * np.maximum(temperature, 1.0) ** 3
    )


def _evaporative_loss_by_node(cfg, temperature: np.ndarray, free_area: np.ndarray) -> np.ndarray:
    return _evaporation_flux_np(cfg, temperature) * free_area * float(cfg.latent_heat_evap)


def _evaporative_loss_slope_by_node(cfg, temperature: np.ndarray, free_area: np.ndarray) -> np.ndarray:
    temp = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0)
    flux = _evaporation_flux_np(cfg, temp)
    dflux_dt = flux * (-float(cfg.evap_B) / np.maximum(temp * temp, 1.0e-12))
    return dflux_dt * free_area * float(cfg.latent_heat_evap)


def _solve_full3d_thermal_state(
    cfg,
    voltage: float,
    geometry: Full3DGeometry,
    area: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
) -> Dict[str, float]:
    voltage = max(float(voltage), 0.0)
    electrode_temp = float(cfg.ambient_temp)
    z, cross_area = _full3d_axial_profile(geometry)
    free_area, effective_area, side_total_area = _side_surface_lumped_by_ring(geometry, area, normals, centers)
    contact_metrics = _end_contact_metrics(geometry, cfg)
    order = np.argsort(z)
    inv_order = np.argsort(order)
    z_s = z[order]
    area_s = cross_area[order]
    free_s = free_area[order]
    effective_s = effective_area[order]
    lower_contact = float(contact_metrics["lower_electrode_contact_area_m2"])
    upper_contact = float(contact_metrics["upper_electrode_contact_area_m2"])
    contact_s = np.zeros_like(area_s)
    lower_sorted_idx = int(np.argmin(z_s))
    upper_sorted_idx = int(np.argmax(z_s))
    contact_s[lower_sorted_idx] = lower_contact
    contact_s[upper_sorted_idx] = upper_contact
    dz = np.maximum(np.diff(z_s), 1.0e-8)
    if len(z_s) < 2:
        resistance = max(float(cfg.rho_elec_ref) * float(cfg.height) / max(math.pi * float(cfg.radius) ** 2, 1.0e-12), float(cfg.min_resistance))
        current = min(voltage / resistance, float(cfg.max_current))
        return {
            "temperature_profile_k": np.asarray([electrode_temp], dtype=np.float64),
            "mean_temperature_k": electrode_temp,
            "max_temperature_k": electrode_temp,
            "resistance_ohm": float(resistance),
            "current_a": float(current),
            "electrical_power_w": float(voltage * current),
            "full_spectrum_radiative_power_w": 0.0,
            "evaporative_power_w": 0.0,
            "thermal_balance_residual_w": 0.0,
            "thermal_converged": True,
            "side_surface_area_m2": 0.0,
            "effective_radiating_area_m2": 0.0,
            "escape_view_factor_proxy": 0.0,
            "axial_resistance_shape_factor_m_inv": float(cfg.height) / max(math.pi * float(cfg.radius) ** 2, 1.0e-12),
            "electrode_conducted_power_w": float(voltage * current),
            "electrode_contact_area_m2": float(contact_metrics["electrode_contact_area_m2"]),
            "noncontact_end_face_area_m2": float(contact_metrics["noncontact_end_face_area_m2"]),
        }

    temperature = np.full(len(z_s), electrode_temp, dtype=np.float64)
    thermal_residual = 0.0
    current = 0.0
    resistance = float(cfg.min_resistance)
    electrical_power = 0.0
    full_radiative_power = 0.0
    evaporative_power = 0.0
    max_delta_observed = 0.0
    thermal_converged = False

    for _ in range(int(cfg.thermal_max_iters)):
        _, rho_nodes = _material_properties_np(cfg, temperature)
        rho_seg = 0.5 * (rho_nodes[:-1] + rho_nodes[1:])
        area_seg = np.maximum(0.5 * (area_s[:-1] + area_s[1:]), 1.0e-12)
        segment_resistance = rho_seg * dz / area_seg
        resistance = max(float(np.sum(segment_resistance)), float(cfg.min_resistance))
        current = min(voltage / resistance, float(cfg.max_current))
        joule_seg = current * current * segment_resistance
        joule_node = np.zeros_like(temperature)
        joule_node[:-1] += 0.5 * joule_seg
        joule_node[1:] += 0.5 * joule_seg

        k_nodes, _ = _material_properties_np(cfg, temperature)
        k_seg = 0.5 * (k_nodes[:-1] + k_nodes[1:])
        conductance = k_seg * area_seg / dz
        contact_length = max(float(getattr(cfg, "full3d_electrode_contact_length_m", 2.0e-4)), 1.0e-8)
        contact_conductance = k_nodes * contact_s / contact_length
        radiation = _radiative_loss_by_node(cfg, temperature, free_s)
        evaporation = _evaporative_loss_by_node(cfg, temperature, free_s)
        rad_slope = _radiative_loss_slope_by_node(cfg, temperature, free_s)
        evap_slope = _evaporative_loss_slope_by_node(cfg, temperature, free_s)

        residual = np.zeros_like(temperature)
        residual[0] += conductance[0] * (temperature[1] - temperature[0])
        residual[-1] += conductance[-1] * (temperature[-2] - temperature[-1])
        residual[1:-1] += conductance[:-1] * (temperature[:-2] - temperature[1:-1])
        residual[1:-1] += conductance[1:] * (temperature[2:] - temperature[1:-1])
        residual += joule_node - radiation - evaporation + contact_conductance * (electrode_temp - temperature)
        thermal_residual = float(np.max(np.abs(residual))) if len(temperature) > 0 else 0.0

        node_count = len(temperature)
        matrix = np.zeros((node_count, node_count), dtype=np.float64)
        rhs = np.zeros(node_count, dtype=np.float64)
        for ridx in range(node_count):
            left = float(conductance[ridx - 1]) if ridx > 0 else 0.0
            right = float(conductance[ridx]) if ridx < node_count - 1 else 0.0
            sink_slope = float(rad_slope[ridx] + evap_slope[ridx])
            contact = float(contact_conductance[ridx])
            matrix[ridx, ridx] = left + right + sink_slope + contact
            if ridx > 0:
                matrix[ridx, ridx - 1] = -left
            if ridx < node_count - 1:
                matrix[ridx, ridx + 1] = -right
            rhs[ridx] += float(
                joule_node[ridx]
                - radiation[ridx]
                - evaporation[ridx]
                + sink_slope * temperature[ridx]
                + contact * electrode_temp
            )
        try:
            solved = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            diag = np.zeros_like(temperature)
            diag += rad_slope + evap_slope + contact_conductance
            diag[:-1] += conductance
            diag[1:] += conductance
            solved = temperature + residual / np.maximum(
                diag,
                1.0e-9,
            )
        updated = temperature.copy()
        updated[:] = solved
        updated = np.maximum(updated, electrode_temp)
        delta_limit = float(getattr(cfg, "full3d_thermal_max_delta_k", 1200.0))
        step = np.clip(updated - temperature, -delta_limit, delta_limit)
        updated = temperature + float(cfg.thermal_relaxation) * step
        updated = np.maximum(updated, electrode_temp)
        max_delta_observed = float(np.max(np.abs(updated - temperature)))
        temperature = updated
        if max_delta_observed < float(cfg.thermal_tol_k) and thermal_residual <= float(getattr(cfg, "full3d_thermal_residual_tol_w", 1.0e-3)):
            thermal_converged = True
            break

    _, rho_nodes = _material_properties_np(cfg, temperature)
    rho_seg = 0.5 * (rho_nodes[:-1] + rho_nodes[1:])
    area_seg = np.maximum(0.5 * (area_s[:-1] + area_s[1:]), 1.0e-12)
    segment_resistance = rho_seg * dz / area_seg
    resistance = max(float(np.sum(segment_resistance)), float(cfg.min_resistance))
    current = min(voltage / resistance, float(cfg.max_current))
    electrical_power = voltage * current
    full_radiative_by_node = _radiative_loss_by_node(cfg, temperature, free_s)
    evaporative_by_node = _evaporative_loss_by_node(cfg, temperature, free_s)
    full_radiative_power = float(np.sum(full_radiative_by_node))
    evaporative_power = float(np.sum(evaporative_by_node))
    k_nodes, _ = _material_properties_np(cfg, temperature)
    contact_length = max(float(getattr(cfg, "full3d_electrode_contact_length_m", 2.0e-4)), 1.0e-8)
    contact_conductance = k_nodes * contact_s / contact_length
    residual = np.zeros_like(temperature)
    residual[0] += conductance[0] * (temperature[1] - temperature[0])
    residual[-1] += conductance[-1] * (temperature[-2] - temperature[-1])
    residual[1:-1] += conductance[:-1] * (temperature[:-2] - temperature[1:-1])
    residual[1:-1] += conductance[1:] * (temperature[2:] - temperature[1:-1])
    joule_seg = current * current * segment_resistance
    joule_node = np.zeros_like(temperature)
    joule_node[:-1] += 0.5 * joule_seg
    joule_node[1:] += 0.5 * joule_seg
    residual += joule_node - full_radiative_by_node - evaporative_by_node + contact_conductance * (electrode_temp - temperature)
    electrode_conducted = float(np.sum(contact_conductance * np.maximum(temperature - electrode_temp, 0.0)))
    thermal_residual = float(np.max(np.abs(residual))) if len(temperature) > 0 else 0.0
    thermal_converged = bool(
        thermal_converged
        or (
            max_delta_observed < float(cfg.thermal_tol_k)
            and thermal_residual <= float(getattr(cfg, "full3d_thermal_residual_tol_w", 1.0e-3))
        )
    )
    unsorted_temperature = temperature[inv_order]
    return {
        "temperature_profile_k": unsorted_temperature,
        "mean_temperature_k": float(np.mean(temperature)),
        "max_temperature_k": float(np.max(temperature)),
        "resistance_ohm": float(resistance),
        "current_a": float(current),
        "electrical_power_w": float(electrical_power),
        "full_spectrum_radiative_power_w": float(full_radiative_power),
        "evaporative_power_w": float(evaporative_power),
        "thermal_balance_residual_w": float(thermal_residual),
        "thermal_max_delta_k": float(max_delta_observed),
        "thermal_converged": bool(thermal_converged),
        "side_surface_area_m2": float(side_total_area),
        "effective_radiating_area_m2": float(np.sum(effective_s)),
        "escape_view_factor_proxy": float(np.sum(effective_s) / max(float(side_total_area), 1.0e-12)),
        "axial_resistance_shape_factor_m_inv": float(np.sum(dz / area_seg)),
        "electrode_conducted_power_w": electrode_conducted,
        "electrode_contact_area_m2": float(contact_metrics["electrode_contact_area_m2"]),
        "lower_electrode_contact_area_m2": lower_contact,
        "upper_electrode_contact_area_m2": upper_contact,
        "noncontact_end_face_area_m2": float(contact_metrics["noncontact_end_face_area_m2"]),
        "missing_electrode_contact_area_m2": float(contact_metrics["missing_electrode_contact_area_m2"]),
    }


def _full3d_fixed_voltage(cfg) -> float | None:
    voltage = getattr(cfg, "full3d_fixed_voltage_v", None)
    return None if voltage is None else float(voltage)


def _full3d_voltage_grid(min_voltage: float, max_voltage: float, points: int, spacing: str = "log") -> np.ndarray:
    lo = max(float(min_voltage), 1.0e-6)
    hi = max(float(max_voltage), lo)
    count = max(int(points), 1)
    if count == 1:
        return np.asarray([hi], dtype=np.float64)
    if str(spacing).lower() == "log":
        return np.geomspace(lo, hi, count, dtype=np.float64)
    return np.linspace(lo, hi, count, dtype=np.float64)


def _full3d_seed_voltages(cfg) -> np.ndarray:
    seeds = [
        0.25,
        0.30,
        0.34,
        0.40,
        0.50,
    ]
    return np.asarray(
        [value for value in seeds if float(cfg.min_voltage) <= value <= float(cfg.max_voltage)],
        dtype=np.float64,
    )


def _evaluate_full3d_geometry_at_voltage(
    cfg,
    geometry: Full3DGeometry,
    voltage: float,
    baseline_metrics: Dict[str, float] | None = None,
) -> Dict[str, float]:
    target_volume = math.pi * float(cfg.radius) ** 2 * float(cfg.height)
    volume = mesh_volume(geometry)
    volume_change = abs(volume - target_volume) / max(target_volume, 1.0e-18)
    area, normals, centers = _mesh_area_normals_centers(geometry)
    voltage = float(voltage)
    surface_area = float(np.sum(area))
    cylinder_area = 2.0 * math.pi * float(cfg.radius) * float(cfg.height) + 2.0 * math.pi * float(cfg.radius) ** 2
    contact_metrics = _end_contact_metrics(geometry, cfg)
    thermal = _solve_full3d_thermal_state(cfg, voltage, geometry, area, normals, centers)
    temperature_profile = np.asarray(thermal["temperature_profile_k"], dtype=np.float64)
    max_temperature = float(thermal["max_temperature_k"])
    mean_temperature = float(thermal["mean_temperature_k"])
    free_area, effective_area_by_ring, side_area = _side_surface_lumped_by_ring(geometry, area, normals, centers)
    band_fraction_by_ring = np.asarray(
        [blackbody_band_fraction(float(temp), float(cfg.in_band_upper_um)) for temp in temperature_profile],
        dtype=np.float64,
    )
    thermal_sink_temp = _radiation_sink_temperature(cfg)
    sphere_temp = float(getattr(cfg, "full3d_sphere_temperature_k", 0.0))
    band_power_0k_sphere = float(np.sum(
        float(cfg.band_emissivity)
        * float(cfg.stefan_boltzmann)
        * float(getattr(cfg, "radiative_cooling_scale", 1.0))
        * effective_area_by_ring
        * np.maximum(np.power(temperature_profile, 4) - sphere_temp ** 4, 0.0)
        * band_fraction_by_ring
    ))
    band_power_300k_environment = float(np.sum(
        float(cfg.band_emissivity)
        * float(cfg.stefan_boltzmann)
        * float(getattr(cfg, "radiative_cooling_scale", 1.0))
        * effective_area_by_ring
        * np.maximum(np.power(temperature_profile, 4) - thermal_sink_temp ** 4, 0.0)
        * band_fraction_by_ring
    ))
    net_band_power = band_power_0k_sphere
    evap_flux = _evaporation_flux_np(cfg, temperature_profile)
    mass_loss_rate = float(np.sum(evap_flux * free_area))
    recession_rate = evap_flux / max(float(cfg.density), 1.0e-12)
    hot_free = free_area > 1.0e-16
    if np.any(hot_free):
        lifetime_s = float(
            np.min(
                float(cfg.feature_fail_ratio)
                * float(cfg.radius)
                / np.maximum(recession_rate[hot_free], float(getattr(cfg, "full3d_lifetime_recession_floor_m_s", 1.0e-300)))
            )
        )
        lifetime_s = min(lifetime_s, float(getattr(cfg, "full3d_lifetime_cap_s", 1.0e300)))
    else:
        lifetime_s = float(getattr(cfg, "full3d_lifetime_cap_s", 1.0e300))
    if baseline_metrics is None:
        baseline_life = lifetime_s
        baseline_power = max(net_band_power, 1.0e-9)
        baseline_efficiency = max(net_band_power / max(float(thermal["electrical_power_w"]), 1.0e-12), 1.0e-12)
    else:
        baseline_life = max(float(baseline_metrics.get("lifetime_s", lifetime_s)), 1.0e-9)
        baseline_power = max(float(baseline_metrics.get("net_radiated_power_0k_sphere_w", net_band_power)), 1.0e-9)
        baseline_efficiency = max(
            float(
                baseline_metrics.get(
                    "energy_conversion_efficiency_0_3um",
                    baseline_metrics.get("net_radiated_power_0k_sphere_w", net_band_power)
                    / max(float(baseline_metrics.get("electrical_power_w", thermal["electrical_power_w"])), 1.0e-12),
                )
            ),
            1.0e-12,
        )
    lifetime_ratio = lifetime_s / baseline_life
    efficiency = net_band_power / max(float(thermal["electrical_power_w"]), 1.0e-12)
    efficiency_ratio = efficiency / baseline_efficiency
    electrode_error = _electrode_error(geometry, cfg)
    temperature_violation_ratio = max(max_temperature / max(float(cfg.max_temp), 1.0e-9) - 1.0, 0.0)
    thermal_converged = bool(thermal.get("thermal_converged", False))
    thermal_residual_violation = max(
        float(thermal["thermal_balance_residual_w"]) / max(float(getattr(cfg, "full3d_thermal_residual_tol_w", 1.0e-3)), 1.0e-12) - 1.0,
        0.0,
    )
    lifecycle_reference = float(getattr(cfg, "full3d_lifecycle_reference_s", 1.0e27))
    lifecycle_factor = lifetime_s / max(lifetime_s + lifecycle_reference, 1.0e-300)
    rated_operating_score = efficiency * lifecycle_factor
    radiation_efficiency_score = efficiency_ratio
    feasible = (
        volume_change <= float(getattr(cfg, "full3d_volume_tolerance_ratio", 1.0e-5))
        and electrode_error <= float(getattr(cfg, "full3d_electrode_tolerance_m", 2.0e-6))
        and max_temperature <= float(cfg.max_temp)
        and lifetime_ratio >= float(cfg.minimum_lifetime_ratio)
        and thermal_converged
    )
    score = (
        efficiency_ratio
        + 0.08 * (net_band_power / baseline_power)
        + 0.05 * min(lifetime_ratio, 10.0)
        + 0.02 * (float(thermal["effective_radiating_area_m2"]) / max(cylinder_area, 1.0e-12))
        - 80.0 * volume_change
        - 1.0e5 * electrode_error
        - 60.0 * temperature_violation_ratio
        - 50.0 * max(float(cfg.minimum_lifetime_ratio) - lifetime_ratio, 0.0)
        - 5.0 * thermal_residual_violation
    )
    if not feasible:
        score -= 10.0
    return {
        "score": float(score),
        "voltage_v": voltage,
        "current_a": float(thermal["current_a"]),
        "resistance_ohm": float(thermal["resistance_ohm"]),
        "electrical_power_w": float(thermal["electrical_power_w"]),
        "full_spectrum_radiative_power_w": float(thermal["full_spectrum_radiative_power_w"]),
        "evaporative_power_w": float(thermal["evaporative_power_w"]),
        "thermal_balance_residual_w": float(thermal["thermal_balance_residual_w"]),
        "thermal_max_delta_k": float(thermal.get("thermal_max_delta_k", 0.0)),
        "thermal_converged": bool(thermal_converged),
        "mean_temperature_k": mean_temperature,
        "max_temperature_k": max_temperature,
        "net_radiated_power_0k_sphere_w": float(net_band_power),
        "net_radiated_power_300k_environment_w": float(band_power_300k_environment),
        "energy_conversion_efficiency_0_3um": float(efficiency),
        "energy_conversion_efficiency_ratio": float(efficiency_ratio),
        "objective": "maximize initial-state 0-3um escaped radiant efficiency under rated voltage search",
        "rated_operating_score": float(rated_operating_score),
        "radiation_efficiency_score": float(radiation_efficiency_score),
        "lifecycle_factor_full3d": float(lifecycle_factor),
        "effective_radiating_area_m2": float(thermal["effective_radiating_area_m2"]),
        "side_surface_area_m2": float(side_area),
        "surface_area_m2": float(surface_area),
        "surface_area_ratio": float(surface_area / max(cylinder_area, 1.0e-12)),
        "free_radiating_surface_area_m2": float(side_area),
        "free_surface_thermal_balance_area_m2": float(side_area),
        "contact_end_face_area_m2": float(contact_metrics["electrode_contact_area_m2"]),
        "end_face_area_m2": float(contact_metrics["end_face_area_m2"]),
        "electrode_disk_area_m2": float(contact_metrics["electrode_disk_area_m2"]),
        "electrode_contact_area_m2": float(contact_metrics["electrode_contact_area_m2"]),
        "lower_electrode_contact_area_m2": float(contact_metrics["lower_electrode_contact_area_m2"]),
        "upper_electrode_contact_area_m2": float(contact_metrics["upper_electrode_contact_area_m2"]),
        "noncontact_end_face_area_m2": float(contact_metrics["noncontact_end_face_area_m2"]),
        "missing_electrode_contact_area_m2": float(contact_metrics["missing_electrode_contact_area_m2"]),
        "lower_contact_fraction_of_tungsten_end": float(contact_metrics["lower_contact_fraction_of_tungsten_end"]),
        "upper_contact_fraction_of_tungsten_end": float(contact_metrics["upper_contact_fraction_of_tungsten_end"]),
        "axial_resistance_shape_factor_m_inv": float(thermal["axial_resistance_shape_factor_m_inv"]),
        "escape_view_factor_proxy": float(thermal["escape_view_factor_proxy"]),
        "blackbody_band_fraction_0_3um": float(np.average(band_fraction_by_ring, weights=np.maximum(effective_area_by_ring, 1.0e-18))),
        "lifetime_s": float(lifetime_s),
        "lifetime_ratio_3d": float(lifetime_ratio),
        "mass_loss_rate_kg_s": float(mass_loss_rate),
        "temperature_violation_ratio": float(temperature_violation_ratio),
        "volume_m3": float(volume),
        "volume_change_ratio_3d": float(volume_change),
        "electrode_max_error_m": float(electrode_error),
        "constraint_feasible_3d": bool(feasible),
        "top_bottom_faces_variable": True,
        "electrode_diameter_mm": 5.0,
        "initial_shape_reference": "specified 5 mm x 15 mm cylinder baseline",
        "electrode_contact_model": "fixed 5 mm electrode disks; tungsten end-face footprint may be smaller or larger, and only overlap area conducts to 300 K",
        "external_sphere_temperature_k": sphere_temp,
        "external_sphere_emissivity": float(getattr(cfg, "full3d_sphere_emissivity", 1.0)),
        "thermal_radiation_sink_temperature_k": thermal_sink_temp,
        "electrode_boundary_temperature_k": float(cfg.ambient_temp),
        "tungsten_voltage_v": voltage,
        "electrode_voltage_drop_v": 0.0,
        "contact_resistance_ohm": 0.0,
        "contact_thermal_resistance_k_w": 0.0,
        "electrode_conducted_power_w": float(thermal.get("electrode_conducted_power_w", 0.0)),
        "voltage_search_mode": "fixed_voltage",
    }


def _full3d_voltage_rank(metrics: Dict[str, float]) -> tuple[int, float, float]:
    feasible = 1 if bool(metrics.get("constraint_feasible_3d", False)) else 0
    if feasible:
        return (
            feasible,
            float(metrics.get("rated_operating_score", metrics.get("energy_conversion_efficiency_0_3um", 0.0))),
            float(metrics.get("net_radiated_power_0k_sphere_w", 0.0)),
        )
    return (feasible, float(metrics.get("score", -float("inf"))), -float(metrics.get("temperature_violation_ratio", 0.0)))


def _search_full3d_rated_condition(
    cfg,
    geometry: Full3DGeometry,
    baseline_metrics: Dict[str, float] | None = None,
) -> Dict[str, float]:
    evaluated: Dict[float, Dict[str, float]] = {}

    def evaluate(voltage_v: float) -> Dict[str, float]:
        key = round(float(voltage_v), 8)
        if key not in evaluated:
            metrics = _evaluate_full3d_geometry_at_voltage(cfg, geometry, float(voltage_v), baseline_metrics)
            metrics["voltage_search_mode"] = "rated_search"
            evaluated[key] = metrics
        return evaluated[key]

    best: Dict[str, float] | None = None
    coarse_grid = _full3d_voltage_grid(
        float(cfg.min_voltage),
        float(cfg.max_voltage),
        int(cfg.voltage_grid_points),
        spacing=getattr(cfg, "voltage_grid_spacing", "log"),
    )
    seed_grid = _full3d_seed_voltages(cfg)
    search_grid = np.unique(np.concatenate([coarse_grid, seed_grid])) if seed_grid.size else coarse_grid
    for voltage_v in search_grid:
        metrics = evaluate(float(voltage_v))
        if best is None or _full3d_voltage_rank(metrics) > _full3d_voltage_rank(best):
            best = metrics

    if best is None:
        raise RuntimeError("Full3D rated-condition search failed to evaluate any voltage.")

    span = max(
        (float(cfg.max_voltage) - float(cfg.min_voltage)) * float(cfg.voltage_focus_ratio),
        (float(cfg.max_voltage) - float(cfg.min_voltage)) / max(float(int(cfg.voltage_grid_points) - 1), 1.0),
    )
    for _ in range(int(cfg.voltage_refine_levels)):
        center = float(best["voltage_v"])
        lo = max(float(cfg.min_voltage), center - span)
        hi = min(float(cfg.max_voltage), center + span)
        local_grid = sorted(
            _full3d_voltage_grid(lo, hi, int(cfg.voltage_refine_points), spacing=getattr(cfg, "voltage_refine_spacing", "log")),
            key=lambda value: abs(float(value) - center),
        )
        for voltage_v in local_grid:
            metrics = evaluate(float(voltage_v))
            if _full3d_voltage_rank(metrics) > _full3d_voltage_rank(best):
                best = metrics
        span *= 0.35

    feasible_count = sum(1 for item in evaluated.values() if bool(item.get("constraint_feasible_3d", False)))
    best = dict(best)
    best["voltage_search_mode"] = "rated_search"
    best["rated_voltage_upper_bound_v"] = float(cfg.max_voltage)
    best["voltage_search_evaluations"] = int(len(evaluated))
    best["voltage_search_feasible_count"] = int(feasible_count)
    return best


def evaluate_full3d_geometry(cfg, geometry: Full3DGeometry, baseline_metrics: Dict[str, float] | None = None) -> Dict[str, float]:
    fixed_voltage = _full3d_fixed_voltage(cfg)
    if fixed_voltage is not None:
        return _evaluate_full3d_geometry_at_voltage(cfg, geometry, fixed_voltage, baseline_metrics)
    return _search_full3d_rated_condition(cfg, geometry, baseline_metrics)


def _full3d_metric_rank(metrics: Dict[str, float]) -> tuple[int, float]:
    return (1 if bool(metrics.get("constraint_feasible_3d", False)) else 0, float(metrics.get("score", -float("inf"))))


def _select_full3d_candidate(candidates: list[tuple[Full3DGeometry, Dict[str, float]]]) -> tuple[Full3DGeometry, Dict[str, float]]:
    return max(candidates, key=lambda item: _full3d_metric_rank(item[1]))


def _full3d_metric_snapshot(metrics: Dict[str, float]) -> Dict[str, object]:
    keys = [
        "archive_index",
        "score",
        "constraint_feasible_3d",
        "voltage_v",
        "voltage_search_mode",
        "voltage_search_feasible_count",
        "energy_conversion_efficiency_0_3um",
        "energy_conversion_efficiency_ratio",
        "radiation_efficiency_score",
        "net_radiated_power_0k_sphere_w",
        "lifetime_ratio_3d",
        "max_temperature_k",
        "temperature_violation_ratio",
        "volume_change_ratio_3d",
        "electrode_max_error_m",
        "electrode_contact_area_m2",
        "noncontact_end_face_area_m2",
        "missing_electrode_contact_area_m2",
        "effective_radiating_area_m2",
        "surface_area_ratio",
        "escape_view_factor_proxy",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _full3d_selection_diagnostics(
    archive_metrics: List[Dict[str, float]],
    selected_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    cfg,
) -> Dict[str, object]:
    feasible = [item for item in archive_metrics if bool(item.get("constraint_feasible_3d", False))]
    nonbaseline = [item for item in archive_metrics if int(item.get("archive_index", -1)) != 0]
    feasible_nonbaseline = [item for item in nonbaseline if bool(item.get("constraint_feasible_3d", False))]
    best_by_score = max(archive_metrics, key=lambda item: float(item.get("score", -float("inf"))))
    best_feasible_by_power = (
        max(feasible, key=lambda item: float(item.get("net_radiated_power_0k_sphere_w", -float("inf"))))
        if feasible
        else None
    )
    best_feasible_by_efficiency = (
        max(feasible, key=lambda item: float(item.get("energy_conversion_efficiency_0_3um", -float("inf"))))
        if feasible
        else None
    )
    selected_idx = int(selected_metrics.get("archive_index", 0))
    baseline_feasible = bool(baseline_metrics.get("constraint_feasible_3d", False))
    fixed_voltage = _full3d_fixed_voltage(cfg)
    mode = "fixed_voltage" if fixed_voltage is not None else "rated_search"
    if not feasible:
        if mode == "fixed_voltage":
            reason = (
                "no full3d candidate satisfied fixed-voltage temperature, lifetime, volume, and electrode constraints; "
                "selected the highest penalized diagnostic score"
            )
        else:
            reason = (
                "no full3d candidate satisfied rated voltage-search temperature, lifetime, volume, and electrode constraints; "
                "selected the highest penalized diagnostic score"
            )
    elif selected_idx == 0:
        reason = "baseline retained: no feasible non-baseline initial shape improved rated 0-3um efficiency score"
    else:
        reason = "selected feasible globally parameterized initial shape with highest constrained rated-efficiency score"

    diagnostics: Dict[str, object] = {
        "selected_archive_index": selected_idx,
        "selection_reason_full3d": reason,
        "voltage_search_mode": mode,
        "fixed_voltage_v": fixed_voltage,
        "rated_voltage_upper_bound_v": float(cfg.max_voltage),
        "optimization_target_full3d": (
            "global initial-shape topology optimization: maximize rated-condition 0-3um escaped "
            "energy-conversion efficiency under volume equality, V<=100V, T<=3000C, and lifetime>=30% baseline"
        ),
        "action_space_full3d": (
            "Phy-DRL-LSM inspired global strategy field: Chebyshev(z) x Fourier(theta) coefficients emit "
            "radiation/evaporation/current/direct strategy maps; these weight physics sensitivities into a "
            "normal speed, then a Lagrange volume projection preserves material volume. Top/bottom footprints "
            "use Chebyshev(radius) x Fourier(theta) in-plane modes; axial end planes remain 15 mm apart and "
            "contact with the fixed 5 mm electrode disk is computed from footprint overlap after projection"
        ),
        "action_dim_full3d": int(full3d_action_dim(cfg)),
        "strategy_channels_full3d": int(getattr(cfg, "full3d_action_strategy_channels", 4)),
        "global_shape_steps_full3d": int(getattr(cfg, "full3d_global_shape_steps", 4)),
        "cem_elite_fraction_full3d": float(getattr(cfg, "full3d_cem_elite_fraction", 0.35)),
        "max_allowed_temperature_k": float(cfg.max_temp),
        "baseline_feasible_full3d": baseline_feasible,
        "archive_candidate_count": int(len(archive_metrics)),
        "archive_feasible_count": int(len(feasible)),
        "archive_feasible_nonbaseline_count": int(len(feasible_nonbaseline)),
        "best_archive_by_score": _full3d_metric_snapshot(best_by_score),
    }
    if best_feasible_by_power is not None:
        diagnostics["best_feasible_archive_by_0k_power"] = _full3d_metric_snapshot(best_feasible_by_power)
    if best_feasible_by_efficiency is not None:
        diagnostics["best_feasible_archive_by_efficiency"] = _full3d_metric_snapshot(best_feasible_by_efficiency)
    if nonbaseline:
        diagnostics["max_nonbaseline_surface_area_ratio"] = float(max(item.get("surface_area_ratio", 0.0) for item in nonbaseline))
        diagnostics["max_nonbaseline_effective_area_m2"] = float(max(item.get("effective_radiating_area_m2", 0.0) for item in nonbaseline))
        diagnostics["max_nonbaseline_efficiency_0_3um"] = float(max(item.get("energy_conversion_efficiency_0_3um", 0.0) for item in nonbaseline))
        diagnostics["min_nonbaseline_temperature_k"] = float(min(item.get("max_temperature_k", float("inf")) for item in nonbaseline))
    return diagnostics


def run_full3d_optimization(
    cfg,
    generations: int = 4,
    population_size: int = 16,
    seed: int = 42,
    use_neural_policy: bool = True,
) -> Full3DResult:
    rng = np.random.default_rng(int(seed))
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    baseline = build_baseline_full3d_geometry(cfg)
    target_volume = math.pi * float(cfg.radius) ** 2 * float(cfg.height)
    baseline = project_full3d_geometry(cfg, baseline, target_volume)
    baseline_metrics = evaluate_full3d_geometry(cfg, baseline)
    baseline_metrics["archive_index"] = 0
    policy = Full3DUNetGNNPolicy().to(device)
    policy.eval()
    best = baseline
    best_metrics = dict(baseline_metrics)
    history_geometries = [baseline]
    history_metrics = [dict(baseline_metrics, generation=0, step=0)]
    candidate_count = 1
    archive_metrics = [dict(baseline_metrics)]
    action_dim = full3d_action_dim(cfg)
    action_mean = np.zeros(action_dim, dtype=np.float64)
    action_sigma = np.full(action_dim, float(getattr(cfg, "full3d_cem_initial_sigma", 1.10)), dtype=np.float64)
    seed_actions = _seed_full3d_actions(cfg)

    for generation in range(1, int(generations) + 1):
        candidates: list[tuple[Full3DGeometry, Dict[str, float], np.ndarray]] = []
        requested = max(int(population_size), 1)
        for local_idx in range(requested):
            if generation == 1 and local_idx < len(seed_actions):
                action = seed_actions[local_idx].copy()
                action_source = "seed"
            elif local_idx == 0:
                action = action_mean.copy()
                action_source = "cem_mean"
            else:
                action = rng.normal(action_mean, action_sigma)
                action_source = "cem_sample"
            geom = build_full3d_initial_shape_from_action(
                cfg,
                action,
                target_volume=target_volume,
                use_neural_policy=bool(use_neural_policy),
                policy=policy,
                device=device,
                rng=rng,
            )
            geom = project_full3d_geometry(cfg, geom, target_volume)
            metrics = evaluate_full3d_geometry(cfg, geom, baseline_metrics)
            metrics["archive_index"] = int(candidate_count)
            metrics["generation"] = int(generation)
            metrics["action_dim_full3d"] = int(action_dim)
            metrics["action_source_full3d"] = action_source
            metrics["action_norm_full3d"] = float(np.linalg.norm(action))
            metrics["cem_sigma_mean_full3d"] = float(np.mean(action_sigma))
            metrics["initial_shape_generation_full3d"] = "global_strategy_field_from_cylinder_inventory"
            candidates.append((geom, metrics, action))
            archive_metrics.append(dict(metrics))
            candidate_count += 1
        candidate_pairs = [(geom, metrics) for geom, metrics, _ in candidates]
        ranked = sorted(candidates, key=lambda item: _full3d_metric_rank(item[1]), reverse=True)
        elite_count = max(1, int(math.ceil(len(ranked) * float(getattr(cfg, "full3d_cem_elite_fraction", 0.35)))))
        elite_actions = np.stack([item[2] for item in ranked[:elite_count]], axis=0)
        elite_mean = np.mean(elite_actions, axis=0)
        elite_sigma = np.std(elite_actions, axis=0) + float(getattr(cfg, "full3d_cem_min_sigma", 0.05))
        smoothing = float(np.clip(getattr(cfg, "full3d_cem_smoothing", 0.55), 0.0, 0.98))
        action_mean = smoothing * action_mean + (1.0 - smoothing) * elite_mean
        action_sigma = np.maximum(
            smoothing * action_sigma + (1.0 - smoothing) * elite_sigma,
            float(getattr(cfg, "full3d_cem_min_sigma", 0.05)),
        )
        generation_best = _select_full3d_candidate([(best, best_metrics), *candidate_pairs])
        if generation_best[1] is not best_metrics:
            best, best_metrics = generation_best[0], dict(generation_best[1])
        history_geometries.append(best)
        history_metrics.append(
            dict(
                best_metrics,
                generation=generation,
                step=len(history_geometries) - 1,
                action_dim_full3d=int(action_dim),
                cem_sigma_mean_full3d=float(np.mean(action_sigma)),
            )
        )

    selection_diagnostics = _full3d_selection_diagnostics(archive_metrics, best_metrics, baseline_metrics, cfg)

    return Full3DResult(
        best_geometry=best,
        baseline_geometry=baseline,
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        history_geometries=history_geometries,
        history_metrics=history_metrics,
        candidate_count=candidate_count,
        archive_metrics=archive_metrics,
        selection_diagnostics=selection_diagnostics,
    )


def export_full3d_mesh(
    geometry: Full3DGeometry,
    cfg,
    output_dir: str | Path,
    output_name: str = "optimized_full3d",
    export_step: bool = True,
) -> Dict[str, object]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stl_path = output / f"{output_name}.stl"
    stp_path = output / f"{output_name}.stp"
    mesh = trimesh.Trimesh(vertices=geometry.vertices, faces=geometry.faces, process=False)
    trimesh.repair.fix_normals(mesh, multibody=False)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    mesh.export(stl_path)
    stp = None
    if export_step:
        ok = _run_freecad_stl_to_step(
            str(stl_path),
            str(stp_path),
            freecad_cmd=getattr(cfg, "freecad_cmd", ""),
            timeout_s=float(getattr(cfg, "freecad_timeout_s", 90.0)),
        )
        stp = str(stp_path) if ok else None
    return {"stl": str(stl_path), "stp": stp, "watertight": bool(mesh.is_watertight)}


def export_full3d_animation(
    geometries: list[Full3DGeometry],
    metrics: list[Dict[str, float]],
    output_dir: str | Path,
    output_name: str = "topology_evolution_full3d",
    fps: int = 3,
) -> Dict[str, str | None]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, geom in enumerate(geometries):
        fig = plt.figure(figsize=(8, 5.0666666667), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        v = geom.vertices * 1.0e3
        tri = geom.faces
        ax.plot_trisurf(v[:, 0], v[:, 2], tri, v[:, 1], cmap="viridis", linewidth=0.05, alpha=0.94)
        ax.set_xlabel("x mm")
        ax.set_ylabel("z mm")
        ax.set_zlabel("y mm")
        ax.set_box_aspect((1, 2.2, 1))
        ax.view_init(elev=18, azim=-62)
        m = metrics[min(idx, len(metrics) - 1)] if metrics else {}
        fig.suptitle(
            f"step={idx} P0K={m.get('net_radiated_power_0k_sphere_w', 0.0):.2f}W "
            f"life={m.get('lifetime_ratio_3d', 1.0):.3f} vol={m.get('volume_change_ratio_3d', 0.0):.2e}",
            fontsize=10,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    gif_path = output / f"{output_name}.gif"
    mp4_path = output / f"{output_name}.mp4"
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


def save_full3d_history_csv(output_dir: str | Path, history: List[Dict[str, float]]) -> str:
    path = Path(output_dir) / "optimization_history_full3d.csv"
    keys = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    return str(path)


def save_full3d_summary(output_dir: str | Path, summary: Dict[str, object]) -> str:
    path = Path(output_dir) / "run_summary_full3d.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def write_full3d_report(output_dir: str | Path, result: Full3DResult, summary: Dict[str, object]) -> str:
    path = Path(output_dir) / "design_strategy_report_full3d.md"
    lines = [
        "# Full 3D Closed-Mesh Optimization Report",
        "",
        "## Physics Rules",
        "",
        "- Geometry is a closed 3D mesh; side wall, top face, and bottom face can all move.",
        "- The transverse electrode separation is fixed at 15 mm.",
        "- The two 5 mm circular electrode disks keep their diameter and relative position fixed, but the tungsten end-face footprint can change.",
        "- Pre-energization volume is projected back to the initial cylinder volume.",
        "- The tungsten voltage is applied only between the two tungsten/electrode contact footprints; electrode voltage drop is zero in this model.",
        "- Electrode cooling is coupled through the actual overlap area between each tungsten end face and the fixed 5 mm electrode disk.",
        "- End faces do not radiate or sublime, including both regions outside the electrode disk and electrode disk regions no longer touched by tungsten.",
        "- Only side/free surfaces radiate to the 300 K environment and sublime.",
        "- The specified 5 mm x 15 mm cylinder is the material-volume and lifetime baseline, not an assumed optimal initial shape.",
        "- The optimizer uses CEM over a Phy-DRL-LSM inspired global initial-shape strategy field; the optional neural policy supplies structured perturbations after candidate generation.",
        "- The action space emits radiation/evaporation/current/direct strategy maps over Chebyshev(z) x Fourier(theta), combines them with physics sensitivities into a normal speed, and applies a Lagrange volume-preserving projection.",
        "- Top/bottom cap footprints use Chebyshev(radius) x Fourier(theta) in-plane modes.",
        "",
        "## Selection",
        "",
        f"- Optimization target: `{summary.get('optimization_target_full3d', '')}`",
        f"- Voltage mode: `{summary.get('voltage_search_mode', '')}`",
        f"- Voltage constraint: `{summary.get('voltage_constraint', '')}`",
        f"- Final voltage: `{summary.get('final_voltage_v', 0.0):.6g} V`",
        f"- Tungsten voltage: `{summary.get('tungsten_voltage_v', summary.get('final_voltage_v', 0.0)):.6g} V`",
        f"- Electrode voltage drop: `{summary.get('electrode_voltage_drop_v', 0.0):.6g} V`",
        f"- Selected archive index: `{summary.get('selected_archive_index', 0)}`",
        f"- Feasible candidates: `{summary.get('archive_feasible_count', 0)} / {summary.get('archive_candidate_count', 0)}`",
        f"- Selection reason: `{summary.get('selection_reason_full3d', '')}`",
        "",
        "## Results",
        "",
        f"- 300K-environment 0-3 um net radiation: `{summary.get('final_net_radiated_power_300k_environment_w', summary.get('final_net_radiated_power_0k_sphere_w', 0.0)):.6g} W`",
        f"- 0-3 um energy-conversion efficiency: `{summary.get('final_energy_conversion_efficiency_0_3um', 0.0):.6g}`",
        f"- Energy-conversion efficiency ratio: `{summary.get('energy_conversion_efficiency_ratio', 0.0):.6g}`",
        f"- Power ratio: `{summary.get('power_ratio_full3d', 0.0):.6g}`",
        f"- Lifetime ratio: `{summary.get('lifetime_ratio_full3d', 0.0):.6g}`",
        f"- Max temperature: `{summary.get('max_temperature_k', 0.0):.6g} K`",
        f"- Temperature violation ratio: `{summary.get('temperature_violation_ratio', 0.0):.6g}`",
        f"- Thermal converged: `{summary.get('thermal_converged', False)}`",
        f"- Electrode contact area: `{summary.get('electrode_contact_area_m2', 0.0):.6g} m^2`",
        f"- Non-contact end-face area: `{summary.get('noncontact_end_face_area_m2', 0.0):.6g} m^2`",
        f"- Missing electrode contact area: `{summary.get('missing_electrode_contact_area_m2', 0.0):.6g} m^2`",
        f"- Volume error: `{summary.get('volume_change_ratio_full3d', 0.0):.6g}`",
        f"- Electrode max error: `{summary.get('electrode_max_error_m', 0.0):.6g} m`",
        f"- Feasible: `{summary.get('feasible', False)}`",
        "",
        "## Artifacts",
        "",
        f"- STL: `{summary.get('stl')}`",
        f"- STP: `{summary.get('stp')}`",
        f"- GIF: `{summary.get('gif')}`",
        f"- MP4: `{summary.get('mp4')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
