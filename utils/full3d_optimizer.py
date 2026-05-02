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
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from utils.exporter import _run_freecad_stl_to_step
from utils.rated_condition import _band_fraction_tensor, _evaporation_flux_kg_m2_s, _material_properties


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

    # cap ring 0 is the fixed 5 mm electrode boundary and reuses the side end ring.
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
    mask = np.ones(geometry.vertices.shape[0], dtype=bool)
    mask[geometry.lower_electrode_indices] = False
    mask[geometry.upper_electrode_indices] = False
    return mask


def _electrode_error(geometry: Full3DGeometry, cfg) -> float:
    radius = float(cfg.radius)
    height = float(cfg.height)
    ids = np.concatenate([geometry.lower_electrode_indices, geometry.upper_electrode_indices])
    points = geometry.vertices[ids]
    radial = np.linalg.norm(points[:, :2], axis=1)
    target_z = np.concatenate([
        np.full(len(geometry.lower_electrode_indices), -0.5 * height),
        np.full(len(geometry.upper_electrode_indices), 0.5 * height),
    ])
    return float(max(np.max(np.abs(radial - radius)), np.max(np.abs(points[:, 2] - target_z))))


def project_full3d_geometry(cfg, geometry: Full3DGeometry, target_volume: float) -> Full3DGeometry:
    vertices = geometry.vertices.copy()
    baseline = build_baseline_full3d_geometry(cfg)
    free = _free_vertex_mask(geometry)
    fixed = ~free
    vertices[fixed] = baseline.vertices[fixed]
    center = np.zeros(3, dtype=np.float64)
    max_delta = float(getattr(cfg, "feature_fail_ratio", 0.20)) * float(cfg.radius)
    delta = vertices[free] - baseline.vertices[free]
    delta_norm = np.linalg.norm(delta, axis=1)
    too_far = delta_norm > max_delta
    if np.any(too_far):
        delta[too_far] *= (max_delta / np.maximum(delta_norm[too_far], 1.0e-12))[:, None]
        vertices[free] = baseline.vertices[free] + delta
    for _ in range(6):
        trial = _clone_geometry(geometry, vertices)
        current_volume = mesh_volume(trial)
        scale = (float(target_volume) / max(current_volume, 1.0e-18)) ** (1.0 / 3.0)
        vertices[free] = center + (vertices[free] - center) * scale
        vertices[fixed] = baseline.vertices[fixed]
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


def evaluate_full3d_geometry(cfg, geometry: Full3DGeometry, baseline_metrics: Dict[str, float] | None = None) -> Dict[str, float]:
    target_volume = math.pi * float(cfg.radius) ** 2 * float(cfg.height)
    volume = mesh_volume(geometry)
    volume_change = abs(volume - target_volume) / max(target_volume, 1.0e-18)
    area, normals, centers = _mesh_area_normals_centers(geometry)
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    voltage = float(getattr(cfg, "full3d_fixed_voltage_v", 100.0))
    avg_radius = math.sqrt(max(volume, 1.0e-18) / (math.pi * float(cfg.height)))
    resistance = float(cfg.rho_elec_ref) * float(cfg.height) / max(math.pi * avg_radius * avg_radius, 1.0e-12)
    current = voltage / max(resistance, float(cfg.min_resistance))
    electrical_power = voltage * current
    surface_area = float(np.sum(area))
    cylinder_area = 2.0 * math.pi * float(cfg.radius) * float(cfg.height) + 2.0 * math.pi * float(cfg.radius) ** 2
    center_to_face = centers - np.mean(geometry.vertices, axis=0)
    escape = np.clip(
        np.sum(normals * center_to_face, axis=1)
        / np.maximum(np.linalg.norm(center_to_face, axis=1), 1.0e-12),
        float(getattr(cfg, "full3d_escape_floor", 0.10)),
        1.0,
    )
    effective_area = float(np.sum(area * escape))
    temperature = (electrical_power / max(float(cfg.band_emissivity) * float(cfg.stefan_boltzmann) * effective_area, 1.0e-12)) ** 0.25
    temperature = float(np.clip(temperature, float(cfg.ambient_temp), float(cfg.max_temp) * 1.5))
    temp_t = torch.full((1,), temperature, dtype=torch.float32, device=device)
    band_fraction = float(_band_fraction_tensor(cfg, temp_t, cfg.in_band_upper_um)[0].item())
    sphere_temp = float(getattr(cfg, "full3d_sphere_temperature_k", 0.0))
    net_band_power = (
        float(cfg.band_emissivity)
        * float(cfg.stefan_boltzmann)
        * effective_area
        * max(temperature ** 4 - sphere_temp ** 4, 0.0)
        * band_fraction
        * float(getattr(cfg, "full3d_sphere_emissivity", 1.0))
    )
    evap_flux = float(_evaporation_flux_kg_m2_s(cfg, temp_t)[0].item())
    mass_loss_rate = evap_flux * surface_area
    lifetime_s = float(cfg.feature_fail_ratio) * float(cfg.radius) / max(evap_flux / max(float(cfg.density), 1.0e-12), 1.0e-18)
    if baseline_metrics is None:
        baseline_life = lifetime_s
        baseline_power = max(net_band_power, 1.0e-9)
    else:
        baseline_life = max(float(baseline_metrics.get("lifetime_s", lifetime_s)), 1.0e-9)
        baseline_power = max(float(baseline_metrics.get("net_radiated_power_0k_sphere_w", net_band_power)), 1.0e-9)
    lifetime_ratio = lifetime_s / baseline_life
    electrode_error = _electrode_error(geometry, cfg)
    temperature_violation_ratio = max(temperature / max(float(cfg.max_temp), 1.0e-9) - 1.0, 0.0)
    feasible = (
        volume_change <= float(getattr(cfg, "full3d_volume_tolerance_ratio", 1.0e-5))
        and electrode_error <= float(getattr(cfg, "full3d_electrode_tolerance_m", 2.0e-6))
        and temperature <= float(cfg.max_temp)
        and lifetime_ratio >= float(cfg.minimum_lifetime_ratio)
    )
    score = (
        net_band_power / baseline_power
        + 0.25 * lifetime_ratio
        + 0.10 * (effective_area / max(cylinder_area, 1.0e-12))
        - 80.0 * volume_change
        - 1.0e5 * electrode_error
        - 60.0 * temperature_violation_ratio
        - 50.0 * max(float(cfg.minimum_lifetime_ratio) - lifetime_ratio, 0.0)
    )
    if not feasible:
        score -= 10.0
    return {
        "score": float(score),
        "voltage_v": voltage,
        "current_a": float(current),
        "resistance_ohm": float(resistance),
        "mean_temperature_k": temperature,
        "max_temperature_k": temperature,
        "net_radiated_power_0k_sphere_w": float(net_band_power),
        "effective_radiating_area_m2": float(effective_area),
        "surface_area_m2": float(surface_area),
        "surface_area_ratio": float(surface_area / max(cylinder_area, 1.0e-12)),
        "escape_view_factor_proxy": float(effective_area / max(surface_area, 1.0e-12)),
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
        "external_sphere_temperature_k": sphere_temp,
        "external_sphere_emissivity": float(getattr(cfg, "full3d_sphere_emissivity", 1.0)),
    }


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
    policy = Full3DUNetGNNPolicy().to(device)
    policy.eval()
    current = baseline
    best = baseline
    best_metrics = dict(baseline_metrics)
    history_geometries = [baseline]
    history_metrics = [dict(baseline_metrics, generation=0, step=0)]
    candidate_count = 1
    amplitude = float(getattr(cfg, "max_depth", 1.6e-4))

    for generation in range(1, int(generations) + 1):
        candidates: list[tuple[Full3DGeometry, Dict[str, float]]] = []
        for _ in range(int(population_size)):
            if bool(use_neural_policy):
                volume = _geometry_to_policy_volume(cfg, current, device)
                with torch.no_grad():
                    strategy = policy(volume)
            else:
                strategy = torch.rand((1, 3, 8, 16, 16), dtype=torch.float32, device=device)
            geom = _apply_strategy_displacement(cfg, current, strategy, amplitude * float(rng.uniform(0.4, 1.4)))
            geom = project_full3d_geometry(cfg, geom, target_volume)
            metrics = evaluate_full3d_geometry(cfg, geom, baseline_metrics)
            candidates.append((geom, metrics))
            candidate_count += 1
        geom, metrics = max(candidates, key=lambda item: item[1]["score"])
        current = geom
        if metrics["score"] > best_metrics["score"]:
            best = geom
            best_metrics = dict(metrics)
        history_geometries.append(best)
        history_metrics.append(dict(best_metrics, generation=generation, step=len(history_geometries) - 1))
        amplitude *= 0.82

    return Full3DResult(
        best_geometry=best,
        baseline_geometry=baseline,
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        history_geometries=history_geometries,
        history_metrics=history_metrics,
        candidate_count=candidate_count,
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
        fig = plt.figure(figsize=(8, 5), dpi=120)
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
        "# 真三维封闭几何优化报告",
        "",
        "## 物理规则",
        "",
        "- 几何是封闭三维网格，侧面、顶面、底面均可参与优化。",
        "- 两端 5mm 圆形电极边界保持直径与相对位置不变。",
        "- 通电前体积投影到初始圆柱体积。",
        "- 有效辐射按 0K、发射率 1 的外接球吸收面统计 0-3 微米净辐射。",
        "- 策略生成器包含轻量 3D U-Net 编码器和图邻域平滑头。",
        "",
        "## 结果",
        "",
        f"- 0K 外接球有效辐射功率：`{summary.get('final_net_radiated_power_0k_sphere_w', 0.0):.6g} W`",
        f"- 功率比：`{summary.get('power_ratio_full3d', 0.0):.6g}`",
        f"- 寿命比：`{summary.get('lifetime_ratio_full3d', 0.0):.6g}`",
        f"- 体积偏差：`{summary.get('volume_change_ratio_full3d', 0.0):.6g}`",
        f"- 电极最大误差：`{summary.get('electrode_max_error_m', 0.0):.6g} m`",
        f"- 温度违规比例：`{summary.get('temperature_violation_ratio', 0.0):.6g}`",
        f"- 可行：`{summary.get('feasible', False)}`",
        "",
        "## 产物",
        "",
        f"- STL：`{summary.get('stl')}`",
        f"- STP：`{summary.get('stp')}`",
        f"- GIF：`{summary.get('gif')}`",
        f"- MP4：`{summary.get('mp4')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
