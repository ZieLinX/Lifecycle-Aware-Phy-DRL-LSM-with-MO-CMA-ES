import math
import importlib
from importlib import util as importlib_util
from typing import Dict, List, Tuple

import numpy as np
import torch

def _safe_find_spec(module_name: str):
    try:
        return importlib_util.find_spec(module_name)
    except ModuleNotFoundError:
        return None


isaaclab_usd = importlib.import_module("isaaclab.usd") if _safe_find_spec("isaaclab.usd") else None
isaaclab_objects = (
    importlib.import_module("isaaclab.core.objects")
    if _safe_find_spec("isaaclab.core.objects")
    else None
)
pxr_usdgeom = importlib.import_module("pxr.UsdGeom") if _safe_find_spec("pxr.UsdGeom") else None


class CylinderPhysicsEnv:
    """
    Cylindrical thin-shell deformation environment.

    The model combines:
    1) spring energy (radial restoring force)
    2) bending-like smoothness energy (graph Laplacian)
    3) damping and semi-implicit Euler integration
    4) Helmholtz free-energy objective F = U - T*S
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.use_usd = False
        self.stage = None
        self.mesh = None
        self.current_step = 0

        # Try USD-backed simulation first; fallback to tensor-only mode.
        if isaaclab_usd is not None and isaaclab_objects is not None and pxr_usdgeom is not None:
            try:
                self.stage = isaaclab_usd.get_context().get_stage()
                self.cyl = isaaclab_objects.DynamicCylinder(
                    prim_path=cfg.prim_path, radius=cfg.radius, height=cfg.height
                )
                self.mesh = pxr_usdgeom.Mesh.Get(self.stage, cfg.prim_path)
                self.use_usd = self.mesh is not None
            except Exception:
                self.use_usd = False

        self.rest_points, self.neighbors = self._build_cylinder_lattice(
            radius=cfg.radius,
            height=cfg.height,
            num_segments=cfg.num_segments,
            num_rings=cfg.num_rings,
            device=self.device,
        )
        self.points = self.rest_points.clone()
        self.velocity = torch.zeros_like(self.points)
        self.dent_field = torch.zeros(self.points.shape[0], dtype=torch.float32, device=self.device)
        self.num_points = self.points.shape[0]
        self.obs_dim = self.num_points * 3
        self.neighbor_index, self.neighbor_mask = self._build_neighbor_tensor(self.neighbors, self.device)

    @staticmethod
    def _build_neighbor_tensor(neighbors: List[List[int]], device: torch.device):
        max_deg = max(len(nbrs) for nbrs in neighbors)
        n = len(neighbors)
        index = torch.zeros((n, max_deg), dtype=torch.long, device=device)
        mask = torch.zeros((n, max_deg), dtype=torch.float32, device=device)
        for i, nbrs in enumerate(neighbors):
            k = len(nbrs)
            index[i, :k] = torch.tensor(nbrs, dtype=torch.long, device=device)
            mask[i, :k] = 1.0
        return index, mask

    @staticmethod
    def _build_cylinder_lattice(
        radius: float,
        height: float,
        num_segments: int,
        num_rings: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """Create a regular cylindrical lattice and 4-neighborhood connectivity."""
        pts = []
        neighbors: List[List[int]] = [[] for _ in range(num_segments * num_rings)]
        z_values = torch.linspace(-0.5 * height, 0.5 * height, num_rings, device=device)
        theta_values = torch.linspace(0.0, 2.0 * math.pi, num_segments + 1, device=device)[:-1]

        for r_idx, z in enumerate(z_values):
            for s_idx, theta in enumerate(theta_values):
                x = radius * torch.cos(theta)
                y = radius * torch.sin(theta)
                pts.append([x.item(), y.item(), z.item()])

                cur = r_idx * num_segments + s_idx
                left = r_idx * num_segments + ((s_idx - 1) % num_segments)
                right = r_idx * num_segments + ((s_idx + 1) % num_segments)
                neighbors[cur].extend([left, right])
                if r_idx > 0:
                    neighbors[cur].append((r_idx - 1) * num_segments + s_idx)
                if r_idx < num_rings - 1:
                    neighbors[cur].append((r_idx + 1) * num_segments + s_idx)

        points = torch.tensor(pts, dtype=torch.float32, device=device)
        return points, neighbors

    def _laplacian_term(self, radial_disp: torch.Tensor) -> torch.Tensor:
        nbr_values = radial_disp[self.neighbor_index]
        nbr_mean = (nbr_values * self.neighbor_mask).sum(dim=1) / self.neighbor_mask.sum(dim=1).clamp_min(1.0)
        return radial_disp - nbr_mean

    def _sync_from_usd(self):
        if not self.use_usd:
            return
        points_usd = self.mesh.GetPointsAttr().Get()
        if points_usd is not None:
            self.points = torch.tensor(np.array(points_usd), dtype=torch.float32, device=self.device)

    def _sync_to_usd(self):
        if self.use_usd:
            self.mesh.GetPointsAttr().Set(self.points.detach().cpu().numpy())

    def reset(self):
        self.current_step = 0
        self.points = self.rest_points.clone()
        self.velocity = torch.zeros_like(self.points)
        self.dent_field = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
        self._sync_to_usd()
        return self.points.flatten().detach().cpu().numpy()

    def get_state(self):
        """Return a detached snapshot for action lookahead."""
        return {
            "points": self.points.clone(),
            "velocity": self.velocity.clone(),
            "dent_field": self.dent_field.clone(),
            "current_step": self.current_step,
        }

    def set_state(self, state):
        """Restore environment snapshot."""
        self.points = state["points"].clone()
        self.velocity = state["velocity"].clone()
        self.dent_field = state["dent_field"].clone()
        self.current_step = int(state["current_step"])
        self._sync_to_usd()

    def evaluate_action(self, actions):
        """
        One-step lookahead:
        simulate action and restore state afterwards.
        """
        snapshot = self.get_state()
        _, reward, done, info = self.step(actions)
        self.set_state(snapshot)
        return reward, done, info

    def compute_free_energy(self, points: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        radial = torch.norm(points[:, :2], dim=1)
        radial_disp = radial - self.cfg.radius

        # U_spring: isotropic radial restoring energy
        u_spring = 0.5 * self.cfg.k_spring * torch.sum(radial_disp.pow(2))

        # U_bend: smoothness penalty from graph Laplacian
        lap = self._laplacian_term(radial_disp)
        u_bend = 0.5 * self.cfg.k_bend * torch.sum(lap.pow(2))

        # Kinetic term
        kinetic = 0.5 * self.cfg.mass * torch.sum(velocity.pow(2))

        # Entropy proxy: variance of local displacement field
        entropy = torch.var(radial_disp)

        return (u_spring + u_bend + kinetic) - (self.cfg.temp_k * entropy)

    def step(self, actions):
        self.current_step += 1
        self._sync_from_usd()

        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device).flatten()
        if a.numel() < 3:
            raise ValueError("actions must contain [index_ratio, indentation, sigma]")

        idx_ratio = float(torch.clamp(a[0], 0.0, 1.0).item())
        depth = float(torch.clamp(a[1], 0.0, 1.0).item()) * self.cfg.max_depth
        sigma = float(torch.clamp(torch.abs(a[2]), self.cfg.min_sigma, self.cfg.max_sigma).item())

        target_idx = int(idx_ratio * (self.num_points - 1))
        target_point = self.points[target_idx]

        # Gaussian spatial profile around target vertex.
        dist = torch.norm(self.points - target_point, dim=1)
        influence = torch.exp(-(dist.pow(2)) / (2.0 * sigma * sigma))

        xy = self.points[:, :2]
        radial_norm = torch.norm(xy, dim=1, keepdim=True).clamp_min(1e-6)
        radial_dir = xy / radial_norm

        # Persistent dents: each action adds a local inward target displacement
        # and multiple dents can coexist at different discrete lattice vertices.
        local_dent = -depth * influence
        self.dent_field = self.dent_field * (1.0 - self.cfg.dent_decay) + local_dent
        self.dent_field = torch.clamp(self.dent_field, -self.cfg.max_total_dent, 0.0)

        # External force nudges points towards newly created local dents.
        f_ext_xy = -radial_dir * depth * influence.unsqueeze(1) * self.cfg.k_input
        f_ext = torch.cat([f_ext_xy, torch.zeros(self.num_points, 1, device=self.device)], dim=1)

        radial = torch.norm(self.points[:, :2], dim=1)
        radial_disp = radial - self.cfg.radius
        lap = self._laplacian_term(radial_disp)

        # Restoring force is measured against the accumulated dent target field.
        radial_error = radial_disp - self.dent_field
        f_spring_xy = -radial_dir * (self.cfg.k_spring * radial_error).unsqueeze(1)
        f_bend_xy = -radial_dir * (self.cfg.k_bend * lap).unsqueeze(1)
        f_internal = torch.cat([f_spring_xy + f_bend_xy, torch.zeros(self.num_points, 1, device=self.device)], dim=1)

        # Damping + integration.
        force_total = f_ext + f_internal - self.cfg.damping * self.velocity
        acc = force_total / self.cfg.mass
        self.velocity = self.velocity + acc * self.cfg.dt
        self.points = self.points + self.velocity * self.cfg.dt

        # Keep minimum radius to avoid collapsing through axis.
        xy = self.points[:, :2]
        r = torch.norm(xy, dim=1, keepdim=True).clamp_min(1e-6)
        collapsed = r < self.cfg.min_radius
        if collapsed.any():
            self.points[:, :2] = torch.where(collapsed, xy / r * self.cfg.min_radius, xy)
            self.velocity[:, :2] *= (~collapsed).float()

        self._sync_to_usd()

        free_energy = self.compute_free_energy(self.points, self.velocity)
        reward = -free_energy.item()
        done = self.current_step >= self.cfg.max_steps
        info: Dict[str, float] = {
            "free_energy": float(free_energy.item()),
            "mean_radius": float(torch.norm(self.points[:, :2], dim=1).mean().item()),
            "active_dent_points": float((torch.abs(self.dent_field) > self.cfg.dent_active_threshold).sum().item()),
            "max_dent_depth": float(torch.abs(self.dent_field).max().item()),
            "step": float(self.current_step),
        }
        return self.points.flatten().detach().cpu().numpy(), reward, done, info