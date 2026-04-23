import math
import importlib
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.feasibility import project_connected_profile
from utils.rated_condition import (
    RatedConditionMetrics,
    ring_radii_from_points,
    search_rated_condition,
    simulate_transient_trajectory,
)


class CylinderPhysicsEnv:
    """
    Cylindrical thin-shell deformation environment.

    Coupled model:
    1) mechanical deformation on cylindrical lattice
    2) electro-thermal update with Joule heating + radiation + conduction
    3) evaporation-driven mass loss and surface recession
    4) constrained objective balancing radiation performance and lifetime
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.use_usd = False
        self.stage = None
        self.mesh = None
        self._UsdGeom = None
        self._Sdf = None
        self._Gf = None
        self._UsdLux = None
        self._UsdShade = None
        self._suppress_usd_sync = False
        self.current_step = 0

        # Optional USD backend: lazy import to avoid SimulationApp import-order issues.
        if getattr(cfg, "use_usd_backend", False):
            try:
                try:
                    usd_ctx_module = importlib.import_module("isaaclab.usd")
                    context = usd_ctx_module.get_context()
                except Exception:
                    usd_ctx_module = importlib.import_module("omni.usd")
                    context = usd_ctx_module.get_context()
                viewport_stage = None
                try:
                    viewport_util = importlib.import_module("omni.kit.viewport.utility")
                    viewport_api = viewport_util.get_active_viewport()
                    if viewport_api is not None:
                        viewport_stage = viewport_api.stage
                except Exception:
                    viewport_stage = None
                self._UsdGeom = importlib.import_module("pxr.UsdGeom")
                self._Sdf = importlib.import_module("pxr.Sdf")
                self._Gf = importlib.import_module("pxr.Gf")
                self._UsdLux = importlib.import_module("pxr.UsdLux")
                self._UsdShade = importlib.import_module("pxr.UsdShade")
                self.stage = viewport_stage if viewport_stage is not None else context.get_stage()
                if self.stage is None:
                    context.new_stage()
                    self.stage = context.get_stage()
                if self.stage is not None:
                    root = self.stage.GetRootLayer()
                    root_id = root.identifier if root is not None else "<no-root-layer>"
                    print(f"[viewer] using stage root: {root_id}", flush=True)
            except Exception as exc:
                print(f"[viewer] USD backend init failed: {exc}", flush=True)
                self.stage = None
                self._UsdGeom = None
                self._Sdf = None
                self._Gf = None
                self._UsdLux = None
                self._UsdShade = None

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
        self.bulge_field = torch.zeros(self.points.shape[0], dtype=torch.float32, device=self.device)
        self.temperature = torch.full(
            (self.points.shape[0],),
            float(self.cfg.ambient_temp),
            dtype=torch.float32,
            device=self.device,
        )
        self.ablation_depth = torch.zeros_like(self.temperature)
        self.num_points = self.points.shape[0]
        self.neighbor_index, self.neighbor_mask = self._build_neighbor_tensor(self.neighbors, self.device)
        self.ring_index = self._build_ring_index(cfg.num_segments, cfg.num_rings, self.device)
        self.area_per_point = (2.0 * math.pi * cfg.radius * cfg.height) / self.num_points
        self.dx = min(
            2.0 * math.pi * cfg.radius / max(cfg.num_segments, 1),
            cfg.height / max(cfg.num_rings - 1, 1),
        )
        self.initial_volume = math.pi * cfg.radius * cfg.radius * cfg.height
        self.initial_mass = self.initial_volume * cfg.density
        self.remaining_mass = self.initial_mass
        self.baseline_metrics: RatedConditionMetrics | None = None
        self.current_metrics: RatedConditionMetrics | None = None
        self.last_score = 0.0
        self.best_score = -float("inf")
        self.obs_dim = 0
        if self.stage is not None and self._UsdGeom is not None and self._Sdf is not None:
            try:
                self._ensure_default_lights()
                self.mesh = self._init_usd_mesh(self._UsdGeom, self._Sdf)
                self.use_usd = self.mesh is not None
            except Exception as exc:
                print(f"[viewer] USD mesh creation failed: {exc}", flush=True)
                self.use_usd = False

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
    def _build_ring_index(num_segments: int, num_rings: int, device: torch.device) -> torch.Tensor:
        idx = []
        for r_idx in range(num_rings):
            for _ in range(num_segments):
                idx.append(r_idx)
        return torch.tensor(idx, dtype=torch.long, device=device)

    @staticmethod
    def _build_surface_faces(num_segments: int, num_rings: int) -> np.ndarray:
        faces = []
        for r_idx in range(num_rings - 1):
            ring0 = r_idx * num_segments
            ring1 = (r_idx + 1) * num_segments
            for s_idx in range(num_segments):
                s_next = (s_idx + 1) % num_segments
                i0 = ring0 + s_idx
                i1 = ring0 + s_next
                i2 = ring1 + s_idx
                i3 = ring1 + s_next
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])
        return np.asarray(faces, dtype=np.int64)

    def _init_usd_mesh(self, UsdGeom, Sdf):
        UsdGeom.Xform.Define(self.stage, "/World")
        old_prim = self.stage.GetPrimAtPath(self.cfg.prim_path)
        if old_prim and old_prim.IsValid():
            self.stage.RemovePrim(self.cfg.prim_path)
        mesh = UsdGeom.Mesh.Define(self.stage, self.cfg.prim_path)
        faces = self._build_surface_faces(self.cfg.num_segments, self.cfg.num_rings)
        points_np = self._to_viewer_points(self.rest_points.detach().cpu().numpy())
        mesh.GetFaceVertexCountsAttr().Set([3] * int(faces.shape[0]))
        mesh.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
        mesh.GetPointsAttr().Set(points_np)
        pmin = points_np.min(axis=0).tolist()
        pmax = points_np.max(axis=0).tolist()
        mesh.CreateExtentAttr().Set([pmin, pmax])
        mesh.CreateDoubleSidedAttr().Set(True)
        mesh.GetSubdivisionSchemeAttr().Set(getattr(self.cfg, "viewer_subdivision_scheme", "catmullClark"))
        mesh.GetVisibilityAttr().Set("inherited")
        prim = mesh.GetPrim()
        if prim:
            prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([(0.95, 0.95, 0.98)])
            prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([1.0])
            self._bind_visible_material(prim)
        print(
            f"[viewer] USD mesh ready at {self.cfg.prim_path} "
            f"(verts={points_np.shape[0]}, faces={faces.shape[0]}, scale={getattr(self.cfg, 'viewer_scale', 1.0)})",
            flush=True,
        )
        return mesh

    def _to_viewer_points(self, points_np: np.ndarray) -> np.ndarray:
        s = float(getattr(self.cfg, "viewer_scale", 1.0))
        lift = float(getattr(self.cfg, "viewer_lift_z", 0.0))
        out = points_np.copy()
        out *= s
        out[:, 2] += lift
        return out

    def _set_light_shadow_safe(self, light) -> None:
        """Enable shadow in a version-compatible way (no hard failure)."""
        try:
            if hasattr(light, "CreateShadowEnableAttr"):
                light.CreateShadowEnableAttr(True)
                return
        except Exception:
            pass

        prim = light.GetPrim() if light is not None else None
        if prim is None or self._Sdf is None:
            return
        for attr_name in ("shadow:enable", "inputs:shadow:enable"):
            try:
                attr = prim.GetAttribute(attr_name)
                if not attr:
                    attr = prim.CreateAttribute(attr_name, self._Sdf.ValueTypeNames.Bool)
                if attr:
                    attr.Set(True)
                    return
            except Exception:
                continue

    def _ensure_default_lights(self):
        if self._UsdLux is None:
            return
        UsdGeom = self._UsdGeom
        if not self.stage.GetPrimAtPath("/World/KeyLight"):
            key = self._UsdLux.DistantLight.Define(self.stage, "/World/KeyLight")
            key.CreateIntensityAttr(4200.0)
            key.CreateColorAttr((1.0, 1.0, 1.0))
            key.CreateAngleAttr(0.35)
            self._set_light_shadow_safe(key)
            key_xf = UsdGeom.Xformable(key.GetPrim())
            key_xf.ClearXformOpOrder()
            key_xf.AddRotateXYZOp().Set((-35.0, 35.0, 0.0))
        if not self.stage.GetPrimAtPath("/World/FillLight"):
            fill = self._UsdLux.DistantLight.Define(self.stage, "/World/FillLight")
            fill.CreateIntensityAttr(650.0)
            fill.CreateColorAttr((0.88, 0.9, 1.0))
            fill.CreateAngleAttr(0.6)
            self._set_light_shadow_safe(fill)
            fill_xf = UsdGeom.Xformable(fill.GetPrim())
            fill_xf.ClearXformOpOrder()
            fill_xf.AddRotateXYZOp().Set((-20.0, -55.0, 0.0))
        if not self.stage.GetPrimAtPath("/World/SideLight"):
            side = self._UsdLux.DistantLight.Define(self.stage, "/World/SideLight")
            side.CreateIntensityAttr(1600.0)
            side.CreateColorAttr((1.0, 0.97, 0.93))
            side.CreateAngleAttr(0.3)
            self._set_light_shadow_safe(side)
            side_xf = UsdGeom.Xformable(side.GetPrim())
            side_xf.ClearXformOpOrder()
            side_xf.AddRotateXYZOp().Set((8.0, 120.0, 0.0))
        if not self.stage.GetPrimAtPath("/World/RimLight"):
            rim = self._UsdLux.DistantLight.Define(self.stage, "/World/RimLight")
            rim.CreateIntensityAttr(900.0)
            rim.CreateColorAttr((0.95, 0.97, 1.0))
            rim.CreateAngleAttr(0.22)
            self._set_light_shadow_safe(rim)
            rim_xf = UsdGeom.Xformable(rim.GetPrim())
            rim_xf.ClearXformOpOrder()
            rim_xf.AddRotateXYZOp().Set((15.0, 165.0, 0.0))
        if not self.stage.GetPrimAtPath("/World/DomeLight"):
            dome = self._UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
            dome.CreateIntensityAttr(35.0)
            dome.CreateColorAttr((1.0, 1.0, 1.0))

    def _bind_visible_material(self, prim):
        if self._UsdShade is None or self._Sdf is None:
            return
        mat_path = "/World/VisibleMaterial"
        material = self._UsdShade.Material.Define(self.stage, mat_path)
        shader = self._UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set((0.72, 0.72, 0.74))
        shader.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(0.46)
        shader.CreateInput("metallic", self._Sdf.ValueTypeNames.Float).Set(0.88)
        shader.CreateInput("emissiveColor", self._Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        shader.CreateInput("opacity", self._Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("opacityThreshold", self._Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self._UsdShade.MaterialBindingAPI(prim).Bind(material)

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

    def _laplacian_scalar(self, values: torch.Tensor) -> torch.Tensor:
        nbr_values = values[self.neighbor_index]
        nbr_mean = (nbr_values * self.neighbor_mask).sum(dim=1) / self.neighbor_mask.sum(dim=1).clamp_min(1.0)
        return nbr_mean - values

    def _cylindrical_surface_distance(self, target_idx: int) -> torch.Tensor:
        """
        Surface distance on a cylinder using unwrapped coordinates:
        s = sqrt((R * dtheta)^2 + dz^2).
        """
        theta = torch.atan2(self.points[:, 1], self.points[:, 0])
        theta0 = theta[target_idx]
        dtheta = torch.atan2(torch.sin(theta - theta0), torch.cos(theta - theta0))
        arc = self.cfg.radius * torch.abs(dtheta)
        dz = self.points[:, 2] - self.points[target_idx, 2]
        return torch.sqrt(arc * arc + dz * dz)

    def _apply_shape_projection(self):
        """Project current radial shape to the dent target for strict hemispherical profile."""
        if not getattr(self.cfg, "use_strict_shape_projection", False):
            return
        alpha = float(np.clip(getattr(self.cfg, "shape_projection_alpha", 1.0), 0.0, 1.0))
        if alpha <= 0.0:
            return

        xy = self.points[:, :2]
        radial = torch.norm(xy, dim=1, keepdim=True).clamp_min(1e-6)
        radial_dir = xy / radial
        target_radius = torch.clamp(
            self.cfg.radius + self.dent_field.unsqueeze(1),
            min=self.cfg.min_radius,
        )
        projected_radius = radial + alpha * (target_radius - radial)
        self.points[:, :2] = radial_dir * projected_radius

        # Remove projected radial velocity component to avoid oscillating away from target.
        vel_xy = self.velocity[:, :2]
        radial_vel = torch.sum(vel_xy * radial_dir, dim=1, keepdim=True)
        self.velocity[:, :2] = vel_xy - alpha * radial_vel * radial_dir

    def _build_observation(self) -> np.ndarray:
        radial = torch.norm(self.points[:, :2], dim=1)
        radial_disp = radial - self.cfg.radius
        temp_norm = (self.temperature - self.cfg.ambient_temp) / max(self.cfg.max_temp - self.cfg.ambient_temp, 1.0)
        ablation_norm = self.ablation_depth / max(self.cfg.feature_fail_ratio * self.cfg.radius, 1.0e-12)
        metrics = self.current_metrics
        if metrics is None:
            global_obs = torch.zeros(6, dtype=torch.float32, device=self.device)
        else:
            life_ratio = 1.0
            if self.baseline_metrics is not None:
                life_ratio = metrics.lifetime_s / max(self.baseline_metrics.lifetime_s, 1.0e-9)
            global_obs = torch.tensor(
                [
                    metrics.voltage_v / max(self.cfg.max_voltage, 1.0),
                    metrics.initial_net_band_power_w / max(
                        self.baseline_metrics.initial_net_band_power_w if self.baseline_metrics is not None else 1.0,
                        1.0e-9,
                    ),
                    metrics.average_net_band_power_w / max(
                        self.baseline_metrics.average_net_band_power_w if self.baseline_metrics is not None else 1.0,
                        1.0e-9,
                    ),
                    metrics.max_temperature_k / max(self.cfg.max_temp, 1.0),
                    life_ratio,
                    metrics.view_factor_proxy,
                ],
                dtype=torch.float32,
                device=self.device,
            )
        obs = torch.cat(
            [
                self.points.flatten(),
                radial_disp.flatten(),
                temp_norm.flatten(),
                ablation_norm.flatten(),
                global_obs,
            ]
        )
        return obs.detach().cpu().numpy()

    def _material_properties(self, temperature: torch.Tensor):
        delta = torch.clamp(temperature - 300.0, min=0.0)
        cp = self.cfg.cp_ref + self.cfg.cp_temp_coeff * delta
        k = torch.clamp(self.cfg.k_ref + self.cfg.k_temp_coeff * delta, min=20.0)
        rho_elec = self.cfg.rho_elec_ref * (1.0 + self.cfg.rho_elec_temp_coeff * delta)
        emissivity = torch.full_like(temperature, float(self.cfg.band_emissivity))
        return cp, k, rho_elec, emissivity

    def _evaporation_flux_kg_m2_s(self, temperature: torch.Tensor) -> torch.Tensor:
        # Input law in g/(cm^2*s); convert to kg/(m^2*s) by factor 10.
        y_g_cm2_s = self.cfg.evap_A * torch.exp(self.cfg.evap_B / torch.clamp(temperature, min=1.0))
        return y_g_cm2_s * 10.0

    def _surface_area_to_volume_ratio(self, radius: torch.Tensor) -> torch.Tensor:
        r = torch.clamp(radius, min=self.cfg.min_radius)
        return 2.0 / r

    def _apply_electrode_constraints(self):
        if not self.cfg.keep_electrode_rings_fixed:
            return
        first_ring = self.ring_index == 0
        last_ring = self.ring_index == int(torch.max(self.ring_index).item())
        locked = first_ring | last_ring
        self.points[locked] = self.rest_points[locked]
        self.velocity[locked] = 0.0
        self.ablation_depth[locked] = 0.0
        self.dent_field[locked] = 0.0
        self.bulge_field[locked] = 0.0

    def _sync_from_usd(self):
        if not self.use_usd or self._suppress_usd_sync:
            return
        # Keep physics state as source of truth; avoid pulling back transformed viewer points.
        return

    def _sync_to_usd(self):
        if self.use_usd and not self._suppress_usd_sync:
            self.mesh.GetPointsAttr().Set(self._to_viewer_points(self.points.detach().cpu().numpy()))

    def reset(self):
        self.current_step = 0
        self.points = self.rest_points.clone()
        self.velocity = torch.zeros_like(self.points)
        self.dent_field = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
        self.bulge_field = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
        self.temperature = torch.full(
            (self.num_points,),
            float(self.cfg.ambient_temp),
            dtype=torch.float32,
            device=self.device,
        )
        self.ablation_depth = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
        self.remaining_mass = self.initial_mass
        self.current_metrics = self._evaluate_geometry()
        self.baseline_metrics = self.current_metrics
        self.last_score = self._score_metrics(self.current_metrics)
        self.best_score = self.last_score
        self._update_pointwise_fields_from_metrics(self.current_metrics)
        self.obs_dim = int(self._build_observation().shape[0])
        self._sync_to_usd()
        return self._build_observation()

    def get_state(self):
        """Return a detached snapshot for action lookahead."""
        return {
            "points": self.points.clone(),
            "velocity": self.velocity.clone(),
            "dent_field": self.dent_field.clone(),
            "bulge_field": self.bulge_field.clone(),
            "temperature": self.temperature.clone(),
            "ablation_depth": self.ablation_depth.clone(),
            "remaining_mass": float(self.remaining_mass),
            "current_step": self.current_step,
            "current_metrics": self.current_metrics,
            "last_score": float(self.last_score),
            "best_score": float(self.best_score),
        }

    def set_state(self, state):
        """Restore environment snapshot."""
        self.points = state["points"].clone()
        self.velocity = state["velocity"].clone()
        self.dent_field = state["dent_field"].clone()
        bulge = state.get("bulge_field")
        if bulge is None:
            self.bulge_field = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
        else:
            self.bulge_field = bulge.clone()
        self.temperature = state["temperature"].clone()
        self.ablation_depth = state["ablation_depth"].clone()
        self.remaining_mass = float(state["remaining_mass"])
        self.current_step = int(state["current_step"])
        self.current_metrics = state.get("current_metrics")
        self.last_score = float(state.get("last_score", 0.0))
        self.best_score = float(state.get("best_score", self.last_score))
        self._sync_to_usd()

    def evaluate_action(self, actions):
        """
        One-step lookahead:
        simulate action and restore state afterwards.
        """
        snapshot = self.get_state()
        self._suppress_usd_sync = True
        try:
            _, reward, done, info = self.step(actions)
            self.set_state(snapshot)
        finally:
            self._suppress_usd_sync = False
        return reward, done, info

    def compute_free_energy(self, points: torch.Tensor, velocity: torch.Tensor | None = None) -> float:
        radial = torch.norm(points[:, :2], dim=1)
        radial_disp = radial - self.cfg.radius
        u_spring = 0.5 * self.cfg.k_spring * torch.sum(radial_disp.pow(2))
        lap = self._laplacian_term(radial_disp)
        u_bend = 0.5 * self.cfg.k_bend * torch.sum(lap.pow(2))
        return float((u_spring + u_bend).item())

    def _ring_radius_profile(self, points: torch.Tensor | None = None) -> torch.Tensor:
        pts = self.points if points is None else points
        return ring_radii_from_points(pts, self.ring_index, self.cfg.num_rings)

    def _compute_volume(self, points: torch.Tensor) -> float:
        ring_radius = self._ring_radius_profile(points)
        dz = self.cfg.height / max(self.cfg.num_rings - 1, 1)
        area = math.pi * ring_radius.pow(2)
        if area.numel() <= 1:
            return float(area[0].item() * self.cfg.height)
        axial_weight = torch.full_like(area, dz)
        axial_weight[0] *= 0.5
        axial_weight[-1] *= 0.5
        return float(torch.sum(area * axial_weight).item())

    def _compute_feature_change_ratio(self, points: torch.Tensor) -> float:
        ring_radius = self._ring_radius_profile(points)
        radius_ratio = torch.max(torch.abs(ring_radius - self.cfg.radius) / max(self.cfg.radius, 1e-12))
        return float(radius_ratio.item())

    def _update_pointwise_fields_from_metrics(self, metrics: RatedConditionMetrics) -> None:
        self.temperature = metrics.ring_temperature_k[self.ring_index].clone()
        self.ablation_depth = (
            metrics.ring_recession_rate_m_s[self.ring_index] * self.cfg.ablation_observation_horizon_s
        ).clone()
        self.remaining_mass = max(
            self.initial_mass - metrics.mass_loss_rate_kg_s * self.cfg.ablation_observation_horizon_s,
            0.0,
        )

    def _evaluate_geometry(self) -> RatedConditionMetrics:
        ring_radius = self._ring_radius_profile(self.points)
        return search_rated_condition(self.cfg, ring_radius, self.initial_volume)

    def _evaluate_transient_window(self, metrics: RatedConditionMetrics, dwell_norm: float) -> Dict[str, float]:
        dwell_time_s = float(np.clip(dwell_norm, 0.0, 1.0)) * float(self.cfg.lifecycle_reference_s)
        if dwell_time_s <= 0.0:
            return {
                "dwell_time_s": 0.0,
                "transient_power_w": 0.0,
                "transient_mean_power_w": 0.0,
                "transient_peak_temp_k": float(self.cfg.ambient_temp),
                "transient_mass_loss_kg": 0.0,
                "transient_power_ratio": 0.0,
            }

        ring_radius = self._ring_radius_profile(self.points)
        transient = simulate_transient_trajectory(
            cfg=self.cfg,
            ring_radius=ring_radius,
            voltage_schedule=float(metrics.voltage_v) * float(getattr(self.cfg, "transient_default_voltage_ratio", 1.0)),
            t_max=dwell_time_s,
            dt=float(self.cfg.transient_dt_s),
        )
        band_power = transient["band_power_w"]
        temp_hist = transient["temperature_k"]
        mass_hist = transient["mass_loss_kg"]
        transient_power_w = float(band_power[-1].item()) if band_power.numel() > 0 else 0.0
        transient_mean_power_w = float(torch.mean(band_power).item()) if band_power.numel() > 0 else 0.0
        transient_peak_temp_k = float(torch.max(temp_hist).item())
        transient_mass_loss_kg = float(mass_hist[-1].item()) if mass_hist.numel() > 0 else 0.0
        baseline_power = max(
            float((self.baseline_metrics or metrics).initial_net_band_power_w),
            1.0e-9,
        )
        return {
            "dwell_time_s": dwell_time_s,
            "transient_power_w": transient_power_w,
            "transient_mean_power_w": transient_mean_power_w,
            "transient_peak_temp_k": transient_peak_temp_k,
            "transient_mass_loss_kg": transient_mass_loss_kg,
            "transient_power_ratio": transient_power_w / baseline_power,
        }

    def _score_metrics(self, metrics: RatedConditionMetrics) -> float:
        baseline = self.baseline_metrics or metrics
        initial_ratio = metrics.initial_net_band_power_w / max(baseline.initial_net_band_power_w, 1.0e-9)
        average_ratio = metrics.average_net_band_power_w / max(baseline.average_net_band_power_w, 1.0e-9)
        lifetime_ratio = metrics.lifetime_s / max(baseline.lifetime_s, 1.0e-9)
        score = (
            self.cfg.reward_weight_initial_power * initial_ratio
            + self.cfg.reward_weight_average_power * average_ratio
            + self.cfg.reward_weight_lifetime * lifetime_ratio
            + self.cfg.reward_weight_uniformity * metrics.temperature_uniformity
            + self.cfg.reward_weight_efficiency * metrics.band_efficiency
            - self.cfg.reward_penalty_feasibility * metrics.feasibility_penalty
            - self.cfg.reward_weight_thermomech * metrics.thermo_mech_penalty
        )
        score -= self.cfg.reward_penalty_mass_loss * max(metrics.mass_loss_rate_kg_s - self.cfg.max_mass_loss_rate, 0.0)
        score -= self.cfg.reward_penalty_temp_violation * max(metrics.max_temperature_k - self.cfg.max_temp, 0.0) ** 2
        score -= self.cfg.reward_penalty_feature_violation * max(
            metrics.feature_change_ratio - self.cfg.feature_fail_ratio,
            0.0,
        )
        score -= self.cfg.reward_penalty_volume_change * metrics.volume_change_ratio
        if lifetime_ratio < self.cfg.minimum_lifetime_ratio:
            score -= 10.0 * (self.cfg.minimum_lifetime_ratio - lifetime_ratio)
        if not metrics.feasible:
            score -= 25.0
        return float(score)

    def _current_movable_mask(self) -> torch.Tensor:
        if not self.cfg.keep_electrode_rings_fixed:
            return torch.ones(self.num_points, dtype=torch.bool, device=self.device)
        first_ring = self.ring_index == 0
        last_ring = self.ring_index == int(torch.max(self.ring_index).item())
        return ~(first_ring | last_ring)

    def _enforce_volume_conservation(self, movable_mask: torch.Tensor) -> None:
        current_volume = self._compute_volume(self.points)
        missing_volume = self.initial_volume - current_volume
        if abs(missing_volume) / max(self.initial_volume, 1.0e-12) < 1.0e-6:
            return
        correction = float(
            np.clip(
                missing_volume / max(self.area_per_point * float(torch.sum(movable_mask.float()).item()), 1.0e-12),
                -0.25 * self.cfg.max_depth,
                0.25 * self.cfg.max_depth,
            )
        )
        xy = self.points[:, :2]
        radial = torch.norm(xy, dim=1, keepdim=True).clamp_min(1.0e-9)
        radial_dir = xy / radial
        corrected_radius = torch.clamp(
            radial.squeeze(1) + correction * movable_mask.float(),
            min=self.cfg.min_radius,
        )
        self.points[:, :2] = radial_dir * corrected_radius.unsqueeze(1)

    def _apply_design_action(self, target_idx: int, depth: float, sigma: float) -> None:
        movable_mask = self._current_movable_mask()
        distance = self._cylindrical_surface_distance(target_idx)
        sigma = max(float(sigma), self.cfg.min_sigma)
        local_profile = torch.exp(-0.5 * (distance / sigma).pow(2)) * movable_mask.float()
        local_profile[target_idx] = 1.0 if movable_mask[target_idx] else 0.0
        local_dent = -float(depth) * local_profile

        compensation_mask = movable_mask & (distance > self.cfg.compensation_exclusion_sigma * sigma)
        if not torch.any(compensation_mask):
            compensation_mask = movable_mask.clone()
            compensation_mask[target_idx] = False
        compensation_weights = compensation_mask.float()
        if self.current_metrics is not None:
            temp_norm = torch.clamp(
                (self.temperature - self.cfg.ambient_temp) / max(self.cfg.max_temp - self.cfg.ambient_temp, 1.0),
                0.0,
                1.0,
            )
            compensation_weights = compensation_weights * (
                1.0 + self.cfg.compensation_cool_bias * (1.0 - temp_norm)
            )
        compensation_weights = compensation_weights * (1.0 + distance / max(float(torch.max(distance).item()), 1.0e-12))
        if float(torch.sum(compensation_weights).item()) > 0.0:
            compensation = (-torch.sum(local_dent) / torch.sum(compensation_weights)) * compensation_weights
        else:
            compensation = torch.zeros_like(local_dent)
        radial_delta = torch.clamp(
            local_dent + compensation,
            min=-self.cfg.max_depth,
            max=self.cfg.max_depth,
        )

        xy = self.points[:, :2]
        radial = torch.norm(xy, dim=1, keepdim=True).clamp_min(1.0e-9)
        radial_dir = xy / radial
        new_radius = torch.clamp(radial.squeeze(1) + radial_delta, min=self.cfg.min_radius)
        self.points[:, :2] = radial_dir * new_radius.unsqueeze(1)
        self._enforce_volume_conservation(movable_mask)
        self._apply_electrode_constraints()
        # Project ring-level radii onto connected profile to prevent floating geometry.
        max_step = float(getattr(self.cfg, "feasibility_area_ratio_max", 5.0)) ** 0.5
        ring_r = self._ring_radius_profile(self.points)
        projected_r = project_connected_profile(
            ring_r,
            min_radius=float(self.cfg.min_radius),
            max_step_ratio=max_step,
            fix_endpoints=bool(getattr(self.cfg, "keep_electrode_rings_fixed", True)),
        )
        # Propagate projected radii back to point positions.
        for r_idx in range(self.cfg.num_rings):
            mask = self.ring_index == r_idx
            if not torch.any(mask):
                continue
            old_r = torch.norm(self.points[mask, :2], dim=1).clamp_min(1.0e-9)
            target_r = float(projected_r[r_idx].item())
            scale = target_r / old_r
            self.points[mask, :2] = self.points[mask, :2] * scale.unsqueeze(1)
        updated_radial_disp = torch.norm(self.points[:, :2], dim=1) - self.cfg.radius
        self.dent_field = torch.clamp(updated_radial_disp, max=0.0)
        self.bulge_field = torch.clamp(updated_radial_disp, min=0.0)
        self.velocity.zero_()

    def _build_info(
        self,
        metrics: RatedConditionMetrics,
        score: float,
        reward: float,
        free_energy: float,
        transient_summary: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        info = {
            "score": float(score),
            "reward_delta": float(reward),
            "free_energy": float(free_energy),
            "rated_voltage_v": float(metrics.voltage_v),
            "mean_temp": float(metrics.mean_temperature_k),
            "max_temp": float(metrics.max_temperature_k),
            "radiation_power": float(metrics.initial_net_band_power_w),
            "average_radiation_power": float(metrics.average_net_band_power_w),
            "mass_loss_rate": float(metrics.mass_loss_rate_kg_s),
            "remaining_mass": float(self.remaining_mass),
            "volume_change_ratio": float(metrics.volume_change_ratio),
            "feature_change_ratio": float(metrics.feature_change_ratio),
            "current_a": float(metrics.current_a),
            "circuit_resistance_ohm": float(metrics.resistance_ohm),
            "band_efficiency": float(metrics.band_efficiency),
            "temperature_uniformity": float(metrics.temperature_uniformity),
            "feasibility_penalty": float(metrics.feasibility_penalty),
            "thermo_mech_penalty": float(metrics.thermo_mech_penalty),
            "min_neck_diameter_mm": float(metrics.min_neck_diameter_mm),
            "max_radius_slope": float(metrics.max_radius_slope),
            "max_axial_stress_pa": float(metrics.max_axial_stress_pa),
            "lifetime_s": float(metrics.lifetime_s),
            "lifetime_ratio": float(
                metrics.lifetime_s / max((self.baseline_metrics or metrics).lifetime_s, 1.0e-9)
            ),
            "thermal_iterations": float(metrics.thermal_iterations),
            "thermal_residual_k": float(metrics.thermal_residual_k),
            "thermal_converged": float(1.0 if metrics.thermal_converged else 0.0),
            "view_factor_proxy": float(metrics.view_factor_proxy),
            "active_dent_points": float((torch.abs(self.dent_field) > self.cfg.dent_active_threshold).sum().item()),
            "max_dent_depth": float(torch.abs(self.dent_field).max().item()),
            "step": float(self.current_step),
        }
        if transient_summary is not None:
            info.update({k: float(v) for k, v in transient_summary.items()})
        return info

    def step(self, actions):
        self.current_step += 1
        self._sync_from_usd()

        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device).flatten()
        if a.numel() < 3:
            raise ValueError("actions must contain [index_ratio, indentation, sigma] or [index_ratio, indentation, sigma, dwell_time]")

        idx_ratio = float(torch.clamp(a[0], 0.0, 1.0).item())
        depth = float(torch.clamp(a[1], 0.0, 1.0).item()) * self.cfg.max_depth
        sigma = float(torch.clamp(torch.abs(a[2]), self.cfg.min_sigma, self.cfg.max_sigma).item())
        dwell_norm = float(torch.clamp(a[3], 0.0, 1.0).item()) if a.numel() >= 4 else 1.0
        target_idx = int(idx_ratio * (self.num_points - 1))

        previous_score = float(self.last_score)
        self._apply_design_action(target_idx, depth, sigma)
        metrics = self._evaluate_geometry()
        self.current_metrics = metrics
        self._update_pointwise_fields_from_metrics(metrics)
        transient_summary = self._evaluate_transient_window(metrics, dwell_norm)
        free_energy = self.compute_free_energy(self.points)
        score = (
            self._score_metrics(metrics)
            + self.cfg.reward_weight_transient_power * float(transient_summary["transient_power_ratio"])
            - self.cfg.reward_penalty_free_energy * free_energy
        )
        reward = score - previous_score
        self.last_score = score
        self.best_score = max(self.best_score, score)

        self._sync_to_usd()

        life_ratio = metrics.lifetime_s / max((self.baseline_metrics or metrics).lifetime_s, 1.0e-9)
        fail_feature = metrics.feature_change_ratio >= self.cfg.feature_fail_ratio
        fail_temp = metrics.max_temperature_k > self.cfg.max_temp
        fail_volume = metrics.volume_change_ratio > self.cfg.volume_tolerance_ratio
        fail_life = life_ratio < self.cfg.minimum_lifetime_ratio
        if getattr(self.cfg, "terminate_on_constraints", True):
            done = self.current_step >= self.cfg.max_steps or fail_feature or fail_temp or fail_volume or fail_life
        else:
            done = self.current_step >= self.cfg.max_steps

        info = self._build_info(metrics, score, reward, free_energy, transient_summary=transient_summary)
        return self._build_observation(), reward, done, info
