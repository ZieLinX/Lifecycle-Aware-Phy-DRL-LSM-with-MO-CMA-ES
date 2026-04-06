import math
import importlib
from typing import Dict, List, Tuple

import numpy as np
import torch


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
        self.obs_dim = self.num_points * 5
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
        obs = torch.cat(
            [
                self.points.flatten(),
                self.velocity.flatten(),
                radial_disp.flatten(),
                temp_norm.flatten(),
                self.ablation_depth.flatten(),
            ]
        )
        return obs.detach().cpu().numpy()

    def _material_properties(self, temperature: torch.Tensor):
        delta = torch.clamp(temperature - 300.0, min=0.0)
        cp = self.cfg.cp_ref + self.cfg.cp_temp_coeff * delta
        k = torch.clamp(self.cfg.k_ref + self.cfg.k_temp_coeff * delta, min=20.0)
        rho_elec = self.cfg.rho_elec_ref * (1.0 + self.cfg.rho_elec_temp_coeff * delta)
        emissivity = self.cfg.emissivity_low + (
            self.cfg.emissivity_high - self.cfg.emissivity_low
        ) * torch.sigmoid((temperature - self.cfg.emissivity_transition_temp) / 220.0)
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

    def compute_free_energy(self, points: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        radial = torch.norm(points[:, :2], dim=1)
        radial_disp = radial - self.cfg.radius

        # U_spring: isotropic radial restoring energy
        u_spring = 0.5 * self.cfg.k_spring * torch.sum(radial_disp.pow(2))

        # U_bend: smoothness penalty from graph Laplacian
        lap = self._laplacian_term(radial_disp)
        u_bend = 0.5 * self.cfg.k_bend * torch.sum(lap.pow(2))

        # Kinetic term
        kinetic = 0.5 * self.cfg.mass_lumped * torch.sum(velocity.pow(2))

        # Entropy proxy: variance of local displacement field
        entropy = torch.var(radial_disp)

        return (u_spring + u_bend + kinetic) - (self.cfg.ambient_temp * entropy)

    def _compute_volume(self, points: torch.Tensor) -> float:
        radius = torch.norm(points[:, :2], dim=1).mean().item()
        return math.pi * radius * radius * self.cfg.height

    def _compute_feature_change_ratio(self, points: torch.Tensor) -> float:
        radius_mean = torch.norm(points[:, :2], dim=1).mean()
        radius_ratio = torch.abs(radius_mean - self.cfg.radius) / max(self.cfg.radius, 1e-12)
        ablation_ratio = torch.max(self.ablation_depth) / max(self.cfg.radius, 1e-12)
        return float(torch.max(radius_ratio, ablation_ratio).item())

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

        # Regular hemispherical dent profile on unwrapped cylinder surface.
        # Use an N-gon footprint (N -> infinity approximates circular hemisphere domain).
        theta = torch.atan2(self.points[:, 1], self.points[:, 0])
        theta0 = theta[target_idx]
        dtheta = torch.atan2(torch.sin(theta - theta0), torch.cos(theta - theta0))
        du = self.cfg.radius * dtheta
        dv = self.points[:, 2] - self.points[target_idx, 2]
        dist = torch.sqrt(du * du + dv * dv)
        radius = sigma

        sides = int(max(getattr(self.cfg, "dent_polygon_sides", 256), 3))
        phi = torch.atan2(dv, du)
        sector = (2.0 * math.pi) / float(sides)
        # Fold direction into one sector and compute direction-dependent boundary radius.
        folded = torch.remainder(phi + math.pi, sector) - 0.5 * sector
        apothem = radius * math.cos(math.pi / float(sides))
        boundary = apothem / torch.clamp(torch.cos(folded), min=1e-6)
        rho = dist / torch.clamp(boundary, min=1e-12)
        inside = rho <= 1.0
        hemi_core = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
        hemi_core[inside] = torch.sqrt(
            torch.clamp(1.0 - rho[inside].pow(2), min=0.0)
        )

        xy = self.points[:, :2]
        radial_norm = torch.norm(xy, dim=1, keepdim=True).clamp_min(1e-6)
        radial_dir = xy / radial_norm

        # Local inward dent.
        local_dent = -depth * hemi_core

        # Volume conservation:
        # Convert missing dent volume to a top-height compensation term.
        # Desired relation: delta_h = dV / A0, where A0 is the original cylinder cross-section area.
        missing_volume = -torch.sum(local_dent) * self.area_per_point
        base_cross_area = math.pi * (self.cfg.radius ** 2)
        delta_h = max(float(missing_volume.item()), 0.0) / max(base_cross_area, 1e-12)

        # Persistent radial target only keeps dents (height compensation is axial).
        self.dent_field = self.dent_field * (1.0 - self.cfg.dent_decay) + local_dent
        self.dent_field = torch.clamp(self.dent_field, -self.cfg.max_total_dent, 0.0)

        # External force follows local dent target.
        f_ext_xy = radial_dir * local_dent.unsqueeze(1) * self.cfg.k_input
        f_ext = torch.cat([f_ext_xy, torch.zeros(self.num_points, 1, device=self.device)], dim=1)

        radial = torch.norm(self.points[:, :2], dim=1)
        radial_disp = radial - self.cfg.radius
        lap = self._laplacian_term(radial_disp)

        # Restoring force is measured against accumulated dent target field.
        radial_error = radial_disp - self.dent_field
        f_spring_xy = -radial_dir * (self.cfg.k_spring * radial_error).unsqueeze(1)
        f_bend_xy = -radial_dir * (self.cfg.k_bend * lap).unsqueeze(1)
        f_internal = torch.cat([f_spring_xy + f_bend_xy, torch.zeros(self.num_points, 1, device=self.device)], dim=1)

        # Damping + integration.
        force_total = f_ext + f_internal - self.cfg.damping * self.velocity
        acc = force_total / self.cfg.mass_lumped
        self.velocity = self.velocity + acc * self.cfg.dt
        self.points = self.points + self.velocity * self.cfg.dt
        self._apply_shape_projection()

        # Axial volume compensation:
        # Add missing volume to cylinder height (top region), using original cross-section area.
        if delta_h > 0.0:
            if self.cfg.keep_electrode_rings_fixed:
                first_ring = self.ring_index == 0
                last_ring = self.ring_index == int(torch.max(self.ring_index).item())
                movable = ~(first_ring | last_ring)
            else:
                movable = torch.ones(self.num_points, dtype=torch.bool, device=self.device)

            if torch.any(movable):
                ring_f = self.ring_index.float()
                ring_min = torch.min(ring_f[movable])
                ring_max = torch.max(ring_f[movable])
                ring_span = torch.clamp(ring_max - ring_min, min=1.0)
                # Compensation is concentrated toward upper region and reaches max at top.
                z_weight = torch.clamp((ring_f - ring_min) / ring_span, 0.0, 1.0) * movable.float()
                self.points[:, 2] = self.points[:, 2] + z_weight * delta_h

        # Keep minimum radius to avoid collapsing through axis.
        xy = self.points[:, :2]
        r = torch.norm(xy, dim=1, keepdim=True).clamp_min(1e-6)
        collapsed = r < self.cfg.min_radius
        if collapsed.any():
            self.points[:, :2] = torch.where(collapsed, xy / r * self.cfg.min_radius, xy)
            self.velocity[:, :2] *= (~collapsed).float()

        # --------------------- Electro-thermal + evaporation ---------------------
        cp, k, rho_elec, emissivity = self._material_properties(self.temperature)
        radius_now = torch.norm(self.points[:, :2], dim=1).clamp_min(self.cfg.min_radius)
        mean_radius = torch.mean(radius_now)
        cross_area = math.pi * float(mean_radius.item()) ** 2
        internal_resistance = max(
            float(torch.mean(rho_elec).item()) * self.cfg.height / max(cross_area, 1e-12),
            self.cfg.min_resistance,
        )
        total_resistance = internal_resistance + float(getattr(self.cfg, "external_series_resistance", 0.0))
        current_ideal = self.cfg.applied_voltage / max(total_resistance, self.cfg.min_resistance)
        t_max_now = float(torch.max(self.temperature).item())
        t0 = float(getattr(self.cfg, "current_derate_temp_start", self.cfg.max_temp * 0.75))
        t1 = float(getattr(self.cfg, "current_derate_temp_end", self.cfg.max_temp * 0.98))
        min_scale = float(getattr(self.cfg, "current_derate_min_scale", 0.15))
        if t1 <= t0:
            derate = 1.0
        else:
            ratio = np.clip((t_max_now - t0) / (t1 - t0), 0.0, 1.0)
            derate = 1.0 - (1.0 - min_scale) * ratio
        current = float(np.clip(current_ideal * derate, 0.0, self.cfg.max_current))
        current_density = current / max(cross_area, 1e-12)

        q_joule = (current_density**2) * rho_elec
        temp_lap = self._laplacian_scalar(self.temperature)
        sa_over_vol = self._surface_area_to_volume_ratio(radius_now)
        q_rad = (
            emissivity
            * self.cfg.stefan_boltzmann
            * getattr(self.cfg, "radiative_cooling_scale", 1.0)
            * (
            self.temperature.pow(4) - self.cfg.ambient_temp**4
            )
        )
        q_conv = getattr(self.cfg, "convective_cooling_coeff", 0.0) * (
            self.temperature - self.cfg.ambient_temp
        )

        evap_flux = self._evaporation_flux_kg_m2_s(self.temperature)
        q_evap = evap_flux * self.cfg.latent_heat_evap
        alpha = k / (self.cfg.density * cp)
        dT_dt = (
            q_joule / (self.cfg.density * cp)
            + alpha * temp_lap / max(self.dx * self.dx, 1e-12)
            - (q_rad + q_conv + q_evap) * sa_over_vol / (self.cfg.density * cp)
        )
        self.temperature = torch.clamp(
            self.temperature + dT_dt * self.cfg.dt,
            min=self.cfg.ambient_temp,
            max=self.cfg.max_temp * 1.2,
        )

        dm_point = evap_flux * self.area_per_point * self.cfg.dt
        dm_total = float(torch.sum(dm_point).item())
        self.remaining_mass = max(self.remaining_mass - dm_total, 0.0)
        thickness_loss = dm_point / (self.cfg.density * self.area_per_point + 1e-12)
        self.ablation_depth = self.ablation_depth + thickness_loss
        self.points[:, :2] = self.points[:, :2] - radial_dir * thickness_loss.unsqueeze(1)

        self._apply_electrode_constraints()

        self._sync_to_usd()

        # ------------------------------ Objective --------------------------------
        free_energy = self.compute_free_energy(self.points, self.velocity)
        current_volume = self._compute_volume(self.points)
        volume_change_ratio = abs(current_volume - self.initial_volume) / max(self.initial_volume, 1e-12)
        feature_change_ratio = self._compute_feature_change_ratio(self.points)
        max_temp_violation = max(float(torch.max(self.temperature).item() - self.cfg.max_temp), 0.0)
        mass_loss_rate = dm_total / max(self.cfg.dt, 1e-12)

        q_rad_net = torch.sum(q_rad * self.area_per_point)
        reward = (
            self.cfg.reward_scale_radiation * float(q_rad_net.item())
            - self.cfg.penalty_mass_loss * max(mass_loss_rate - self.cfg.max_mass_loss_rate, 0.0)
            - self.cfg.penalty_temp_violation * (max_temp_violation**2)
            - self.cfg.penalty_feature_violation * max(feature_change_ratio - self.cfg.feature_fail_ratio, 0.0)
            - self.cfg.penalty_volume_change * volume_change_ratio
            - 0.02 * float(free_energy.item())
        )

        fail_feature = feature_change_ratio >= self.cfg.feature_fail_ratio
        fail_temp = torch.max(self.temperature).item() >= self.cfg.max_temp * 1.02
        if getattr(self.cfg, "terminate_on_constraints", True):
            done = self.current_step >= self.cfg.max_steps or fail_feature or fail_temp
        else:
            done = self.current_step >= self.cfg.max_steps
        info: Dict[str, float] = {
            "free_energy": float(free_energy.item()),
            "mean_radius": float(torch.norm(self.points[:, :2], dim=1).mean().item()),
            "active_dent_points": float((torch.abs(self.dent_field) > self.cfg.dent_active_threshold).sum().item()),
            "max_dent_depth": float(torch.abs(self.dent_field).max().item()),
            "height_compensation_step": float(delta_h),
            "shape_projection_alpha": float(getattr(self.cfg, "shape_projection_alpha", 0.0)),
            "mean_temp": float(torch.mean(self.temperature).item()),
            "max_temp": float(torch.max(self.temperature).item()),
            "mass_loss_rate": float(mass_loss_rate),
            "remaining_mass": float(self.remaining_mass),
            "volume_change_ratio": float(volume_change_ratio),
            "feature_change_ratio": float(feature_change_ratio),
            "radiation_power": float(q_rad_net.item()),
            "convective_power": float(torch.sum(q_conv * self.area_per_point).item()),
            "current_a": float(current),
            "circuit_resistance_ohm": float(total_resistance),
            "step": float(self.current_step),
        }
        return self._build_observation(), reward, done, info