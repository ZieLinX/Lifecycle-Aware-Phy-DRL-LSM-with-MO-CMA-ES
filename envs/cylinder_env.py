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
        """
        计算材料物性参数随温度的变化

        热导率: k(T) = k_ref + k_temp_coeff * (T - 300)
        比热容: cp(T) = cp_ref + cp_temp_coeff * (T - 300)
        电阻率: 分段线性 - 低温段和高温段使用不同温度系数
        发射率: 使用赛题规定的简化等效发射率模型
        """
        delta = torch.clamp(temperature - 300.0, min=0.0)

        # 热导率 [W/(m·K)]
        k = torch.clamp(
            self.cfg.k_ref + self.cfg.k_temp_coeff * delta,
            min=20.0,
            max=250.0
        )

        # 比热容 [J/(kg·K)]
        cp = self.cfg.cp_ref + self.cfg.cp_temp_coeff * delta

        # 电阻率 [Ω·m] - 分段线性模型
        # 低温段 (T < rho_elec_transition_temp): 使用低温度系数
        # 高温段 (T >= rho_elec_transition_temp): 使用高温度系数
        T_trans = self.cfg.rho_elec_transition_temp
        coeff_low = self.cfg.rho_elec_temp_coeff_low
        coeff_high = self.cfg.rho_elec_temp_coeff_high

        # 创建温度依赖的温度系数
        coeff = torch.where(
            temperature < T_trans,
            torch.full_like(temperature, coeff_low),
            torch.full_like(temperature, coeff_high)
        )
        rho_elec = self.cfg.rho_elec_ref * (1.0 + coeff * delta)

        # 发射率 - 赛题规定的简化模型
        # 使用等效平均发射率 (考虑0-3μm波段辐射占比)
        emissivity = torch.full_like(temperature, self.cfg.emissivity_effective)

        return cp, k, rho_elec, emissivity

    def _evaporation_flux_kg_m2_s(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        计算蒸发率 [kg/(m²·s)]

        赛题公式: γₑ = A·exp(-B/T) [g/(cm²·s)]
        - A = 3.9×10⁸ g/(cm²·s)
        - B = 1.023×10⁵ K
        转换因子: 1 g/(cm²·s) = 10 kg/(m²·s)

        注意: B为正值时，温度升高导致exp(-B/T)增大，符合"温度越高蒸发越快"的物理规律
        """
        y_g_cm2_s = self.cfg.evap_A * torch.exp(-self.cfg.evap_B / torch.clamp(temperature, min=1.0))
        return y_g_cm2_s * 10.0

    def _surface_area_to_volume_ratio(self, radius: torch.Tensor) -> torch.Tensor:
        r = torch.clamp(radius, min=self.cfg.min_radius)
        return 2.0 / r

    def _compute_radiation_with_occlusion(self, points: torch.Tensor, temperature: torch.Tensor, emissivity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算辐射散热，考虑几何自遮挡效应

        简化遮挡模型：
        1. 对于每个表面点，计算其相对于外接球的有效发射因子
        2. 凹陷区域（局部凹点）的有效发射面积被周围表面遮挡
        3. 使用基于曲率的遮挡估计：负曲率（凹陷）区域降低有效辐射

        返回:
        - q_rad: 每个点的辐射散热功率 [W/m²]
        - occlusion_factor: 每个点的遮挡因子 [0, 1]
        """
        n_points = points.shape[0]

        # 计算每个点的法向量（近似：通过邻域平面拟合）
        xy = points[:, :2]
        radial = torch.norm(xy, dim=1)
        # 圆柱表面的法向量方向为径向（指向外部）
        radial_norm = torch.clamp(radial, min=1e-6).unsqueeze(1)
        normal = torch.cat([xy / radial_norm, torch.zeros(n_points, 1, device=self.device)], dim=1)

        # 计算局部曲率：使用Laplacian估计
        # 正值表示凸起（增大辐射），负值表示凹陷（减小辐射）
        radial_disp = radial - self.cfg.radius
        lap_curvature = self._laplacian_scalar(radial_disp)

        # 基于曲率的遮挡因子
        # 曲率为负（凹陷）时，遮挡因子减小
        # 使用sigmoid函数平滑过渡
        curvature_threshold = 0.0
        curvature_scale = 50.0  # 调整灵敏度
        occlusion_factor = torch.sigmoid((lap_curvature - curvature_threshold) * curvature_scale)
        occlusion_factor = torch.clamp(occlusion_factor, min=0.3, max=1.0)  # 最小遮挡30%

        # 计算辐射散热
        # Stefan-Boltzmann定律: q = ε·σ·(T⁴ - T_ambient⁴)
        T_ambient = self.cfg.ambient_temp
        q_rad = (
            emissivity
            * self.cfg.stefan_boltzmann
            * occlusion_factor  # 考虑遮挡
            * (temperature.pow(4) - T_ambient**4)
        )

        # 计算总辐射功率
        total_power = torch.sum(q_rad * self.area_per_point)

        return q_rad, occlusion_factor

    def _compute_feature_scales(self, points: torch.Tensor) -> Dict[str, float]:
        """
        计算多个特征尺度，用于判断器件失效

        赛题规定: 当任意特征尺度变化率 [Li(t)-Li(0)]/Li(0) >= 20% 时认为器件失效

        特征尺度定义:
        - mean_radius: 平均半径
        - max_radius: 最大局部半径
        - min_radius: 最小局部半径
        - height: 总高度
        - surface_area: 估算表面积
        """
        xy = points[:, :2]
        radii = torch.norm(xy, dim=1)

        feature_scales = {
            "mean_radius": float(torch.mean(radii).item()),
            "max_radius": float(torch.max(radii).item()),
            "min_radius": float(torch.min(radii).item()),
            "height": float(torch.max(points[:, 2]) - torch.min(points[:, 2]).item()),
            "surface_area": float(self.num_points * self.area_per_point),  # 简化估计
        }

        return feature_scales

    def _compute_lifetime(self, current_state: Dict) -> float:
        """
        估算器件剩余寿命

        基于当前蒸发率和特征尺度退化率，预测器件达到失效阈值的时间

        返回: 预估剩余寿命 [秒]，如果已经失效返回0
        """
        # 特征尺度变化率
        feature_ratio = self._compute_feature_change_ratio(self.points)

        # 如果已经失效，返回0
        if feature_ratio >= self.cfg.feature_fail_ratio:
            return 0.0

        # 基于当前蒸发率和质量损失速率，估算特征尺度退化时间
        # 简化模型：假设特征尺度变化与质量损失成比例
        mass_remaining = self.remaining_mass
        mass_fraction = mass_remaining / max(self.initial_mass, 1e-12)

        # 基于质量损失估算寿命
        # 假设特征尺度变化与质量损失有某种对应关系
        # 当质量损失达到某阈值时，特征尺度变化达到20%
        mass_loss_for_failure = self.initial_mass * self.cfg.feature_fail_ratio
        mass_lost = self.initial_mass - mass_remaining

        if mass_lost >= mass_loss_for_failure:
            return 0.0

        # 估算当前质量损失速率
        if hasattr(self, '_last_mass_loss_rate'):
            current_rate = self._last_mass_loss_rate
        else:
            current_rate = 1e-9  # 假设初始损失速率

        if current_rate > 1e-12:
            remaining_mass_for_failure = mass_loss_for_failure - mass_lost
            remaining_lifetime = remaining_mass_for_failure / current_rate
        else:
            remaining_lifetime = 1e6  # 假设极大寿命

        return float(remaining_lifetime)

    def _estimate_initial_lifetime(self) -> float:
        """
        估算初始形状的器件寿命

        用于计算优化后寿命与初始寿命的比值（赛题要求: 优化后寿命 >= 初始寿命的30%）
        """
        # 初始形状的特征尺度
        initial_radii = self.cfg.radius
        initial_volume = self.initial_volume

        # 使用初始温度场估算蒸发率
        initial_temp = self.cfg.ambient_temp + 100.0  # 假设初始温升100K
        evap_flux = self.cfg.evap_A * math.exp(-self.cfg.evap_B / initial_temp) * 10.0  # kg/(m²·s)
        initial_area = 2 * math.pi * self.cfg.radius * self.cfg.height
        initial_mass_loss_rate = evap_flux * initial_area

        # 假设特征尺度变化与质量损失线性相关
        # 20%特征尺度变化对应约15%的质量损失
        mass_loss_for_failure = self.initial_mass * 0.15

        if initial_mass_loss_rate > 1e-12:
            return mass_loss_for_failure / initial_mass_loss_rate
        else:
            return 1e6  # 极大寿命

    def initialize_lifetime_tracking(self):
        """初始化寿命跟踪相关变量"""
        self.initial_lifetime = self._estimate_initial_lifetime()
        self.lifetime_history = []
        self._last_mass_loss_rate = 1e-9

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

        # 初始化寿命跟踪
        self.initialize_lifetime_tracking()

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

        # 支持两种动作格式:
        # 格式1: [index_ratio, depth, sigma] - 保持向后兼容（仅凹陷）
        # 格式2: [index_ratio, action_type, depth, sigma] - 新格式（凹陷/凸起）
        if a.numel() >= 4:
            # 新格式: [index_ratio, action_type, depth, sigma]
            idx_ratio = float(torch.clamp(a[0], 0.0, 1.0).item())
            action_type = int(torch.clamp(a[1], 0.0, 2.0).item())  # 0=凹陷, 1=凸起, 2=无操作
            depth = float(torch.clamp(a[2], 0.0, 1.0).item())
            sigma = float(torch.clamp(torch.abs(a[3]), self.cfg.min_sigma, self.cfg.max_sigma).item())
        elif a.numel() >= 3:
            # 旧格式: [index_ratio, depth, sigma] (仅凹陷)
            idx_ratio = float(torch.clamp(a[0], 0.0, 1.0).item())
            action_type = 0  # 默认凹陷
            depth = float(torch.clamp(a[1], 0.0, 1.0).item())
            sigma = float(torch.clamp(torch.abs(a[2]), self.cfg.min_sigma, self.cfg.max_sigma).item())
        else:
            raise ValueError("actions must contain at least [index_ratio, depth, sigma]")

        target_idx = int(idx_ratio * (self.num_points - 1))

        # 如果是无操作类型，直接跳过几何变形
        if action_type == 2:
            # 无操作，跳过变形但继续物理仿真
            local_dent = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
            local_bulge = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
            delta_h = 0.0
        else:
            # 计算凹陷/凸起场
            # Regular hemispherical profile on unwrapped cylinder surface.
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

            # 根据动作类型计算变形
            if action_type == 0:
                # 凹陷（减材）：向内变形
                depth_value = depth * self.cfg.max_depth
                local_dent = -depth_value * hemi_core
                local_bulge = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
            else:
                # 凸起（增材）：向外变形
                depth_value = depth * getattr(self.cfg, 'max_bulge', self.cfg.max_depth)
                local_dent = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
                local_bulge = depth_value * hemi_core

            # 计算体积变化
            dent_volume = -torch.sum(local_dent) * self.area_per_point
            bulge_volume = torch.sum(local_bulge) * self.area_per_point
            net_volume_change = bulge_volume - dent_volume

            # 体积守恒：将体积变化转换为高度补偿
            base_cross_area = math.pi * (self.cfg.radius ** 2)
            delta_h = net_volume_change / max(base_cross_area, 1e-12)

            # 更新变形场
            self.dent_field = self.dent_field * (1.0 - self.cfg.dent_decay) + local_dent
            self.dent_field = torch.clamp(self.dent_field, -self.cfg.max_total_dent, 0.0)

            self.bulge_field = self.bulge_field * (1.0 - getattr(self.cfg, 'bulge_decay', self.cfg.dent_decay)) + local_bulge
            self.bulge_field = torch.clamp(self.bulge_field, 0.0, self.cfg.max_total_bulge)

        # 计算外力
        radial = torch.norm(self.points[:, :2], dim=1)
        radial_disp = radial - self.cfg.radius
        lap = self._laplacian_term(radial_disp)

        # 总变形场 = 凹陷 + 凸起
        total_deform = self.dent_field + self.bulge_field

        # 恢复力相对于总变形场
        radial_error = radial_disp - total_deform
        f_spring_xy = -radial_dir * (self.cfg.k_spring * radial_error).unsqueeze(1)
        f_bend_xy = -radial_dir * (self.cfg.k_bend * lap).unsqueeze(1)

        # 外力由凹陷和凸起共同决定
        if action_type == 2:
            f_ext_xy = torch.zeros(self.num_points, device=self.device)
        elif action_type == 0:
            f_ext_xy = (radial_dir * local_dent.unsqueeze(1) * self.cfg.k_input).squeeze(-1)
        else:
            f_ext_xy = (radial_dir * local_bulge.unsqueeze(1) * self.cfg.k_input).squeeze(-1)

        f_internal = torch.cat([f_spring_xy + f_bend_xy, torch.zeros(self.num_points, 1, device=self.device)], dim=1)
        f_ext = torch.cat([f_ext_xy.unsqueeze(-1) if f_ext_xy.dim() == 1 else f_ext_xy, torch.zeros(self.num_points, 1, device=self.device)], dim=1)

        # Damping + integration.
        force_total = f_ext + f_internal - self.cfg.damping * self.velocity
        acc = force_total / self.cfg.mass_lumped
        self.velocity = self.velocity + acc * self.cfg.dt
        self.points = self.points + self.velocity * self.cfg.dt
        self._apply_shape_projection()

        # Axial volume compensation:
        # Add missing volume to cylinder height (top region), using original cross-section area.
        if abs(delta_h) > 1e-10:
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

        # 使用带遮挡的辐射计算
        q_rad, occlusion_factor = self._compute_radiation_with_occlusion(
            self.points, self.temperature, emissivity
        )

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

        # 更新质量损失速率跟踪
        self._last_mass_loss_rate = dm_total / max(self.cfg.dt, 1e-12)

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

        # 计算寿命相关指标
        remaining_lifetime = self._compute_lifetime({})

        # 奖励函数：包含辐射功率和寿命奖励
        lifetime_reward = 0.0
        if hasattr(self, 'initial_lifetime') and self.initial_lifetime > 0:
            lifetime_ratio = remaining_lifetime / self.initial_lifetime
            # 赛题要求: 优化后寿命 >= 初始寿命的30%
            if lifetime_ratio >= 0.3:
                lifetime_reward = self.cfg.reward_scale_lifetime * lifetime_ratio
            else:
                # 寿命不足30%的惩罚
                lifetime_reward = -10.0 * (0.3 - lifetime_ratio)

        reward = (
            self.cfg.reward_scale_radiation * float(q_rad_net.item())
            + lifetime_reward
            - self.cfg.penalty_mass_loss * max(mass_loss_rate - self.cfg.max_mass_loss_rate, 0.0)
            - self.cfg.penalty_temp_violation * (max_temp_violation**2)
            - self.cfg.penalty_feature_violation * max(feature_change_ratio - self.cfg.feature_fail_ratio, 0.0)
            - self.cfg.penalty_volume_change * volume_change_ratio
            - 0.02 * float(free_energy.item())
        )

        fail_feature = feature_change_ratio >= self.cfg.feature_fail_ratio
        fail_temp = torch.max(self.temperature).item() >= self.cfg.max_temp * 1.02
        fail_lifetime = hasattr(self, 'initial_lifetime') and self.initial_lifetime > 0 and remaining_lifetime / self.initial_lifetime < 0.3

        if getattr(self.cfg, "terminate_on_constraints", True):
            done = self.current_step >= self.cfg.max_steps or fail_feature or fail_temp or fail_lifetime
        else:
            done = self.current_step >= self.cfg.max_steps

        # 记录寿命历史
        if hasattr(self, 'lifetime_history'):
            self.lifetime_history.append(remaining_lifetime)

        info: Dict[str, float] = {
            "free_energy": float(free_energy.item()),
            "mean_radius": float(torch.norm(self.points[:, :2], dim=1).mean().item()),
            "active_dent_points": float((torch.abs(self.dent_field) > self.cfg.dent_active_threshold).sum().item()),
            "max_dent_depth": float(torch.abs(self.dent_field).max().item()),
            "active_bulge_points": float((torch.abs(self.bulge_field) > self.cfg.dent_active_threshold).sum().item()),
            "max_bulge_height": float(torch.abs(self.bulge_field).max().item()),
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
            # 新增：寿命相关指标
            "remaining_lifetime": float(remaining_lifetime),
            "initial_lifetime": float(getattr(self, 'initial_lifetime', 0.0)),
            "lifetime_ratio": float(remaining_lifetime / max(getattr(self, 'initial_lifetime', 1.0), 1e-12)),
        }
        return self._build_observation(), reward, done, info