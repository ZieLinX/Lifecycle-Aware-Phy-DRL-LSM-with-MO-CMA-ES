import torch
import numpy as np
from omni.isaac.core.objects import DynamicCylinder
from pxr import UsdGeom

class CylinderPhysicsEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda:0"
        self.stage = omni.usd.get_context().get_stage()
        
        # 创建圆柱体
        self.cyl = DynamicCylinder(prim_path=cfg.prim_path, radius=cfg.radius, height=cfg.height)
        self.mesh = UsdGeom.Mesh.Get(self.stage, cfg.prim_path)

    def compute_free_energy(self, points):
        """
        根据图片公式推导: F = U - TS
        假设 U 与格点位移平方成正比 (Harmonic Oscillator)
        """
        # 计算位移矢量 (偏离标准圆柱表面的程度)
        dist_from_center = torch.norm(points[:, :2], dim=1)
        displacement = dist_from_center - self.cfg.radius
        
        # 势能 U (简谐近似)
        u_energy = 0.5 * torch.sum(displacement**2)
        
        # 熵项 S (假设与形变局域化程度相关)
        entropy = -torch.var(displacement) 
        
        # Helmholtz 自由能
        free_energy = u_energy - (self.cfg.temp_k * entropy)
        return free_energy

    def step(self, actions):
        # 1. 获取当前网格顶点
        points_usd = self.mesh.GetPointsAttr().Get()
        points = torch.tensor(np.array(points_usd), device=self.device)

        # 2. 解析动作并应用形变
        target_idx = int(actions[0] * (len(points) - 1))
        depth = actions[1] * 0.1  # 最大凹陷 10cm
        sigma = actions[2] * 0.2  # 凹陷影响范围
        
        # 计算高斯凹陷
        dists = torch.norm(points - points[target_idx], dim=1)
        influence = torch.exp(-dists**2 / (2 * sigma**2))
        
        # 向心收缩 (产生凹陷)
        direction = -points[:, :2] / torch.norm(points[:, :2], dim=1, keepdim=True)
        points[:, :2] += direction * depth * influence.unsqueeze(1)

        # 3. 更新 USD 网格
        self.mesh.GetPointsAttr().Set(points.cpu().numpy())

        # 4. 计算奖励 (目标：自由能最小化)
        energy = self.compute_free_energy(points)
        reward = -energy # 能量越低奖励越高
        
        return points.flatten(), reward, False, {}