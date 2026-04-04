from omni.isaac.lab.utils import configclass

@configclass
class CylinderPhysicsCfg:
    prim_path = "/World/Cylinder"
    radius = 0.5
    height = 1.0
    # 物理常数 (对应图片推导)
    temp_k = 600.0   # 目标温度 (K)
    kb = 1.38e-23    # 玻尔兹曼常数
    hbar = 1.054e-34 # 约化普朗克常数
    num_observations = 3072 # 1024个顶点 * xyz
    num_actions = 3        # [选中索引比例, 凹陷深度, 影响范围]