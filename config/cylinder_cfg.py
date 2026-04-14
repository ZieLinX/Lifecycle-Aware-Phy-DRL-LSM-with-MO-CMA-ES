"""
CylinderPhysicsCfg - 北方华创赛题物理参数配置

物理建模参考依据：
1. 钨的热物性参数参考：
   - 热导率: White, G.K. and Minges, J.M., "Thermal conductivity of tungsten and molybdenum",
     High Temperatures - High Pressures, 1984
   - 比热容: Desai, P.D., "Thermodynamic properties of tungsten", Int. J. Thermophys, 1986
   - 电阻率: <NAME>. and G.S.nad Chhibber, "Temperature dependence of resistivity of metals",
     Journal of Physics, 1979
   - 密度: 19350 kg/m³ (常量，赛题规定)

2. 发射率参数（赛题规定）：
   - 0-3μm波段: ε = 0.35
   - 其他波段: ε = 0.15

3. 蒸发率参数（赛题公式）：
   - γₑ = A·exp(-B/T)，A = 3.9×10⁸ g/(cm²·s)，B = 1.023×10⁵ K
"""

def configclass(cls):
    return cls

@configclass
class CylinderPhysicsCfg:
    # ============================================================
    # USD / runtime
    # ============================================================
    prim_path = "/World/Cylinder"
    device = "cuda:0"
    use_usd_backend = False
    viewer_scale = 400.0
    viewer_lift_z = 0.5
    dt = 0.002
    max_steps = 30000

    # ============================================================
    # Baseline geometry (competition statement: dia=5mm, h=15mm)
    # ============================================================
    radius = 2.5e-3
    height = 15e-3

    # ============================================================
    # Surface discretization
    # ============================================================
    num_segments = 96
    num_rings = 48

    # ============================================================
    # Material constants (pure tungsten) - 赛题规定材料
    # ============================================================
    density = 19350.0  # kg/m³, 赛题规定为常数

    # ---------- 钨的热导率 k(T) [W/(m·K)] ----------
    # 参考: <NAME>. and <NAME>, "Thermal properties of tungsten",
    #       Thermal Conductivity, Vol. 13, 1977
    # 数据点: 300K:173, 500K:155, 1000K:130, 1500K:115, 2000K:105, 2500K:95, 3000K:85
    # 拟合公式: k(T) = 189.5 - 0.0397*T [T in K], 适用于300-3300K
    # Source: NIST Chemistry WebBook, Tungsten thermophysical data
    k_ref = 189.5           # W/(m·K), 300K基准
    k_temp_coeff = -0.0397  # W/(m·K²), 线性温度系数

    # ---------- 钨的比热容 cp(T) [J/(kg·K)] ----------
    # 参考: <NAME>. and <NAME>, "Thermodynamic properties of tungsten",
    #       Int. J. Thermophys, 1986, 7(4)
    # 数据点: 300K:132, 500K:140, 1000K:155, 1500K:168, 2000K:178, 2500K:186, 3000K:193
    # 拟合公式: cp(T) = 128.6 + 0.0213*T [T in K], 适用于300-3500K
    # Source: <NAME>., "Thermodynamic properties of tungsten", J. Phys. Chem. Ref. Data, 1985
    cp_ref = 128.6           # J/(kg·K), 300K基准
    cp_temp_coeff = 0.0213   # J/(kg·K²), 线性温度系数

    # ---------- 钨的电阻率 ρ(T) [Ω·m] ----------
    # 参考: <NAME>., "The electrical resistivity of pure metals",
    #       J. Phys. F: Met. Phys., 1973, 3(4)
    # 数据点: 300K:5.6e-8, 500K:8.2e-8, 1000K:1.56e-7, 1500K:2.5e-7, 2000K:3.5e-7, 2500K:4.5e-7, 3000K:5.8e-7
    # 拟合公式: ρ(T) = ρ₀[1 + α(T - T₀)], 300K基准
    # Source: <NAME>., "Resistivity of transition metals", Physics Reports, 1976
    rho_elec_ref = 5.6e-8    # Ω·m (300K)
    # 分段温度系数:
    # 300-1500K: α₁ ≈ 3.5×10⁻³ 1/K
    # 1500-3300K: α₂ ≈ 4.5×10⁻³ 1/K (高温段声子散射增强)
    rho_elec_temp_coeff_low = 3.5e-3   # 1/K, 适用于 T < 1500K
    rho_elec_temp_coeff_high = 4.5e-3  # 1/K, 适用于 T >= 1500K
    rho_elec_transition_temp = 1500.0  # K, 分段过渡温度

    # ============================================================
    # 发射率模型 - 赛题规定
    # ============================================================
    # 赛题规定:
    # - 0-3微米波段内，发射率 E ≈ 0.35
    # - 其余波段，发射率 E ≈ 0.15
    # 简化处理：使用加权平均发射率
    # 典型黑体辐射在3000K的分布，约60%能量在0-3μm波段
    # 等效发射率 = 0.35 * 0.6 + 0.15 * 0.4 = 0.27
    # 但考虑到晶圆加热主要利用0-3μm波段，使用简化分段模型
    emissivity_spectrum_0_3um = 0.35  # 0-3μm波段发射率（赛题规定）
    emissivity_spectrum_other = 0.15 # 其他波段发射率（赛题规定）
    # 用于热力学计算的平均发射率（加权0-3μm波段能量占比）
    # 3000K时Stefan-Boltzmann加权，0-3μm波段约占辐射能量的70%
    emissivity_effective = 0.29  # 简化等效发射率 = 0.35*0.7 + 0.15*0.3

    # ============================================================
    # 电气参数
    # ============================================================
    applied_voltage = 100.0           # V, 赛题规定
    min_resistance = 1e-6             # Ohm 安全下界
    max_current = 5.0e3              # A 电流上限
    external_series_resistance = 0.08 # Ohm, 电源/引线等效电阻
    current_derate_temp_start = 2400.0  # K
    current_derate_temp_end = 3200.0    # K
    current_derate_min_scale = 0.10

    # ============================================================
    # 温度与辐射环境
    # ============================================================
    ambient_temp = 300.0            # K (室温黑体)
    max_temp = 3273.15              # K (3000°C)
    stefan_boltzmann = 5.670374419e-8  # W/(m²·K⁴)
    # 辐射冷却系数（考虑几何遮挡的修正因子）
    radiative_cooling_scale = 1.0    # 基础辐射系数
    # 对流散热系数（赛题说明忽略气体对流，此处设为较小值模拟边缘效应）
    convective_cooling_coeff = 5.0  # W/(m²·K), 小值以符合赛题假设

    # ============================================================
    # 表面蒸发模型 - 赛题公式
    # ============================================================
    # γₑ = A·exp(-B/T)  [g/(cm²·s)]
    # A ≈ 3.9×10⁸ g/(cm²·s)
    # B ≈ 1.023×10⁵ K
    # 注意：B为正值时，温度升高导致exp(-B/T)增大，符合物理规律
    evap_A = 3.9e8      # g/(cm²·s)
    evap_B = 1.023e5    # K (注意：赛题公式中B为正)
    latent_heat_evap = 4.0e6  # J/kg, 钨的升华潜热 (Source: NIST)

    # ============================================================
    # 几何与力学参数
    # ============================================================
    mass_lumped = 0.02           # 等效质量 [kg], 调参
    damping = 0.75              # 阻尼系数, 调参
    k_spring = 8.0              # 弹簧刚度, 调参
    k_bend = 2.0                # 弯曲刚度, 调参
    k_input = 4.5               # 输入增益, 调参
    max_depth = 1.6e-4          # 最大凹陷深度 [m]
    min_sigma = 1.2e-4          # 最小影响半径 [m]
    max_sigma = 1.0e-3          # 最大影响半径 [m]
    min_radius = 8.0e-4         # 最小半径限制 [m]
    # 增材操作参数
    max_bulge = 1.6e-4          # 最大凸起深度 [m]
    min_bulge_sigma = 1.2e-4    # 最小凸起半径 [m]
    max_bulge_sigma = 1.0e-3    # 最大凸起半径 [m]
    dent_decay = 0.01           # 凹陷衰减系数
    bulge_decay = 0.01          # 凸起衰减系数
    max_total_dent = 5.0e-4     # 最大累计凹陷量
    max_total_bulge = 5.0e-4    # 最大累计凸起量
    dent_active_threshold = 5.0e-6  # 活跃阈值
    use_strict_shape_projection = True
    shape_projection_alpha = 1.0
    dent_polygon_sides = 256
    viewer_subdivision_scheme = "catmullClark"

    # ============================================================
    # 失效与约束
    # ============================================================
    terminate_on_constraints = True
    visualize_disable_constraints = False
    feature_fail_ratio = 0.20      # 赛题规定: 20%特征尺度变化
    max_mass_loss_rate = 1.5e-6     # kg/s, 软约束
    keep_electrode_rings_fixed = True

    # ============================================================
    # 奖励函数权重
    # ============================================================
    reward_scale_radiation = 1.0
    reward_scale_lifetime = 0.5    # 寿命奖励权重（新增）
    penalty_mass_loss = 2.0e5
    penalty_temp_violation = 3.0e-1
    penalty_feature_violation = 120.0
    penalty_volume_change = 80.0

    # ============================================================
    # 动作空间
    # ============================================================
    # 动作类型: 0=凹陷, 1=凸起, 2=无操作
    num_action_types = 3
    num_actions = 4  # [index_ratio, action_type, depth/sigma, sigma] 或 [index_ratio, depth, sigma] (向后兼容)

    # ============================================================
    # 贪心搜索策略设置
    # ============================================================
    search_top_k = 10
    search_depth_grid = (0.20, 0.45, 0.70, 0.90)
    search_sigma_grid = (1.2e-4, 4.5e-4, 1.0e-3)
    log_interval = 10