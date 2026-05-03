# 圆柱钨体 full3d 物理优化方案说明

## 1. 当前正式工作流

当前仓库已收敛为 **full3d-only** 工作流：

- 正式入口：`optimize_3d.py`
- 核心后端：`utils/full3d_optimizer.py`
- 几何表示：封闭三维 mesh，侧面、顶面、底面均可变化
- 优化路线：Phy-DRL-LSM inspired global initial-shape strategy-field CEM
- 输出目录：`outputs/three_d_runs/<experiment>_<stamp>/`

旧 1D/RL/hybrid/sidefield 入口、环境、测试和工具已移除；文档中的历史记录仅作为过程复盘，不再作为可运行路线。

## 2. 统一物理口径

- `100V` 是额定工况搜索上限，不是固定工作电压。
- 正式优化默认搜索 `V <= 100V` 的稳态工况，选择综合辐射收益和蒸发/寿命后的最优稳态。
- 初始 5mm x 15mm 圆柱的额定电压约 `0.34V`，固定 20V/100V 只适合做过温诊断。
- 正式评分主目标是额定工况下 `0-3um` escaped energy-conversion efficiency；净辐射功率仍保留为诊断字段。
- 电压只加在钨棒两端，不包含铜电极电压降。
- 钨棒内部显式计算轴向导热。
- 两端铜电极仍是固定的 `5mm` 圆形电极盘，温度为 `300K`。
- 钨棒两端面始终相距 `15mm`，但端面 footprint 可以改变，并不强制等于 `5mm` 圆。
- 电极导热边界按钨端面与固定 `5mm` 电极盘的实际重叠面积计算。
- 当前口径忽略钨棒-铜电极之间的接触热阻和接触电阻。
- 自由表面计算向 `300K` 环境的净辐射/散热和升华。
- 所有端面区域都不计向外辐射和升华；超出电极盘的端面不导热，原电极盘内但已不被钨端面覆盖的区域会减少接触导热面积。
- 0-3 微米有效辐射按外接吸收面统计；发射率使用题面给定口径：
  - `0-3 um`：`0.35`
  - 其他波段：`0.15`
- 蒸发律采用：

$$
Y_e = A \exp(B / T)
$$

其中 `A = 3.9e8 [g/(cm^2*s)]`，`B = -1.023e5 [K]`。

## 3. 几何与约束

- `5mm x 15mm` 圆柱只作为通电前材料体积基准和寿命 baseline，不再假设为最优初始形状。
- 每个 action 直接生成完整的通电前初始几何；通电前材料体积投影回初始圆柱体积。
- 两端 `5mm` 圆形电极盘保持直径和相对位置不变；钨棒端面 footprint 可变。
- 钨棒横向长度/端距保持 `15mm`。
- 侧面自由表面允许三维变形，端面允许平面内 footprint 变形。
- 形状评估同时检查体积、电极误差、温度、寿命和 `0-3um` energy-conversion efficiency。
- 候选必须满足 `lifetime_ratio >= 0.30`、`T <= 3000C`、体积相等和端距/电极边界约束。
- 固定电压模式 `--fixed-voltage <V>` 仅用于诊断固定电压是否过温，不用于正式优化。

## 4. 动作空间与优化算法

当前 full3d 不再是“圆柱附近局部形变 CEM”，而是显式定义 **Phy-DRL-LSM inspired global initial-shape strategy-field** 动作空间：

- strategy channels：`radiation`、`evaporation`、`current`、`direct`，默认 `strategy_channels_full3d = 4`。
- 侧壁策略场：`Chebyshev(z) x Fourier(theta)` 低阶系数输出四类 strategy map。
- 物理组合：strategy map 与辐射、蒸发、电流等局部物理敏度组合为法向速度场。
- 体积约束：法向速度经过 Lagrange 体积保持投影，生成同体积的完整初始几何。
- 顶/底面 footprint：`Chebyshev(radius) x Fourier(theta)` 平面内模式继续负责上下端面 footprint 变化。
- 端距约束：上下端面仍位于相距 `15mm` 的平面上。
- 电极接触：动作后按端面 footprint 与固定电极盘的重叠面积重算导热接触面积。
- 全局推进：`global_shape_steps_full3d` 表示每个 action 生成初始几何时的策略场推进步数。
- 可用 `--action-axial-modes`、`--action-circum-modes`、`--action-cap-radial-modes`、`--action-strategy-channels`、`--global-shape-steps` 调整动作空间。

优化算法采用 CEM：

- 每一代在上述动作空间中采样候选形变。
- 对每个候选 mesh 执行 `V <= 100V` 的额定电压搜索。
- 优先选择满足体积、电极、温度、热收敛和 `lifetime_ratio >= 0.30` 的候选。
- 精英排序的主指标是额定工况下 `0-3um` escaped energy-conversion efficiency。
- 用精英候选更新动作分布；轻量 3D U-Net/GNN 仅作为可选结构化扰动源。

## 5. 运行命令

快速检查：

```bash
python -u optimize_3d.py --smoke --no-step --experiment-name full3d_smoke
```

正式优化：

```bash
python -u optimize_3d.py \
  --experiment-name mcga_full3d_4090 \
  --output-dir outputs/three_d_runs \
  --generations 4 \
  --population-size 16 \
  --thermal-iters 640 \
  --no-step
```

固定电压诊断示例：

```bash
python -u optimize_3d.py --smoke --no-step --experiment-name full3d_fixed20_diag --fixed-voltage 20
```

可选动作空间/CEM 参数：

```bash
python -u optimize_3d.py \
  --experiment-name mcga_full3d_4090 \
  --action-axial-modes 5 \
  --action-circum-modes 3 \
  --action-cap-radial-modes 4 \
  --action-strategy-channels 4 \
  --global-shape-steps 4 \
  --cem-initial-sigma 1.10 \
  --cem-elite-fraction 0.35 \
  --no-step
```

## 6. 关键产物

- `optimized_full3d.stl`
- `optimized_full3d.stp`（未加 `--no-step` 且 FreeCAD 可用时）
- `topology_evolution_full3d.gif`
- `topology_evolution_full3d.mp4`
- `optimization_history_full3d.csv`
- `run_summary_full3d.json`
- `design_strategy_report_full3d.md`

关键 summary 字段：

- `optimization_target_full3d`
- `strategy_channels_full3d`
- `global_shape_steps_full3d`
- `final_energy_conversion_efficiency_0_3um`
- `energy_conversion_efficiency_ratio`

## 7. 交接索引

- RTX4090 服务器运行说明见 `docs/cloud_train_rtx4090_zh.md`。
- 历史实现、物理口径修正、legacy 删除记录见 `docs/workflow_log.md`。
