# 进阶任务工作流日志

> 记录本轮 `make_cylinder_great_again` 进阶任务的实现过程、关键决策、验证与交付结果。

## 0. 背景与约束

- 仓库：`D:\TEMP\codec\make_cylinder_great_again`
- 分支：`xzh`
- conda 环境：`mcga_xzh`
- 训练设备：`NVIDIA GeForce RTX 2050`，`torch 2.7.1+cu118`
- 用户要求：
  - 必须先做题目合规性检查
  - 必须在独显上训练
  - 必须完成进阶任务
  - 输出必须包含从初始形状到最终形状的拓扑演化动画

## 1. 本轮完成项

### 1.1 合规性修正

已完成以下硬约束修复：

1. `external_series_resistance` 默认改为 `0.0`，满足 Q23“电压只加在钨两侧”。
2. 增加 `thermal_boundary_mode`，支持 `fixed_room_temp` / `infinite_room_temp`。
3. 在 `config/cylinder_cfg.py` 中引入：
   - `make_training_cfg()`：训练粗网格
   - `make_eval_cfg()`：评估细网格 `160 x 151`
4. 细网格最小离散尺度满足 `dx <= 0.1 mm`。

### 1.2 进阶任务实现

已完成以下进阶功能：

1. **瞬态升温轨迹与“何时决策”**
   - `utils/rated_condition.py` 新增 `simulate_transient_trajectory()`
   - 环境动作扩展为可带 `dwell_time`
   - score 中加入瞬态带内功率项
2. **工艺可行性约束**
   - 新增 `utils/feasibility.py`
   - 引入最小颈部直径、截面比、轴向坡度软惩罚
3. **热-机械耦合软惩罚**
   - 新增 `utils/thermo_mech.py`
   - 引入轴向热应力估算和软惩罚
4. **RL 训练链**
   - 新增 `envs/cylinder_vec_env.py`
   - 新增 `train_rl.py`
   - 新增 `config/rl_games_ppo.yaml`
   - `train_rl.py` 顶部强制 `torch.cuda.is_available()`
5. **导出链**
   - 新增 ring-profile 到 3D 网格导出
   - 新增拓扑演化动画导出 `gif/mp4`

## 2. 特征尺度定义

题目中的“特征尺度变化超过 20% 失效”在本轮实现里采用双层定义：

1. **主特征尺度**
   - 局部等效直径 `d_eq(z) = 2r(z)`
   - 失效判据主值：
     `max_z |d_eq(z) - 5mm| / 5mm`
2. **制造附加尺度**
   - 最小颈部直径 `min_neck_diameter_m`
   - 相邻截面面积比
   - 轴向半径坡度

这样既对齐了题面“20% 变化”要求，也避免只看平均半径而漏掉细颈、陡坡等不可制造形态。

## 3. 环境与依赖核验

### 已核验通过

- `mcga_xzh` 存在
- `torch.cuda.is_available() == True`
- GPU：`NVIDIA GeForce RTX 2050`
- `rl_games`
- `gymnasium`
- `scipy`
- `tensorboard`
- `imageio`
- `matplotlib`

### 本轮新增依赖

- `imageio`
- `matplotlib`

用途：拓扑演化动画导出。

## 4. 验证结果

### 4.1 单元测试

命令：

```bash
conda run -n mcga_xzh python -m unittest discover -s tests -p "test_*.py"
```

结果：

- 共 `11` 项测试
- 全部通过

覆盖内容包括：

1. 静态编译
2. 物理回归
3. 规划器
4. 可行性约束
5. 瞬态求解
6. VecEnv
7. RL smoke end-to-end

### 4.2 RL smoke end-to-end

命令：

```bash
conda run -n mcga_xzh python train_rl.py --smoke
```

结果：

- 设备：`cuda:0`
- 训练链：成功
- checkpoint：成功
- 细网格最终评估：成功
- STL：成功
- STP：成功
- 拓扑演化 GIF：成功

对应正式产物目录：

- `outputs/final_eval/mcga_phy_drl_22-23-25-50`

摘要指标：

- `baseline_voltage_v = 5.0`
- `final_voltage_v = 5.0`
- `baseline_initial_power_w = 612.8871`
- `final_initial_power_w = 499.4379`
- `lifetime_ratio = 0.9968`
- `feature_change_ratio = 0.00506`
- `volume_change_ratio = 6.08e-06`
- `feasible = False`

说明：

1. 端到端训练/评估/导出链已经打通。
2. 该 smoke 结果主要用于验证工程链闭环，而非最终性能最优。

### 4.3 长预算训练尝试

本轮还尝试了更长预算的独显训练（非 smoke），但在 RTX 2050 上实际运行时间明显高于最初预估，且 `rl_games` 在当前环境下日志刷新较差，不适合作为本轮交付基线。因此本轮保留：

1. **代码与训练链的完整实现**
2. **已跑通的 GPU smoke 基线结果**
3. **可直接继续追加更长训练的命令入口**

## 5. 关键决策记录

1. **不使用 IsaacLab / IsaacSim**
   - `mcga_xzh` 环境未装 Isaac
   - RTX 2050 显存有限
   - 改用 `rl_games + 自写 VecEnv + batched rated-condition`
2. **双档网格**
   - 训练用粗网格
   - 最终评估与导出用细网格
3. **动作不直接编辑体素**
   - RL 输出低阶策略场系数
   - 物理敏度组合生成速度场
   - 再用体积投影保证守恒
4. **保留原 planner 路径**
   - `train.py` 不废弃
   - 新增 `train_rl.py` 走 Phy-DRL 路线

## 6. 交付产物

### 代码

- `train_rl.py`
- `envs/cylinder_vec_env.py`
- `utils/feasibility.py`
- `utils/thermo_mech.py`
- `utils/animation.py`

### 结果

- checkpoint：`outputs/rl_runs/mcga_phy_drl_22-23-25-50/nn/*.pth`
- 评估摘要：`outputs/final_eval/mcga_phy_drl_22-23-25-50/run_summary.json`
- 逐步指标：`outputs/final_eval/mcga_phy_drl_22-23-25-50/rollout_metrics.csv`
- STL：`outputs/final_eval/mcga_phy_drl_22-23-25-50/optimized_cylinder.stl`
- STP：`outputs/final_eval/mcga_phy_drl_22-23-25-50/optimized_cylinder.stp`
- 动画：`outputs/final_eval/mcga_phy_drl_22-23-25-50/topology_evolution.gif`

## 7. 接手后新增：非 RL 绑定的混合物理优化路线

### 7.1 用户新判断

用户指出：`100V` 是最大电压，时间上远远不是最佳。因此本轮将主线从“必须依赖强化学习”调整为：

1. RL 保留为可选对照和 smoke 闭环。
2. 新增 `optimize_hybrid.py`，走物理信息混合优化：CEM 全局搜索 + 局部细化 + 细网格可行候选重评估。
3. 输出不仅包含 STL/STP/动画，还包含 `design_strategy_report.md`，用于提炼给人类设计者的启发。

### 7.2 关键修正

1. `100V` 只作为额定工况搜索上限：
   - `min_voltage` 改为 `0.01`
   - 电压粗搜索/细化支持对数网格
   - 取消电流 cap 作为工作模式，`max_current` 仅保留为数值保护
2. 额定工况搜索优先选择可行电压：
   - 若存在满足温度、体积、特征尺度和工艺约束的电压，绝不选不可行高功率点
3. 瞬态时间不再由一个任意 dwell 点代表：
   - 新增 `summarize_transient_selection()`
   - 在瞬态窗口内搜索最佳采样时间
4. 热求解增加单步温升限制，避免显式迭代在窄电压区间数值跳变。
5. Hybrid 最终导出不盲信训练粗网格 best，而是把所有候选 archive 在 `160 x 151` 细网格上重评估，只从寿命比例 `>= 30%` 的候选中选最终设计。

### 7.3 新增代码

- `optimize_hybrid.py`
- `utils/hybrid_optimizer.py`
- `tests/test_hybrid_optimizer.py`

### 7.4 已完成的正式混合优化结果

命令：

```bash
python optimize_hybrid.py --no-step --output-dir outputs\hybrid_runs --experiment-name mcga_hybrid_sota --generations 3 --population-size 10 --num-modes 5 --local-iterations 1 --thermal-iters 640 --seed 13
```

最终产物目录：

- `outputs/hybrid_runs/mcga_hybrid_sota_04-30-02-35`

摘要指标：

- `baseline_voltage_v = 2.8457596`
- `final_voltage_v = 2.8457596`
- `baseline_initial_power_w = 113.01236`
- `final_initial_power_w = 120.88685`
- `initial_power_ratio = 1.06968`
- `average_power_ratio = 1.06722`
- `lifetime_ratio = 0.60922`
- `feature_change_ratio = 0.00579`
- `volume_change_ratio = 0.0`
- `feasible = True`

产物：

- `optimized_cylinder.stl`
- `optimized_cylinder.stp`
- `topology_evolution.gif`
- `topology_evolution.mp4`
- `optimization_history.csv`
- `run_summary.json`
- `design_strategy_report.md`

设计启发：当前细网格可行最优不是大幅拓扑变形，而是中部微收颈 + 两侧弱补偿外鼓。它在寿命仍为初始 60.9% 的前提下，提高约 6.97% 初始 0-3um 净辐射功率。

### 7.5 验证

非 RL 路线测试命令：

```bash
python -m unittest tests.test_exporter_resolution tests.test_feasibility tests.test_hybrid_optimizer tests.test_physics_regression tests.test_planner tests.test_static_compile tests.test_transient tests.test_vec_env
```

结果：

- 共 `13` 项测试
- 全部通过
- 注意：`test_train_rl_smoke` 依赖 `tensorboardX` 的 multiprocessing queue，在 Codex 沙箱内会触发 Windows 权限错误；此前非沙箱运行通过，本轮 hybrid 完成后未重复长跑。

## 8. 接手后新增：把“3D半成品”修复为最小可行真3D代理物理

### 8.1 问题定位

- 现有 `optimize_3d.py` / `utils/hybrid_optimizer_3d.py` 的几何变量是 `r(z,theta)`，但评估阶段会先用 `effective_ring_profile()` 折算回 1D profile，再调用 1D 额定工况搜索与瞬态，导致“3D 只停留在导出/可视化，物理仍是 2D”的偏差。

### 8.2 修复目标（对齐题面与官方澄清）

- 3D 优化必须满足：候选几何为 `r(z,theta)` 且额定工况/瞬态评估 **不再折算 1D**。
- `100V` 仅作为电压搜索上限；端面（接触电极）不计辐射与升华；体积守恒；端面半径固定为 `2.5mm`。

### 8.3 实现要点（最小可行 3D 代理物理）

- 在 `utils/rated_condition.py` 新增：
  - `search_rated_condition_3d_batch()`：对 `V<=100V` 做粗搜+细化，直接在 `r(z,theta)` 上求稳态指标。
  - `simulate_transient_trajectory_3d()`：3D 瞬态升温轨迹。
- 代理物理假设：
  - 电学：每个 `theta` 方向是一条轴向串联通道；各通道并联、共享端电压。
  - 热学：每个 `theta` 通道做 1D 轴向导热（不做周向导热耦合，后续可增强）。
  - 辐射/升华：仅统计侧壁自由表面 patch（不包含端面）。

### 8.4 接线与兼容

- `utils/hybrid_optimizer_3d.evaluate_radius_fields()` 改为调用 3D rated-condition/瞬态；并在做“最佳瞬态采样时间”选择前，将 3D 温度按 `theta` 聚合为 `per-ring peak` 以兼容 `summarize_transient_selection()` 的张量形状假设。

### 8.5 验证

- 新增单测：`tests/test_hybrid_optimizer_3d.py`
- 通过命令（conda 环境 `mcga_xzh`）：

```bash
conda run -n mcga_xzh python -m unittest discover -s ./tests -p "test_hybrid_optimizer_3d.py"
```

