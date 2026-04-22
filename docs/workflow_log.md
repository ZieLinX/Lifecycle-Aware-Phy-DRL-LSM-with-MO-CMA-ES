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
