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

### 8.1 问题定位（历史；已修复）

- **曾存在的问题**：`optimize_3d.py` / `utils/hybrid_optimizer_3d.py` 的几何变量是 `r(z,theta)`，但评估阶段曾用 `effective_ring_profile()` 折算回 1D profile，再调用 1D 额定工况搜索与瞬态，导致“3D 只停留在导出/可视化，物理仍是 2D”的偏差。
- **当前状态**：评估链已改为 `search_rated_condition_3d_batch()` + `simulate_transient_trajectory_3d()`，不再折算 1D；详见仓库提交 `71b632b` / `8a02926` 及单测 `tests/test_hybrid_optimizer_3d.py`。

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

## 9. Agent 交接说明（RL 训练 vs 真 3D 优化、产物与已知现象）

本节供后续 agent 快速对齐口径，**不要求重复踩坑**；与代码实现细节以仓库为准。

### 9.1 两条几何/优化主线（务必区分）

| 主线 | 入口 / 核心模块 | 几何表示 | 物理评估 |
| --- | --- | --- | --- |
| **Phy-DRL（当前 RL）** | `train_rl.py`、`envs/cylinder_vec_env.py` | **轴对称 ring profile**：`ring_radius` 形状 `(num_envs, num_rings)`，即仅随轴向 `z` 变化的一维剖面 | `search_rated_condition_batch()`、`simulate_transient_trajectory()` 等 **1D/轴对称** 路径 |
| **真 3D 表面优化（混合 CEM）** | `optimize_3d.py`、`utils/hybrid_optimizer_3d.py` | **半径场** `r(z, theta)`，形状 `(num_rings, num_segments)` | `search_rated_condition_3d_batch()`、`simulate_transient_trajectory_3d()` |

**结论**：在 `CylinderVecEnv` 上跑的 RL **不是** “完整三维表面 \(r(z,\theta)\) 自由度”的训练；若题目或交付要求明确要 **3D 非轴对称** 优化，应走 `optimize_3d.py` 路线（或未来将 VecEnv 升级为 `r(z,θ)` 并统一 3D rated-condition，属较大改造）。

### 9.2 为什么 `topology_evolution.gif` 看起来像「平面图」

- 动画由 `utils/animation.py` 的 `export_topology_evolution_animation()` 生成：使用 matplotlib 绘制 **轴向位置 `z` vs 半径 `r(z)`** 的子午剖面（`fill_between` + 上下对称 `±r`），**不是** 3D mesh 旋转体渲染。
- 因此即使用 GPU 训练、即使物理在 GPU 上算，**GIF 仍是 2D 剖面可视化**，不代表「训练在 3D mesh 空间里画三角面」。

### 9.3 为什么剖面「几乎不变」仍可能正常

- `CylinderVecEnv._apply_action()` 中半径更新带 **`max_depth * 0.35`** 等小步缩放，且端面环固定、再经 `project_connected_profile_batch` 约束，**可视化尺度上可能极不明显**。
- 判断是否真在动：看 `rollout_metrics.csv` / `run_summary.json` 中的 `feature_change_ratio`、`volume_change_ratio`、功率与寿命字段，而非仅凭 GIF 肉眼。

### 9.4 控制台 `life` 与「两个簇」（~1000 vs ~1）

- 训练日志里打印的 `life` 一般为 **相对基准寿命的比值**（`lifetime_s / baseline_lifetime_s`），**不是「年」**。
- 代理物理下某些电压/几何组合蒸发极弱会导致 **寿命极大**，比值出现数百上千与接近 1 的样本并存，叠加策略探索，易形成 **双峰/两簇**；需结合 `feasible`、`P0-3`、`V*` 一起看，避免误读为「两个物理世界」。

### 9.5 `info/kl` 长期高于 `kl_threshold`（`config/rl_games_ppo.yaml` 中为 `0.01`）

- `kl_threshold` 存在于 rl_games 配置中；**日志中的 KL 为监控量**，是否等价于「每步已严格把更新压到阈值内」取决于 rl_games 版本与内部实现。
- 本环境奖励来自重物理评估，**非平稳、尺度大** 时早期 KL 偏高较常见；是否构成问题应结合 **学习率是否自适应下降、回报曲线、是否发散** 判断，不能仅凭 KL 数值单独定性。

### 9.6 GIF 上 `V* = 0.00V` 与 log 不一致（已修复）

- `utils/animation.py` 中 `_render_frame()` 叠加文字使用键名 **`rated_voltage_v`**；而 rollout / metrics 字典中额定电压字段为 **`voltage_v`**。
- 键名不匹配时 `get` 默认得到 `0.0`，故 GIF 上显示 **`0.00V`**，**不代表** 物理上电压为 0；以 log 与 `rollout_metrics.csv` 中 `voltage_v` 为准。
- 本轮已改 `utils/animation.py`：动画读取指标时优先使用 `voltage_v`，并兼容旧键 `rated_voltage_v`；`P0-3`、`life`、`feasible` 也做了多键 fallback，以兼容 RL 与 3D 优化产物。

### 9.7 RTX4090 云端训练与实时输出（参考）

- 推荐使用仓库内 `docs/cloud_train_rtx4090_zh.md`（SSH、conda、tmux、产物路径）。
- 训练脚本 `train_rl.py` 支持 `--console-interval`（每 N RL step 打印 `V*`/`P0-3`/寿命比/feasible）、`--realtime-interval`（形状快照）、可选 `--torch-compile`（启用 rl_games 侧 `torch_compile`，具体效果依环境而定）。
- **日志缓冲**：云端建议 `python -u train_rl.py`；若 `conda run` 输出异常，可直接使用 conda 环境内 `python.exe` 路径执行（详见 `docs/cloud_train_rtx4090_zh.md`）。

### 9.8 性能瓶颈备忘（给后续优化 agent）

- 单核 CPU 高占用 + GPU 低占用时，常见瓶颈是 **大量小 kernel 发射**（如热迭代 `for` 内多算子）与 **Python 侧循环**，而非「显卡不够强」。
- 已在 `utils/rated_condition.py` 等对 **feasibility / thermomech** 做批量向量化以降低 per-env Python 循环开销；若仍瓶颈，需 profiling 后再决策（例如是否对热迭代体做进一步融合，属改动面较大的工作）。

### 9.9 文档与仓库状态

- 工作流与实验记录以 **`docs/workflow_log.md`**（本文件）与 **`docs/research_solution_zh.md`** 为主。
- 题目附件与 Q&A：`docs/研究背景与要求.png`、`docs/opencode.txt.xlsx`（勿提交临时导出的 `docs/_xlsx_export/`，已在 `.gitignore` 忽略）。

## 10. 本轮接手：复盘 RTX4090 训练异常并修正 RL 评估/可视化

### 10.1 输入与本地 git 状态

- 用户提供 RTX4090 训练产物目录：`RTX4090/`。
- 本轮接手时分支为 `xzh`；工作区已有未提交文档变更：
  - `docs/workflow_log.md`
  - `docs/research_solution_zh.md`
- `RTX4090/` 为用户放入的未跟踪训练日志/产物目录，本轮只读取分析，不纳入提交。

### 10.2 对五个严重问题的结论

1. **为什么生成的 GIF 是平面图？是否真 3D？**
   - RTX4090 运行的是 `train_rl.py` / `CylinderVecEnv`，几何自由度是轴对称 `ring_radius(z)`，不是完整 `r(z,theta)` 三维表面。
   - `topology_evolution.gif` 本来就是 z-r 子午剖面可视化，不是 3D mesh 渲染。
   - 若要真 3D 非轴对称优化，应运行 `optimize_3d.py` 路线。
2. **为什么圆柱面没有发生过变化？**
   - RTX4090 最终评估实际只执行了 2 步；`run_summary.json` 中 `feature_change_ratio=0.025808`，说明几何有变化但只有约 2.58% 特征尺度变化。
   - 最终 `lifetime_ratio=0.267119`，低于 `minimum_lifetime_ratio=0.30`，因此评估提前终止；肉眼看 GIF 会显得变化很小。
3. **为什么 `info/kl` 始终高于阈值？**
   - TensorBoard event 中 `info/kl` 最小约 `0.00886`、最大约 `0.47123`、最后约 `0.01702`，确实经常高于 `kl_threshold=0.01`。
   - 同一 event 中 `info/last_lr` 最终降到 `1e-6`，说明 rl_games 的 adaptive LR 已在响应 KL 偏高；问题不只是“阈值没生效”，还包括奖励尺度和极端寿命样本导致策略更新不稳定。
4. **为什么模型像有两个方向（life ~1000 与 life ~1）？**
   - 日志显示大量 `V*=2.51` 或接近 `0.01V` 的低功率样本有 `life` 数百到近千；同时 `V*=5.75~6.63V` 的样本功率更高但 `life` 常接近 1 或低于 1。
   - 原奖励直接线性使用未截断的 `lifetime_ratio`，容易让低功率超长寿命样本与高功率短寿命样本形成两簇。
5. **为什么 GIF 中 `V*` 始终为 `0.00V`，但 log 正常？**
   - 确认为显示层键名 bug：最终评估 `metrics_history` 用 `voltage_v`，动画只读 `rated_voltage_v`。
   - 本轮已修复为优先读 `voltage_v`，兼容 `rated_voltage_v`。

### 10.3 本轮代码修正

- `utils/animation.py`
  - 新增 `_metric_float()` / `_metric_value()`，统一动画指标 fallback。
  - `V*` 优先读 `voltage_v`，不再在最终 GIF 中错误显示 `0.00V`。
  - 标题改为 `Axisymmetric Ring-Profile Evolution`，避免把 RL GIF 误解为真 3D。
- `config/cylinder_cfg.py`
  - 新增 `reward_lifetime_ratio_cap=5.0` 与 `observation_lifetime_ratio_cap=5.0`。
- `envs/cylinder_vec_env.py`
  - 奖励中的 `lifetime_ratio` 改为只在奖励项内截断，不改原始物理指标。
  - 观测中的寿命比也做截断，降低超长寿命低功率样本对策略的支配。
- `train_rl.py`
  - 最终评估的 baseline/final 动画指标补 `lifetime_ratio` 与 `geometry_mode`。
  - `run_summary.json` 新增 `geometry_mode`、`physics_mode`、`minimum_lifetime_ratio`、`rated_feasible`、`constraint_feasible`、`termination_reasons`、`max_radius_delta_mm`、`mean_abs_radius_delta_mm`。
  - `feasible` 改为完整约束可行性，避免仅额定工况 `feasible=true` 掩盖 `lifetime_ratio < minimum_lifetime_ratio`。

### 10.4 验证

本轮已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_animation tests.test_vec_env tests.test_static_compile
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_animation tests.test_exporter_resolution tests.test_feasibility tests.test_hybrid_optimizer tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_planner tests.test_static_compile tests.test_transient tests.test_vec_env
```

结果：第一次 5 项测试通过；第二次 19 项非长训练测试通过。

## 11. 本轮追问：确认真 3D GIF/MP4 与更新 RTX4090 Ubuntu 执行文档

### 11.1 用户问题

- 用户追问：现在是否能够生成 3D 的 GIF 和 MP4。
- 用户要求：让子 agent 在原文档基础上写一份 RTX4090 Ubuntu 服务器执行训练/优化的文档。

### 11.2 结论

- `train_rl.py` 路线仍是轴对称 `ring_radius(z)` RL；它生成的 `topology_evolution.gif/.mp4` 是 z-r 剖面，不是真 3D。
- `optimize_3d.py` 路线是真 3D 半径场 `r(z, theta)` 优化；它调用 `export_3d_evolution_animation()`，输出：
  - `topology_evolution_3d.gif`
  - `topology_evolution_3d.mp4`（取决于 imageio/ffmpeg/libx264 是否可用）

### 11.3 本地 smoke 验证

已执行：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --device cpu --smoke --no-step --output-dir outputs\test_three_d_runs --experiment-name smoke_3d_doccheck --generations 1 --population-size 4 --thermal-iters 16 --seed 7
```

结果目录：

- `outputs/test_three_d_runs/smoke_3d_doccheck_05-02-17-55`

已确认生成：

- `optimized_cylinder_3d.stl`
- `topology_evolution_3d.gif`
- `topology_evolution_3d.mp4`
- `run_summary_3d.json`
- `optimization_history_3d.csv`
- `design_strategy_report_3d.md`

该 smoke 用 CPU 和极小参数，仅验证导出链，不代表优化质量。4090 Ubuntu 上应使用 `--device cuda:0` 和更高 `generations/population-size/thermal-iters`。

## 12. RTX4090 smoke 报错修复：兼容旧 NumPy 无 `np.trapezoid`

### 12.1 用户反馈

用户在 RTX4090 Ubuntu 服务器先跑 `train_rl.py --smoke`，启动阶段在 `utils/rated_condition.py` 的黑体光谱带积分处失败：

```text
AttributeError: module 'numpy' has no attribute 'trapezoid'
```

根因：服务器环境 NumPy 版本较旧，提供 `np.trapz` 但没有较新的 `np.trapezoid` API。

### 12.2 修复

- 在 `utils/rated_condition.py` 新增 `_trapezoid_integral()`：
  - 优先使用 `np.trapezoid`
  - 旧 NumPy 回退到 `np.trapz`
  - 如果两者都不存在，再使用本地梯形积分实现
- 将 `_blackbody_band_fraction_cached()` 中两处 `np.trapezoid(...)` 改为 `_trapezoid_integral(...)`。
- 在 `tests/test_physics_regression.py` 增加旧 NumPy 兼容测试，模拟 `np.trapezoid` / `np.trapz` 都不可用时仍能计算黑体带积分。

### 12.3 验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_physics_regression tests.test_static_compile tests.test_vec_env
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_animation tests.test_exporter_resolution tests.test_feasibility tests.test_hybrid_optimizer tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_planner tests.test_static_compile tests.test_transient tests.test_vec_env
```

结果：第一次 8 项通过；第二次 20 项非长训练测试通过。

## 13. RTX4090 Ubuntu smoke 通过记录

### 13.1 RL smoke

用户在 RTX4090 Ubuntu 服务器执行：

```bash
python -u train_rl.py --smoke --experiment-name mcga_phy_drl_4090_smoke
```

结果：

- CUDA 设备：`NVIDIA GeForce RTX 4090`
- 训练完成并保存 checkpoint：
  - `outputs/rl_runs/mcga_phy_drl_4090_smoke_02-20-38-01/nn/last_mcga_phy_drl_4090_smoke_ep_1_rew__30.59399_.pth`
- 训练过程 GIF 已生成：
  - `outputs/rl_runs/mcga_phy_drl_4090_smoke/realtime/training_evolution.gif`
- final eval 已完成：
  - `outputs/final_eval/mcga_phy_drl_4090_smoke_02-20-38-01`

说明：该路线仍为轴对称 RL，用于 smoke 和对照，不是真 3D 优化。

### 13.2 真 3D smoke

用户在 RTX4090 Ubuntu 服务器执行：

```bash
python -u optimize_3d.py --smoke --no-step --experiment-name mcga_3d_4090_smoke
```

结果目录：

- `outputs/three_d_runs/mcga_3d_4090_smoke_05-02-20-39`

关键结果：

- `optimized_cylinder_3d.stl` 已生成
- `topology_evolution_3d.gif` 已生成
- `run_summary_3d.json` / `optimization_history_3d.csv` / `design_strategy_report_3d.md` 已生成
- `stp=None`：符合预期，因为命令使用了 `--no-step`
- `mp4=None`：说明当前服务器视频编码器链路未写出 MP4；不影响 3D 优化和 GIF，可安装/修复 `ffmpeg`、`imageio-ffmpeg` 后重跑或后处理

smoke 指标显示 baseline 与 final 几乎相同、`selected_archive_index=0`，这是 smoke 小种群/短跑预期现象，不代表正式 3D 优化质量。

## 14. FreeCAD STEP smoke 卡住处理

### 14.1 用户反馈

用户在 RTX4090 Ubuntu 服务器执行：

```bash
python -u optimize_3d.py --smoke --experiment-name freecad_check_step
```

现象：日志停在 `[3d] device: NVIDIA GeForce RTX 4090` 后，GPU 负载未起来，一个 CPU 进程持续高占用，十几分钟不结束。

判断：3D 优化 smoke 本身此前已能完成；这次没有 `--no-step`，卡住点高度可能是 FreeCAD CLI 在 STL -> STEP 转换阶段的 `makeShapeFromMesh` / `Part.makeSolid`。

### 14.2 修复

- `utils/exporter.py`
  - FreeCAD subprocess 增加 `timeout`。
  - 超时后打印提示并返回 `stp=None`，保留 STL/GIF/JSON 等产物，不再无限等待。
- `config/cylinder_cfg.py`
  - 新增 `freecad_timeout_s = 90.0`。
- `optimize_3d.py`
  - 新增命令行参数 `--freecad-timeout`。
- `train_rl.py`
  - final eval STEP 导出同样支持 `--freecad-timeout`。
- `docs/cloud_train_rtx4090_zh.md`
  - 补充 FreeCAD 卡住处理：正式优化继续建议 `--no-step`，STEP 单独用短超时检查。

### 14.3 建议服务器操作

当前卡住进程先停止：

```bash
Ctrl+C
pkill -f FreeCADCmd || true
pkill -f optimize_3d.py || true
```

拉最新代码后检查 STEP：

```bash
git pull origin xzh
python -u optimize_3d.py --smoke --experiment-name freecad_check_step --freecad-timeout 20
```

若 20 秒内 FreeCAD 未完成，脚本应超时返回，`stp` 为 `None`，但 STL/GIF 仍保留。

### 14.4 验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_exporter_resolution tests.test_static_compile tests.test_hybrid_optimizer_3d
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_animation tests.test_exporter_resolution tests.test_feasibility tests.test_hybrid_optimizer tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_planner tests.test_static_compile tests.test_transient tests.test_vec_env
```

结果：第一次 6 项通过；第二次 21 项非长训练测试通过。

### 14.5 RTX4090 服务器验证

用户在 RTX4090 Ubuntu 服务器执行：

```bash
python -u optimize_3d.py --smoke --experiment-name freecad_check_step --freecad-timeout 20
```

结果符合预期：

- FreeCAD STEP 转换在 20 秒后超时返回：
  - `[export] FreeCAD STEP conversion timed out after 20.0s; STL export is still valid.`
- 脚本没有继续卡死，正常结束。
- 输出目录：
  - `outputs/three_d_runs/freecad_check_step_05-02-21-39`
- 已生成：
  - `optimized_cylinder_3d.stl`
  - `topology_evolution_3d.gif`
  - `run_summary_3d.json`
  - `optimization_history_3d.csv`
  - `design_strategy_report_3d.md`
- `stp=None`：FreeCAD 超时后的预期结果。
- `mp4=None`：仍是视频编码器链路问题，不影响 GIF 和核心 3D 产物。

## 15. RTX4090 正式 3D 命令参数缺值记录

### 15.1 用户反馈

用户执行正式 3D 优化命令时写成：

```bash
python -u optimize_3d.py --experiment-name mcga_3d_4090 --output-dir outputs/three_d_runs --generations 4 --population-size 16 --axial-modes 4 --circum-modes 2 --thermal-iters 2>&1 | tee -a logs/train_$(date +%F_%H%M%S).log
```

报错：

```text
optimize_3d.py: error: argument --thermal-iters: expected one argument
```

原因：`--thermal-iters` 后面必须跟整数；当前命令里 `--thermal-iters` 后直接进入 shell 重定向 `2>&1`，所以 argparse 认为该参数缺值。

### 15.2 推荐命令

正式 3D 优化建议继续跳过 STEP，使用：

```bash
mkdir -p logs
python -u optimize_3d.py \
  --experiment-name mcga_3d_4090 \
  --output-dir outputs/three_d_runs \
  --generations 4 \
  --population-size 16 \
  --axial-modes 4 \
  --circum-modes 2 \
  --thermal-iters 640 \
  --no-step \
  2>&1 | tee -a logs/train_$(date +%F_%H%M%S).log
```

## 16. 3D 正式优化选回 baseline 的诊断与修复

### 16.1 用户反馈

用户在 RTX4090 Ubuntu 服务器执行正式 3D 路线：

```bash
python -u optimize_3d.py \
  --experiment-name mcga_3d_4090 \
  --output-dir outputs/three_d_runs \
  --generations 4 \
  --population-size 16 \
  --axial-modes 4 \
  --circum-modes 2 \
  --thermal-iters 640 \
  --no-step \
  2>&1 | tee -a logs/train_$(date +%F_%H%M%S).log
```

服务器结果显示：

- `selected_archive_index=0`
- `initial_power_ratio_3d=1.0`
- `lifetime_ratio_3d=1.0`
- `feature_change_ratio_3d=0.0`
- `surface_area_ratio=1.0`

这说明 3D 优化链路和 GIF/STL 导出已经运行，但细网格复评最终选回了初始圆柱，正式跑没有产出非基准形变。

### 16.2 本地排查结论

- 首代种群中存在重复零候选，archive 会把与 baseline 完全相同的候选作为非基准记录，干扰诊断。
- 旧 3D rated-condition/transient 的辐射和蒸发侧面积仍近似使用 `r * dz * dtheta`，没有把 `dr/dz` 和 `dr/dtheta` 带来的曲面面元面积纳入物理计算。
- `run_summary_3d.json` 只写最终选择，没有写最佳非基准候选、可行数量、最大表面积变化等诊断，因此服务器上只能看到 `selected_archive_index=0`，看不到非基准候选输在哪里。

### 16.3 修改内容

- `utils/rated_condition.py`
  - 新增 3D 曲面面元计算：`sqrt(r^2(1+(dr/dz)^2)+(dr/dtheta)^2) dz dtheta`。
  - 3D rated-condition 和 transient 的辐射、蒸发功率现在使用真实曲面侧面积。
  - 输出 `surface_area_ratio_3d` 作为底层物理诊断。
- `utils/hybrid_optimizer_3d.py`
  - `evaluate_radius_fields()` 不再对功率二次乘 `surface_gain`，避免真实面积进入物理后重复补偿。
  - 放开默认 CEM 搜索幅度：`hybrid3d_initial_sigma=0.075`、`hybrid3d_max_sigma=0.140`、`hybrid3d_max_log_delta=0.160`。
  - 增强 3D 种子，加入更明显的轴向和周向模式。
  - 首代种群去重，不再重复塞零候选。
  - 3D 报告写入选择原因和 archive 诊断。
- `optimize_3d.py`
  - 新增 CLI 参数：`--initial-sigma`、`--min-sigma`、`--max-sigma`、`--max-log-delta`、`--circum-penalty`。
  - `run_summary_3d.json` 新增：
    - `selection_reason_3d`
    - `archive_candidate_count`
    - `archive_feasible_count`
    - `archive_feasible_nonbaseline_count`
    - `best_nonbaseline_by_score`
    - `best_feasible_nonbaseline_by_initial_power`
    - `max_nonbaseline_surface_area_ratio`
- `tests/test_hybrid_optimizer_3d.py`
  - 新增回归测试，确认有斜率的 3D 半径场会得到 `surface_area_ratio_3d > 1.0`。
- `docs/cloud_train_rtx4090_zh.md`
  - 追加服务器复跑和诊断字段查看方法。

### 16.4 验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_static_compile
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_hybrid_optimizer_3d
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_static_compile
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --device cpu --smoke --no-step --experiment-name local_diag_smoke
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_animation tests.test_exporter_resolution tests.test_feasibility tests.test_hybrid_optimizer tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_planner tests.test_static_compile tests.test_transient tests.test_vec_env
```

结果：

- 3D/physics/static 组合测试 10 项通过。
- 本地 CPU 3D smoke 正常生成 STL/GIF/MP4，并在 `run_summary_3d.json` 输出 `selection_reason_3d`、`best_feasible_nonbaseline_by_initial_power`、`max_nonbaseline_surface_area_ratio` 等诊断字段。
- 非长训练全量单元测试 22 项通过。
