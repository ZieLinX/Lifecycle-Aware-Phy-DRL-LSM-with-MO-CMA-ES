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

## 17. 体积守恒、形变动作与 Phy-DRL-LSM 思路评估

### 17.1 当前体积守恒状态

- 3D hybrid 路线 `optimize_3d.py` / `utils/hybrid_optimizer_3d.py`：
  - `_project_radius_field()` 会固定两端电极环，对内部 `r(z, theta)` 做 slope/connectivity 限制，并按内部体积缩放半径。
  - 目标体积为初始圆柱体积，当前 smoke 中 `volume_change_ratio_3d` 约为 `1e-7` 量级，属于数值误差。
  - 注意：若 `min_radius`、端点固定、slope 限制冲突，理论上仍可能只保证在容差内；当前仍用 `volume_tolerance_ratio=0.02` 做最终约束。
- 2D hybrid 路线 `optimize_hybrid.py` / `utils/hybrid_optimizer.py`：
  - `_project_volume_and_connectivity()` 会固定端点、投影连通性/斜率，并缩放内部半径使体积接近初始体积。
- RL 训练路线 `train_rl.py` / `envs/cylinder_vec_env.py`：
  - action 输出三组策略场系数，组合成轴向速度场。
  - `_project_volume_preserving_velocity()` 做一阶拉格朗日式速度投影：`sum(2*pi*r*dz*velocity)=0`。
  - 之后还会 clamp 最小半径、固定端点、投影连通性；这些后处理可能重新引入小体积误差。
  - 当前依靠 `volume_change_ratio` 惩罚和终止条件兜底，不是每步最终半径的严格体积重投影。
- 旧 `CylinderPhysicsEnv` 点阵路线：
  - 动作是局部 dent + 远场 compensation，并调用 `_enforce_volume_conservation()` 做近似体积修正。
  - 后续 connected-profile 投影可能再次改变体积，因此也是容差/惩罚保证，不是严格数学等式保证。

### 17.2 当前形变动作

- RL 训练实际使用 `CylinderVecEnv`：
  - `action_dim = 6 * 3 + 1 = 19`。
  - 前 18 维 reshape 为 3 组、每组 6 个 Chebyshev 低阶系数。
  - 三组策略场分别是 radiation / evaporation / current 权重场，经 sigmoid 映射为 `alpha_rad(z)`、`alpha_evap(z)`、`alpha_cur(z)`。
  - 物理敏度为 `rad=temp_norm^4`、`evap=recession_rate_norm`、`cur=1/r^2`。
  - 几何速度：`raw_velocity = alpha_rad*rad - alpha_evap*evap + alpha_cur*cur`，再体积投影、缩放并更新 `ring_radius(z)`。
  - 最后一维 action 为 dwell，用于 transient 采样时间策略。
- 3D hybrid 路线：
  - 不是 RL action，而是 CEM 搜索 Chebyshev axial modes 与 Fourier circum modes 的系数。
  - 几何是 `r(z, theta)`，支持非轴对称 3D 起伏；候选经 `_project_radius_field()` 投影到体积/端点/斜率约束空间。
- 旧点阵环境：
  - action 为 `[index_ratio, indentation, sigma]`，可选第四维 dwell。
  - 表示在圆柱表面某一点做局部凹陷，并在远场补偿隆起。

### 17.3 对 Phy-DRL-LSM 方案的借鉴判断

- 已经可以直接借鉴：
  - “RL 不直接编辑体素，而输出物理策略权重场”这一点已经和当前 `CylinderVecEnv` 一致，但当前是一维轴向场。
  - 体积守恒拉格朗日投影可升级为 action 后最终半径重投影，弥补 clamp/connectivity 后的体积漂移。
  - 输出 `alpha_rad/alpha_evap/alpha_cur` 的历史可做可解释性产物，解释何时、何地、为何增材/减材。
- 适合下一阶段实现：
  - 把 `CylinderVecEnv` 从 `r(z)` 扩展到 `r(z, theta)`，复用当前 3D hybrid 的 basis/projection/physics。
  - 在 3D action 中使用低阶二维基函数输出三个策略场，而不是直接上体素级 3D U-Net。
  - 每次 action 后调用和 3D hybrid 类似的最终体积投影，保证最终几何体积在数值误差内不变。
- 暂不建议直接上：
  - 完整体素 SDF + level-set + 3D U-Net/MinkowskiEngine/GNN surrogate。实现量很大，依赖重，且当前圆柱任务的有效自由度未必需要完整拓扑优化。
  - 严格 0.1 mm 体素化会让 5 mm x 15 mm 圆柱网格较粗，容易引入台阶误差；当前 `r(z, theta)` 曲面参数化更适合 STL/GIF/物理求解闭环。

### 17.4 建议路线

短期优先做“3D Phy-DRL low-order strategy fields”：

1. 新增 `Cylinder3DVecEnv`，状态为 `r(z, theta)`、温度场、蒸发率、view factor proxy。
2. action 输出三组低阶 2D basis 系数：`alpha_rad(z,theta)`、`alpha_evap(z,theta)`、`alpha_cur(z,theta)`，另加 dwell。
3. 用当前 3D rated-condition 和 transient 作为内环额定工况搜索。
4. action 后做最终体积/连通/斜率投影，并把 `volume_change_ratio_3d` 控制到 `1e-5` 或更低。
5. 导出 alpha 权重云图/GIF，作为可解释性报告的一部分。

### 17.5 需要修正的题目规则

- 当前 3D hybrid 仍固定 `r[:,0,:]` 和 `r[:,-1,:]` 为 2.5mm，等价于把两端截面都固定为圆形；这不符合用户新要求“圆柱底面和顶面也可以被改变，只要求电极一定是 5mm 圆形”。
- 下一阶段应把“电极”从“整张顶/底圆面固定”改为“外部电路连接点/连接环保持直径 5mm 和相对位置不变”，而优化体可以在靠近端面的非电极区域改变形状。
- 当前 `r(z,theta)` 表示的是沿 z 轴单值半径场，天然不支持真正任意拓扑、端面起伏或内腔；若要允许顶/底面自由变形，至少要升级为端面可变的封闭 star-shaped surface，或转向 SDF/level-set 表示。
- 题目中的外接球应建模为 0K、发射率 1 的吸收面；有效辐射目标应是能从器件表面到达外接球、处于 0-3 微米波段的净辐射功率。当前代码使用 `view_factor_proxy` 和遮挡 proxy，不是严格外接球可见性积分。
- 最佳可落地方案应先做“封闭曲面 + 低阶 3D 策略场 + 体积投影 + 外接球可见性采样”，再考虑完整体素 SDF/level-set。

## 18. full3d 后端：真 3D 物理规则修正

### 18.1 用户要求

用户明确要求修正所有物理规则，底线是必须真 3D：

- 体积：通电前材料体积应固定。
- 外界：外接球视为 0K、发射率 1 的吸收面。
- 电极：电极必须保持 5mm 圆形。
- 几何：顶面和底面也可以改变，不能只优化侧面。
- 策略：上完整 3D U-Net/GNN 思路。

### 18.2 实现内容

- 新增 `utils/full3d_optimizer.py`。
  - 使用封闭三维 mesh 表示几何。
  - 侧面、顶面、底面都作为可优化表面。
  - 两端 5mm 圆形电极边界固定直径和相对位置。
  - `project_full3d_geometry()` 以闭合 mesh 体积为准，把通电前体积投影回初始圆柱体积。
  - `evaluate_full3d_geometry()` 统计 0K、发射率 1 外接球吸收的 0-3 微米净辐射功率。
  - 增加 `Full3DUNetGNNPolicy`：轻量 3D U-Net 编码器 + 图邻域平滑头，用于生成策略场。
  - 导出 `optimized_full3d.stl/.stp`、`topology_evolution_full3d.gif/.mp4`、`run_summary_full3d.json`、`design_strategy_report_full3d.md`。
- `optimize_3d.py`
  - 新增 `--backend full3d|sidefield`，默认 `full3d`。
  - `sidefield` 保留旧 `r(z,theta)` 路线以便对照。
  - 新增 `--fixed-voltage`、`--cap-rings`、`--full3d-volume-tolerance`、`--full3d-neural-policy/--no-full3d-neural-policy`。
- `config/cylinder_cfg.py`
  - 增加 full3d 体积、电极、外接球和策略模型相关配置。
- `tests/test_full3d_optimizer.py`
  - 检查封闭 mesh 体积、电极误差、0K 外接球、U-Net/GNN 策略模型。
  - 检查优化后顶/底面非电极区域确实发生位移，防止退回只动侧面的假 3D。
- `tests/test_static_compile.py`
  - 纳入 `utils/full3d_optimizer.py` 编译检查。
- `docs/cloud_train_rtx4090_zh.md`
  - 增加 full3d 默认后端和 RTX4090 命令。

### 18.3 本地验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_static_compile tests.test_full3d_optimizer
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --backend full3d --device cpu --smoke --no-step --experiment-name full3d_smoke_check
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_full3d_optimizer tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_static_compile
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --backend sidefield --device cpu --smoke --no-step --experiment-name sidefield_compat_smoke
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_animation tests.test_exporter_resolution tests.test_feasibility tests.test_full3d_optimizer tests.test_hybrid_optimizer tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_planner tests.test_static_compile tests.test_transient tests.test_vec_env
```

full3d smoke 结果：

- 生成 `optimized_full3d.stl`
- 生成 `topology_evolution_full3d.gif`
- 本地环境生成 `topology_evolution_full3d.mp4`
- `volume_change_ratio_full3d` 约 `2.8e-8`
- `electrode_max_error_m` 约 `4e-19`
- 顶/底面非电极区域发生位移
- 100V 固定电压下当前候选温度超限，`feasible=False`，这是物理约束诊断，不再误标为可行
- full3d/sidefield/physics/static 组合测试 13 项通过。
- 旧 `--backend sidefield` CPU smoke 兼容通过。
- 非长训练全量单元测试 25 项通过。

## 19. full3d 固定电压温度诊断修复

### 19.1 用户反馈

用户在 RTX4090 Ubuntu 服务器执行：

```bash
python -u optimize_3d.py \
  --backend full3d \
  --smoke \
  --no-step \
  --experiment-name full3d_v_sweep \
  --fixed-voltage 20
```

输出仍显示 `max_temperature_k=4909.725`，与之前 100V smoke 完全相同；这说明 `--fixed-voltage` 虽然进入了配置，但 full3d 温度报告被硬裁剪掩盖了电压变化。

### 19.2 修复内容

- `utils/full3d_optimizer.py`
  - 移除 full3d 温度 `clip(max_temp * 1.5)`，改为求解固定电压下的未裁剪热平衡温度。
  - 电阻随温度按钨电阻率系数更新，热平衡包含全谱辐射冷却和蒸发潜热项。
  - 0-3 微米到 0K 外接球的净辐射仍作为优化目标，新增 `blackbody_band_fraction_0_3um`、`electrical_power_w`、`full_spectrum_radiative_power_w`、`thermal_balance_residual_w` 等诊断。
  - 候选选择改为优先选择满足体积、电极、温度、寿命约束的可行候选；若固定电压下全 archive 不可行，则明确写出 `selection_reason_full3d`。
  - `Full3DResult` 增加 `archive_metrics` 和 `selection_diagnostics`，便于服务器端复盘。
  - full3d MP4 帧尺寸改为 960x608，避免 imageio 因宏块尺寸自动 resize 的警告。
- `optimize_3d.py`
  - `run_summary_full3d.json` 新增 `fixed_voltage_v`、`baseline_feasible_full3d`、`archive_feasible_count`、`selected_archive_index`、`selection_reason_full3d`、`best_archive_by_score` 等字段。
- `tests/test_full3d_optimizer.py`
  - 新增回归测试确认 20V 和 100V 下 full3d 温度诊断不同，且温度随固定电压升高而上升。
  - 保留顶/底面可动性测试，但不再要求最终选择一定选中变形候选，因为选择器现在优先物理可行性。

### 19.3 本地验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_full3d_optimizer tests.test_static_compile
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --backend full3d --device cpu --smoke --no-step --experiment-name full3d_fixed_voltage_check --fixed-voltage 20
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_animation tests.test_exporter_resolution tests.test_feasibility tests.test_full3d_optimizer tests.test_hybrid_optimizer tests.test_hybrid_optimizer_3d tests.test_physics_regression tests.test_planner tests.test_static_compile tests.test_transient tests.test_vec_env
```

验证结果：

- full3d/static 组合测试 6 项通过。
- full3d CPU smoke 正常生成 STL/GIF/MP4/JSON/报告。
- 完整非长训练测试集 27 项通过。
- 本地探针显示固定电压已生效：5V、10V、20V、50V、100V 输出不同温度，20V 不再被报告为原来的裁剪值 `4909.725K`。
- 在当前简化模型下 20V 仍超温，summary 会明确写 `archive_feasible_count=0` 和 `selection_reason_full3d=no full3d candidate satisfied fixed-voltage...`。

## 20. full3d 默认改回额定电压搜索

### 20.1 RTX4090 复测结果

用户在 RTX4090 Ubuntu 服务器拉取最新 `xzh` 后执行：

```bash
python -u optimize_3d.py --backend full3d --smoke --no-step --experiment-name full3d_v_sweep --fixed-voltage 20
```

结果确认：

- `fixed_voltage_v=20.0`，说明参数已生效。
- `max_temperature_k=6292.129150390625`，不再是旧裁剪值 `4909.725K`。
- `archive_feasible_count=0`，固定 20V 工况下所有 full3d 候选都过温/不可行。
- `mp4` 正常生成，且未再出现 imageio macro-block resize 警告。

### 20.2 判断

该结果不是“电压参数失效”，而是固定电压诊断揭示了更核心的物理问题：5mm x 15mm 钨柱电阻极低，若强行固定 20V，焦耳功率达到 `4.247e5 W` 量级，远超 3000 摄氏度约束。仓库既有题意判断和文档均指向：`100V` 是电压上限，正式评估应搜索 `V <= 100V` 的额定工况，而不是把 20V 或 100V 作为固定工作点。

### 20.3 修复内容

- `config/cylinder_cfg.py`
  - `full3d_fixed_voltage_v` 默认从 `100.0` 改为 `None`。
  - `None` 表示 full3d 默认做额定电压搜索；设为浮点数时进入固定电压诊断模式。
- `optimize_3d.py`
  - `--fixed-voltage` 改为可选参数，默认不传。
  - summary 增加 `voltage_search_mode`、`voltage_constraint`、`baseline_voltage_v`、`final_voltage_v`、`rated_voltage_upper_bound_v`。
- `utils/full3d_optimizer.py`
  - `evaluate_full3d_geometry()` 拆为固定电压评估和额定电压搜索两层。
  - 默认对 `min_voltage..max_voltage` 做粗搜与局部细化，优先选择满足温度、寿命、体积、电极约束的可行电压。
  - `--fixed-voltage <V>` 仍保留，用于复查固定电压为什么不可行。
  - `selection_reason_full3d` 区分 fixed-voltage 与 rated-search 两种模式。
- `docs/cloud_train_rtx4090_zh.md`
  - 明确正式 full3d 命令不要加 `--fixed-voltage`。
  - 补充固定电压诊断命令和 `archive_feasible_count=0` 的解释。
- `tests/test_full3d_optimizer.py`
  - 新增默认额定电压搜索回归测试。
  - 保留显式 fixed-voltage 温度响应测试。

## 21. full3d 物理口径补充：电极边界、自由表面与额定搜索

### 21.1 本轮文档修正范围

本轮只做文档口径同步，不修改代码文件，不运行长测试。写入范围限定为：

- `docs/workflow_log.md`
- `docs/cloud_train_rtx4090_zh.md`

### 21.2 full3d 物理边界口径

- 钨棒需要显式计算轴向导热。
- 两端接触铜电极按 `300K` 固定温度边界处理。
- 当前口径忽略钨棒-铜电极之间的接触热阻和接触电阻。
- 电压只加在钨棒两端，不包含铜电极电压降。
- 自由表面计算向 `300K` 环境的净辐射/散热和升华。
- 端面接触电极的区域不计向外辐射，也不计升华；只有自由表面参与这些表面损失和收益。

### 21.3 额定电压搜索口径

- `100V` 是额定搜索上限，不是固定工作电压。
- full3d 正式优化不应在命令里加 `--fixed-voltage 100` 或 `--fixed-voltage 20`。
- 额定搜索应在 `V <= 100V` 的候选稳态中，选择综合辐射收益和蒸发/寿命后的最优稳态。
- 被选中的最优稳态电压必须满足 `<=100V`。
- 初始 5mm x 15mm 圆柱的额定电压约 `0.34V`，可作为口径说明：固定 20V/100V 会显著偏离正式额定搜索含义，通常只适合作为过温诊断。

### 21.4 RTX4090 文档同步

`docs/cloud_train_rtx4090_zh.md` 的 full3d 段落已同步：

- 明确默认 full3d 后端是封闭三维网格，旧 `sidefield` 才是 `r(z, theta)` 半径场。
- 补充轴向导热、300K 铜电极固定温度边界、电压只跨钨棒、端面接触区不计辐射/升华的说明。
- 正式命令保持不加 `--fixed-voltage`，继续让程序搜索 `V<=100V` 的额定工况。
- 固定电压命令仅保留为诊断模式，并明确固定 20V/100V 不是正式优化口径。

## 22. full3d-only 文档收尾：legacy 1D/RL/hybrid/sidefield 已移除

### 22.1 本轮背景

用户已明确允许删除旧 1D/RL 代码和无用 outputs。代码侧在 full3d 物理口径修正后，进一步移除了 legacy 入口、环境、配置、测试、工具和历史输出；文档同步为 full3d-only 口径。

### 22.2 当前正式工作流

当前仓库正式工作流只保留：

- `optimize_3d.py`
- `utils/full3d_optimizer.py`

正式优化路线为 full3d 封闭三维 mesh 后端：

- 不再使用 `train_rl.py` / `CylinderVecEnv` 轴对称 RL 路线。
- 不再使用 `optimize_hybrid.py` / 1D hybrid 路线。
- 不再使用旧 `sidefield` / `r(z, theta)` 半径场路线。
- 不再保留 legacy 1D/RL/hybrid/sidefield 相关配置、测试、环境、工具和历史 outputs 作为当前可运行交付。

### 22.3 文档同步

- `docs/cloud_train_rtx4090_zh.md`
  - 已检查：当前文档已经是 full3d-only 执行文档。
  - 未再给出 `train_rl.py`、`optimize_hybrid.py` 或 `--backend sidefield` 的正式命令。
  - 正式命令保持 `python -u optimize_3d.py ...`，默认 full3d。
- `docs/research_solution_zh.md`
  - 已从旧 Phy-DRL / hybrid 方案说明改写为 full3d-only 方案说明。
  - 删除了旧 `train_rl.py`、`optimize_hybrid.py`、axisymmetric ring profile、hybrid outputs 等当前已失效的运行建议。
  - 保留当前 full3d 物理口径、正式命令、固定电压诊断说明和关键产物列表。

### 22.4 后续阅读口径

`docs/workflow_log.md` 前面的 1D/RL/hybrid/sidefield 内容是历史过程记录，不代表当前仓库仍保留这些入口。后续 agent 应以 `docs/cloud_train_rtx4090_zh.md` 和 `docs/research_solution_zh.md` 的 full3d-only 说法作为当前操作口径。

## 23. full3d 额定工况与拓扑动作空间再校准

### 23.1 用户新增澄清

用户指出上一轮仍有任务理解偏差。本任务是固定初始材料体积下的三维拓扑优化问题，不是固定 `100V` 评估一个几何，也不是只优化圆柱侧面。

本轮重新对齐以下口径：

- `100V` 是系统最大允许电压上限，不是固定工作电压。
- 每个候选几何都必须先搜索综合 `0-3um` 辐射收益和蒸发/寿命后的额定稳态；被选额定电压必须 `<=100V`。
- 初始 5mm x 15mm 圆柱的额定电压约 `0.34V`，作为 sanity check。
- 寿命比必须不低于初始圆柱寿命的 `30%`。
- 钨棒内部需要计算轴向导热；两端铜电极按 `300K` 固定温度边界。
- 忽略钨-铜接触热阻和接触电阻。
- 电压只跨钨棒两侧，不包含电极压降。
- 自由表面向 `300K` 环境做净辐射散热并参与升华；电极接触端面不计辐射和升华。

### 23.2 代码修正

- `utils/full3d_optimizer.py`
  - 保留默认 `rated_search`，每个候选几何在 `V <= 100V` 下搜索额定工况。
  - 热平衡中自由表面散热改用全部自由表面积对 `300K` 环境净辐射；外接吸收面 `0K` 的 `0-3um` 有效输出仍使用 escaped/effective area 统计。
  - 新增显式 full3d 拓扑动作空间：
    - 侧壁 `Chebyshev(z) x Fourier(theta)` 径向位移；
    - 顶/底面 `Chebyshev(radius) x Fourier(theta)` 轴向位移；
    - 两端 5mm 电极边界节点 mask 固定；
    - 每次动作后执行体积投影和电极误差校正。
  - `run_full3d_optimization()` 改为 CEM：按动作分布采样候选、用 full3d 物理额定搜索评估、按精英候选更新动作均值和方差。
  - 轻量 3D U-Net/GNN 保留为可选结构化扰动源，不再作为未定义动作空间的唯一优化器。
- `config/cylinder_cfg.py`
  - 新增 `full3d_action_axial_modes`、`full3d_action_circum_modes`、`full3d_action_cap_radial_modes`。
  - 新增 `full3d_cem_elite_fraction`、`full3d_cem_initial_sigma`、`full3d_cem_min_sigma`、`full3d_cem_smoothing`。
- `optimize_3d.py`
  - 新增 CLI 参数 `--action-axial-modes`、`--action-circum-modes`、`--action-cap-radial-modes`、`--cem-initial-sigma`、`--cem-elite-fraction`。
  - summary 写入 `optimizer`、`action_space_full3d`、`action_dim_full3d`、`cem_elite_fraction_full3d`。
- `tests/test_full3d_optimizer.py`
  - 新增显式拓扑动作空间回归测试，确认侧面和端面非电极区域都能被动作移动且电极保持约束。
  - 增加自由表面热平衡面积字段检查。

### 23.3 本地验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_static_compile
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest tests.test_full3d_optimizer
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --device cpu --smoke --no-step --experiment-name codex_full3d_cem_smoke
```

CPU smoke 摘要：

- `baseline_voltage_v = 0.34`
- `final_voltage_v = 0.34`
- `voltage_search_mode = rated_search`
- `action_dim_full3d = 91`
- `power_ratio_full3d = 1.1156`
- `lifetime_ratio_full3d = 0.9825`
- `volume_change_ratio_full3d = 2.89e-08`
- `electrode_voltage_drop_v = 0.0`
- `electrode_boundary_temperature_k = 300.0`
- `feasible = True`

说明：该 smoke 仅验证物理/优化/导出链闭环和口径正确性，不代表长预算 RTX4090 最优结果。

## 24. 端面 footprint 与电极接触面积边界修正

### 24.1 用户新增澄清

用户进一步指出：钨棒横向长度/两端间距始终是 `15mm`，但与电极接触的钨端面可以改变，不一定是 `5mm` 直径圆形。因此端面需要区分：

- 超出固定圆形电极盘的钨端面：不考虑向外辐射、不考虑升华、不参与电极导热。
- 原本电极盘覆盖但现在没有钨端面接触的区域：不属于钨表面，但会减少电极导热接触面积，从而影响钨棒内部温度分布。

### 24.2 代码修正

- `utils/full3d_optimizer.py`
  - 不再把端面外缘圆环固定为 `2.5mm` 半径。
  - `project_full3d_geometry()` 改为保持所有轴向 ring/端面 z 坐标，从而保证端距始终 `15mm`；体积投影只缩放平面内坐标。
  - 新增端面 footprint 与固定 `5mm` 电极盘的重叠面积计算，输出：
    - `electrode_contact_area_m2`
    - `noncontact_end_face_area_m2`
    - `missing_electrode_contact_area_m2`
  - 热求解不再把两端节点硬设为 `300K`；改为按照实际接触面积构造到 `300K` 电极的导热通道。
  - 所有端面区域从辐射和升华面积中排除，只有侧面自由表面参与辐射散热和蒸发寿命。
  - full3d 动作空间中顶/底面动作从轴向位移改为端面 footprint 的平面内径向位移。
- `optimize_3d.py`
  - summary 增加端面 footprint/contact 诊断字段。
  - 电极约束文案改为“电极盘固定，钨端面 footprint 可变，只有重叠面积导热”。
- `tests/test_full3d_optimizer.py`
  - 更新端面动作测试：检查端面 footprint 平面内变化而非 z 方向移动。
  - 新增接触面积回归测试：端面 footprint 缩小时，电极接触面积下降、缺失接触面积上升，平均温度升高。
- `config/cylinder_cfg.py`
  - 新增 `full3d_electrode_contact_length_m`，用于把实际接触面积转换为端部导热通道强度。

### 24.3 文档同步

- `docs/research_solution_zh.md` 已同步端面 footprint 可变、电极盘固定、端面不辐射/不升华、实际重叠面积导热。
- `docs/cloud_train_rtx4090_zh.md` 已新增服务器结果检查字段：`electrode_contact_area_m2`、`noncontact_end_face_area_m2`、`missing_electrode_contact_area_m2`。

### 24.4 本地验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest discover -s tests -p "test_*.py"
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --device cpu --smoke --no-step --experiment-name codex_contact_footprint_smoke
```

验证结果：

- 单元测试 `11` 项通过。
- CPU smoke 正常生成 STL/GIF/MP4/summary。
- smoke 中 `baseline_voltage_v = 0.34`，`final_voltage_v = 0.34`，仍为额定搜索。
- summary 输出 `electrode_contact_area_m2`、`noncontact_end_face_area_m2`、`missing_electrode_contact_area_m2`。
- `electrode_max_error_m = 0.0` 表示端距/端面 z 约束满足，不再表示端面半径必须等于 `2.5mm`。

## 25. full3d 全局初始形状策略场 CEM 口径同步

### 25.1 本轮用户要求

用户明确当前代码方向已从“圆柱附近局部形变 CEM”改为 **Phy-DRL-LSM inspired global initial-shape strategy-field CEM**。因此后续文档和运行解释不得再把指定 `5mm x 15mm` 圆柱理解为默认最优初始形状或局部微扰中心。

当前统一口径：

- 指定圆柱只作为通电前材料体积基准和寿命 baseline。
- 每个 action 直接生成一个完整的通电前初始几何，而不是在圆柱附近做少量局部修补。
- 每个候选几何仍保持两端平面距离 `15mm`，且材料体积与初始圆柱相等。
- 两端固定电极仍为 `5mm` 圆盘；钨端面 footprint 可变，电极导热按 footprint 与固定电极圆盘的重叠面积计算。
- 所有端面不参与辐射、不参与升华；只有自由侧表面参与向 `300K` 环境的净辐射散热和蒸发寿命计算。
- 每个候选几何都在内环搜索 `V <= 100V` 的额定稳态；`100V` 只是上限，最优额定电压不要求接近 `100V`。
- 评分主目标改为额定工况下 `0-3um` escaped energy-conversion efficiency，而不是单纯净辐射功率。
- 硬约束仍包括 `lifetime >= 30% baseline`、`T <= 3000C`、体积相等、端距/电极边界满足要求。

### 25.2 动作空间口径

full3d action 输出四类 strategy map：

1. `radiation`
2. `evaporation`
3. `current`
4. `direct`

这些 strategy map 与局部物理敏度组合为自由表面的法向速度场，再通过 Lagrange 体积保持投影生成全局初始几何。该流程借鉴 Phy-DRL-LSM 的“物理策略场 + 体积约束投影”思想，但当前优化器使用 CEM 在策略场参数空间中搜索。

上下端面 footprint 继续使用平面内模式，不做端面轴向位移；端距固定为 `15mm`。电极固定 `5mm` 圆盘，不随钨端面 footprint 变形；导热强度由两者重叠面积决定。

### 25.3 文档与 summary 字段同步

后续运行/方案文档应重点检查以下字段，确认运行的是全局初始形状策略场口径：

- `optimization_target_full3d`
- `strategy_channels_full3d`
- `global_shape_steps_full3d`
- `final_energy_conversion_efficiency_0_3um`
- `energy_conversion_efficiency_ratio`

其中：

- `optimization_target_full3d` 应说明目标为 rated-condition `0-3um` escaped energy-conversion efficiency，并包含 `V<=100V`、体积相等、温度和寿命约束。
- `strategy_channels_full3d` 默认应为 `4`，对应 radiation / evaporation / current / direct。
- `global_shape_steps_full3d` 表示每个 action 生成完整初始几何时的全局策略场推进步数。
- `final_energy_conversion_efficiency_0_3um` 是正式主目标字段。
- `energy_conversion_efficiency_ratio` 用于和初始圆柱 baseline 对比。

### 25.4 代码修改范围

本轮同步完成了代码实现和文档口径，核心代码修改范围包括：

- `utils/full3d_optimizer.py`
  - 移除 `project_full3d_geometry()` 中相对圆柱的小位移截断，允许同体积材料在更大全局空间内重分布。
  - 新增 global initial-shape strategy-field action：四类 strategy map 与物理敏度组合为法向速度场，并做 Lagrange 体积保持投影。
  - 新增 `build_full3d_initial_shape_from_action()`，每个 action 从圆柱材料基准直接生成完整通电前初始几何。
  - CEM 主循环改为在全局初始形状 action 空间中采样和更新，而不是从上一代几何做局部小步扰动。
  - 额定工况内环保留 `V <= 100V` 搜索，但电压上限不作为优化目标；主评分改为 `0-3um` energy-conversion efficiency。
- `config/cylinder_cfg.py`
  - 新增 `full3d_action_strategy_channels`、`full3d_global_shape_steps`、`full3d_global_step_m`、`full3d_global_step_decay`、`full3d_global_max_radius_m`、`full3d_neural_policy_amplitude_m`。
- `optimize_3d.py`
  - 新增 CLI 参数 `--action-strategy-channels`、`--global-shape-steps`、`--global-step-m`、`--global-max-radius-m`。
  - summary 新增 `optimization_target_full3d`、`strategy_channels_full3d`、`global_shape_steps_full3d`、`final_energy_conversion_efficiency_0_3um`、`energy_conversion_efficiency_ratio`。
- `tests/test_full3d_optimizer.py`
  - 新增全局初始形状 action 回归测试，确认生成几何不是圆柱附近微扰，并保持同体积与额定电压约束。

文档同步范围包括：

- `docs/workflow_log.md`
- `docs/research_solution_zh.md`
- `docs/cloud_train_rtx4090_zh.md`

### 25.5 本地验证

已通过：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest discover -s tests -p "test_*.py"
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --device cpu --smoke --no-step --experiment-name codex_global_initial_shape_smoke --output-dir outputs/three_d_runs --generations 1 --population-size 3 --thermal-iters 120
```

CPU smoke 摘要：

- `method = full3d_phydrl_lsm_cem_global_initial_shape_300k_sink`
- `voltage_search_mode = rated_search`
- `baseline_voltage_v = 0.34`
- `final_voltage_v = 0.3`
- `final_energy_conversion_efficiency_0_3um = 0.0037781866984802665`
- `energy_conversion_efficiency_ratio = 2.360031453348258`
- `selected_archive_index = 2`
- `selection_reason_full3d = selected feasible globally parameterized initial shape with highest constrained rated-efficiency score`

该结果说明 smoke 已选中非基线的全局初始形状候选；`final_voltage_v` 没有接近 `100V`，符合“100V 只是上限而不是目标电压”的物理口径。

## 26. Lifecycle-Aware Phy-DRL-LSM SOTA 升级

### 26.1 本轮实现概览

本轮在 full3d 全局初始形状 strategy-field 路线之上，升级为 **Lifecycle-Aware Phy-DRL-LSM SOTA** 工作流。核心变化是：优化器不只比较单个额定稳态的效率，还显式记录生命周期准静态演化、可见性遮挡、特征尺度失效和多目标 Pareto archive，便于后续在效率、寿命、温度、可制造性之间做可解释选择。

### 26.2 新增 CLI

`optimize_3d.py` 新增或同步以下入口参数：

- `--optimizer`
  - 支持 `cem`、`cmaes`、`mo-cmaes`、`turbo-surrogate`。
  - 默认使用 `mo-cmaes`，用于多目标策略场搜索和 Pareto archive 构建。
- `--objective-mode`
  - 支持 `efficiency`、`lifecycle`、`sota`。
  - 默认 `sota`，将额定效率、生命周期、温度/寿命约束和特征尺度约束统一纳入选择逻辑。
- `--lifecycle-steps`
  - 控制 quasi-static lifecycle trace 的离散步数。
- `--visibility-rays`
  - 控制 ray-cast visibility 诊断射线数量。
- `--feature-scale-mode`
  - 当前支持 `sdf`，用于特征尺度失效判据。
- `--surrogate-train-every`
  - 控制 surrogate/加速优化相关训练或刷新频率。

### 26.3 新增优化与诊断模块口径

- **MO-CMA-ES / Pareto archive**
  - 在 strategy-field action 空间内做多目标采样和更新。
  - archive 保留候选的非支配排序、约束状态、效率、寿命、温度、体积和几何诊断。
- **quasi-static lifecycle trace**
  - 对选中候选记录随蒸发/升华推进的准静态生命周期轨迹。
  - 用于确认 `lifetime >= 30% baseline` 不是只在额定初始时刻成立。
- **ray-cast visibility**
  - 对自由表面到外接吸收面的可逃逸辐射做 ray-cast visibility 诊断。
  - 输出遮挡/可见性相关字段，区分几何表面积增加和实际 escaped `0-3um` 有效输出。
- **feature-scale failure**
  - 增加基于 `feature-scale-mode=sdf` 的特征尺度失效检查。
  - 用于约束蒸发/升华导致的关键尺度变化，避免只优化短时高效率但快速失效的形状。
- **surrogate / policy scaffold**
  - 新增 `train_surrogate.py`，用于基于 `pareto_archive_full3d.json` 的 surrogate 训练脚手架。
  - 新增 `train_policy.py`，用于基于 Pareto archive 的 policy 蒸馏/训练脚手架。
  - 当前主优化仍以真实 full3d 物理评估为准，训练脚手架用于后续加速和策略复用。

### 26.4 新增产物

正式 full3d 输出目录中新增：

- `pareto_archive_full3d.csv`
- `pareto_archive_full3d.json`
- `lifecycle_trace_full3d.csv`
- `visibility_diagnostics_full3d.csv`
- `surrogate_train_metrics.json`
- `policy_train_metrics.json`

其中：

- `pareto_archive_full3d.csv/json` 是候选级多目标诊断和非支配结果。
- `lifecycle_trace_full3d.csv` 是选中几何的准静态生命周期演化。
- `visibility_diagnostics_full3d.csv` 是 ray-cast escaped visibility 诊断。
- `surrogate_train_metrics.json` 与 `policy_train_metrics.json` 记录训练脚手架状态、样本量和指标。

### 26.5 本地验证

本轮本地验证命令：

```bash
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe -m unittest discover -s tests -p "test_*.py"
C:\Users\XiZie\.conda\envs\mcga_xzh\python.exe optimize_3d.py --device cpu --smoke --no-step --experiment-name codex_lifecycle_sota_smoke --output-dir outputs/three_d_runs --generations 1 --population-size 3 --thermal-iters 120
```

验证重点：

- `unittest discover` 覆盖 full3d optimizer、静态编译、产物导出和新增诊断路径。
- CPU smoke 能完成 `mo-cmaes` 默认优化入口、额定电压搜索、生命周期/可见性诊断和产物写出。
- smoke 输出目录应包含 `pareto_archive_full3d.csv/json`、`lifecycle_trace_full3d.csv`、`visibility_diagnostics_full3d.csv`、`surrogate_train_metrics.json`、`policy_train_metrics.json`。

### 26.6 后续阅读口径

后续 agent 判断当前 full3d SOTA 路线时，应优先查看：

- `optimizer_full3d`
- `objective_mode_full3d`
- `optimization_target_full3d`
- `pareto_archive_csv`
- `pareto_archive_json`
- `lifecycle_trace_csv`
- `visibility_diagnostics_csv`
- `surrogate_train_metrics_json`
- `policy_train_metrics_json`

这些字段比单一 `power_ratio_full3d` 更能反映当前“生命周期感知 + 多目标 Pareto + escaped efficiency”的正式优化口径。
