# 圆柱钨体 full3d 物理优化方案说明

## 1. 当前正式工作流

当前仓库已收敛为 **full3d-only** 工作流：

- 正式入口：`optimize_3d.py`
- 核心后端：`utils/full3d_optimizer.py`
- 几何表示：封闭三维 mesh，侧面、顶面、底面均可变化
- 输出目录：`outputs/three_d_runs/<experiment>_<stamp>/`

旧 1D/RL/hybrid/sidefield 入口、环境、测试和工具已移除；文档中的历史记录仅作为过程复盘，不再作为可运行路线。

## 2. 统一物理口径

- `100V` 是额定工况搜索上限，不是固定工作电压。
- 正式优化默认搜索 `V <= 100V` 的稳态工况，选择综合辐射收益和蒸发/寿命后的最优稳态。
- 初始 5mm x 15mm 圆柱的额定电压约 `0.34V`，固定 20V/100V 只适合做过温诊断。
- 电压只加在钨棒两端，不包含铜电极电压降。
- 钨棒内部显式计算轴向导热。
- 两端接触铜电极按 `300K` 固定温度边界处理。
- 当前口径忽略钨棒-铜电极之间的接触热阻和接触电阻。
- 自由表面计算向 `300K` 环境的净辐射/散热和升华。
- 接触铜电极的端面区域不计向外辐射和升华。
- 0-3 微米有效辐射按外接吸收面统计；发射率使用题面给定口径：
  - `0-3 um`：`0.35`
  - 其他波段：`0.15`
- 蒸发律采用：

$$
Y_e = A \exp(B / T)
$$

其中 `A = 3.9e8 [g/(cm^2*s)]`，`B = -1.023e5 [K]`。

## 3. 几何与约束

- 通电前材料体积投影回初始圆柱体积。
- 两端 5mm 圆形电极边界保持直径和相对位置不变。
- 非电极自由表面允许三维变形。
- 形状评估同时检查体积、电极误差、温度、寿命和辐射收益。
- 固定电压模式 `--fixed-voltage <V>` 仅用于诊断固定电压是否过温，不用于正式优化。

## 4. 运行命令

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

## 5. 关键产物

- `optimized_full3d.stl`
- `optimized_full3d.stp`（未加 `--no-step` 且 FreeCAD 可用时）
- `topology_evolution_full3d.gif`
- `topology_evolution_full3d.mp4`
- `optimization_history_full3d.csv`
- `run_summary_full3d.json`
- `design_strategy_report_full3d.md`

## 6. 交接索引

- RTX4090 服务器运行说明见 `docs/cloud_train_rtx4090_zh.md`。
- 历史实现、物理口径修正、legacy 删除记录见 `docs/workflow_log.md`。
