# 生命周期感知全三维钨棒拓扑优化

在固定体积、电极间距和温度/寿命约束下，搜索钨棒器件的三维初始形状，以提高 0–3 μm 净出射辐射、生命周期平均输出和能量转换效率。

本项目实现 **Lifecycle-Aware Phy-DRL-LSM + MO-CMA-ES**：用 Chebyshev/Fourier 低维策略场驱动闭合网格形变，体积投影保持材料量不变，再用真实物理评估器进行多目标 Pareto 搜索。

> 当前主结果来自 MO-CMA-ES 的真实物理评估。`train_policy.py` 和 `train_surrogate.py` 是后续训练入口，本次长跑未启用神经策略或 surrogate-only 评估。

## 优化内容

- **几何**：纯钨闭合三角网格；侧壁和上下端面足迹均可变化。
- **物理评估**：温度相关电阻、焦耳热、轴向导热、电极导热、表面辐射、升华损失和 ray-cast 自遮挡。
- **生命周期**：准静态更新自由表面，任一特征尺度相对变化达到 20% 时判定失效。
- **多目标**：最大化初始 `P0`、寿命、生命周期平均 `Pavg` 和可逃逸比例 `visibility`。
- **搜索器**：MO-CMA-ES Pareto 非支配排序与 hypervolume contribution。

## 约束

| 项目 | 设置 |
| --- | --- |
| 基准体积 | 直径 5 mm、高度 15 mm 圆柱体积 |
| 电极 | 两端固定 5 mm 圆形铜电极，间距 15 mm |
| 额定电压 | 搜索范围不超过 100 V（不是优化目标） |
| 最高温度 | 不超过 3000 °C（3273.15 K） |
| 寿命 | 不低于基准圆柱寿命的 30% |
| 网格 | 闭合、可导出 STL |

## 快速开始

### 安装

需要 Python 3.10+ 以及 `torch`、`numpy`、`scipy`、`matplotlib`、`imageio`、`Pillow`、`trimesh`。根据 CUDA 版本安装对应的 PyTorch。

```bash
pip install numpy scipy matplotlib imageio pillow trimesh
```

### 冒烟测试

```bash
python optimize_3d.py --smoke --device cpu --output-dir outputs/smoke
```

### RTX 4090 长跑

```bash
python optimize_3d.py --experiment-name mcga_sota_mocma_4090_long --output-dir outputs/three_d_runs --device cuda:0 --optimizer mo-cmaes --objective-mode sota --generations 30 --population-size 48 --thermal-iters 800 --lifecycle-steps 16 --visibility-rays 512 --visibility-batch-size 512 --visibility-device auto --eval-workers 2 --torch-threads 8 --no-step
```

常用参数可通过 `python optimize_3d.py --help` 查看。`--no-step` 会跳过可选的 FreeCAD STEP 转换。

## 结果摘要

下表为 2026-05-04 RTX 4090 长跑（共评估 1441 个候选，1437 个满足约束）的代表性结果。`61` 是默认导出的综合折中候选，`234` 和 `335` 分别偏向最高功率和最高效率。

| 指标 | 基准圆柱 | 折中 `61` | 最高功率 `234` | 最高效率 `335` |
| --- | ---: | ---: | ---: | ---: |
| 初始 `P0` (W) | 1.6828 | 1.8351 | **2.2229** | 2.2180 |
| 生命周期平均 `Pavg` (W) | 1.5871 | 1.8124 | **2.2124** | 2.1936 |
| 0–3 μm 效率比 | 1.00× | 2.16× | 2.95× | **3.20×** |
| 额定电压 (V) | 0.34 | 0.30 | 0.30 | 0.30 |
| 最高温度 (K) | 1235.7 | 1104.3 | 1113.0 | 1117.8 |

结果是优化器与当前快速物理模型的联合输出；正式研究应对 Pareto 候选提高射线数，并使用更高保真 3D 热/电复评。

## 输出文件

每次运行会在 `outputs/three_d_runs/<experiment>_<timestamp>/` 下写入：

- `optimized_full3d.stl`：默认的 Pareto 折中几何（`archive_index=61`）。
- `topology_evolution_full3d.gif` / `.mp4`：形状演化动画。
- `run_summary_full3d.json`：运行配置和最终指标。
- `pareto_archive_full3d.csv` / `.json`：候选及 Pareto archive。
- `lifecycle_trace_full3d.csv`：导出候选的生命周期轨迹。

如需导出 `234` 或 `335` 的几何，当前版本需要在优化过程中保存对应候选的网格；archive 表中的数值不能直接恢复 STL。

## 目录

```text
optimize_3d.py          # 优化入口
config/                 # 材料、几何和运行配置
utils/full3d_optimizer.py
                         # 几何、物理评估、生命周期和 MO-CMA-ES
train_policy.py         # 策略训练脚手架
train_surrogate.py      # surrogate 训练脚手架
tests/                  # 静态检查与优化器测试
```

## 已知局限

- 热传导采用轴向 ring lumped 模型，不是完整 3D FEM。
- 升华寿命采用准静态 recession；`visibility` 使用有限射线采样，存在采样误差。
- 本项目的“真实物理评估”指代码内快速 evaluator，不等同于实验或高保真商业仿真。
