# RTX4090 Ubuntu full3d 执行文档

当前仓库只保留 `optimize_3d.py` 的 full3d 封闭三维网格路线。旧 1D/RL、hybrid、sidefield 入口和历史输出已删除。

## 1. full3d 口径

- 几何是封闭三维 mesh；侧面、顶面、底面都可以变化。
- 两端 5mm 圆形电极边界保持直径和相对位置不变。
- 通电前材料体积投影回初始圆柱体积。
- 钨棒内部显式计算轴向导热；两端铜电极为 `300K` 固定温度边界。
- 忽略钨棒和铜电极之间的接触热阻、接触电阻。
- 电压只加在钨棒两端，铜电极电压降为 0。
- 只有自由表面参与向 `300K` 环境的净辐射/散热和升华；接触电极的端面不计辐射和升华。
- `100V` 是系统允许的额定搜索上限，不是固定工作电压。
- 默认从 `V <= 100V` 中搜索综合辐射收益和蒸发/寿命后的最优稳态；初始 5mm x 15mm 圆柱额定电压约 `0.34V`。
- `--fixed-voltage <V>` 只用于诊断固定电压是否过温，不用于正式优化。

## 2. 环境准备

```bash
sudo apt-get update
sudo apt-get install -y git tmux htop ffmpeg
conda create -n mcga_4090 python=3.10 -y
conda activate mcga_4090
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib imageio imageio-ffmpeg trimesh
```

如果需要 STEP 导出，再安装 FreeCAD CLI；正式优化可以直接加 `--no-step` 跳过。

## 3. 快速检查

```bash
python -u optimize_3d.py --smoke --no-step --experiment-name full3d_smoke
```

固定电压诊断示例：

```bash
python -u optimize_3d.py --smoke --no-step --experiment-name full3d_fixed20_diag --fixed-voltage 20
```

若固定电压诊断出现 `archive_feasible_count=0`，说明该固定电压工况本身过温；正式优化应去掉 `--fixed-voltage`。

## 4. 正式优化

```bash
mkdir -p logs
python -u optimize_3d.py \
  --experiment-name mcga_full3d_4090 \
  --output-dir outputs/three_d_runs \
  --generations 4 \
  --population-size 16 \
  --thermal-iters 640 \
  --no-step \
  2>&1 | tee -a logs/full3d_$(date +%F_%H%M%S).log
```

输出目录默认是：

```text
outputs/three_d_runs/<experiment>_<stamp>/
```

关键产物：

- `optimized_full3d.stl`
- `topology_evolution_full3d.gif`
- `topology_evolution_full3d.mp4`
- `optimization_history_full3d.csv`
- `run_summary_full3d.json`
- `design_strategy_report_full3d.md`

## 5. 长任务运行

```bash
tmux new -s mcga_full3d
conda activate mcga_4090
cd ~/work/make_cylinder_great_again
python -u optimize_3d.py --experiment-name mcga_full3d_4090 --output-dir outputs/three_d_runs --generations 4 --population-size 16 --thermal-iters 640 --no-step
```

断开会话：`Ctrl+b` 后按 `d`。恢复会话：

```bash
tmux attach -t mcga_full3d
```

## 6. 常看字段

```bash
python - <<'PY'
import json, pathlib
root = sorted(pathlib.Path("outputs/three_d_runs").glob("mcga_full3d_4090_*"))[-1]
s = json.loads((root / "run_summary_full3d.json").read_text())
for k in [
    "baseline_voltage_v",
    "final_voltage_v",
    "tungsten_voltage_v",
    "electrode_voltage_drop_v",
    "thermal_radiation_sink_temperature_k",
    "electrode_boundary_temperature_k",
    "thermal_converged",
    "power_ratio_full3d",
    "lifetime_ratio_full3d",
    "selection_reason_full3d",
]:
    print(k, "=", s.get(k))
PY
```

## 7. 产物回收

建议把服务器结果放到本地仓库根目录的 `RTX4090/`，该目录已被 `.gitignore` 忽略。

```bash
mkdir -p RTX4090
scp -r <user>@<host>:~/work/make_cylinder_great_again/outputs/three_d_runs/<run> RTX4090/
scp <user>@<host>:~/work/make_cylinder_great_again/logs/full3d_*.log RTX4090/
```
