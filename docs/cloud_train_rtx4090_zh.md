# RTX4090 Ubuntu 服务器训练文档

这份文档面向可 SSH 登录的 Ubuntu RTX4090 服务器，按当前仓库脚本说明如何安装环境、运行训练、查看日志和回收产物。  
仓库里有两条路线，必须分开理解：

1. `train_rl.py` 是**轴对称 RL 路线**，训练的是 ring profile / 剖面策略。
2. `optimize_3d.py` 是**真 3D 路线**，直接优化 `r(z, theta)` 三维半径场。

## 1. 两条路线和产物

### 1.1 `train_rl.py` 轴对称 RL 路线

- 训练目录默认是 `outputs/rl_runs/<experiment>_<stamp>/`
- 训练中实时快照会写到 `outputs/rl_runs/<experiment>_<stamp>/realtime/`
- 训练结束后会自动做最终评估，产物落到 `outputs/final_eval/<run>/`
- 关键产物：
  - `run_summary.json`
  - `rollout_metrics.csv`
  - `optimized_cylinder.stl`
  - `optimized_cylinder.stp`（需要 FreeCAD CLI 才能生成）
  - `topology_evolution.gif`
  - `topology_evolution.mp4`（本机编码器可用时才会生成）

训练过程里的实时动画文件名是 `training_evolution.gif/.mp4`，它在 `realtime/` 目录里，不是最终评估产物。

### 1.2 `optimize_3d.py` 真 3D 路线

- 输出目录默认是 `outputs/three_d_runs/<experiment>_<stamp>/`
- 关键产物：
  - `topology_evolution_3d.gif`
  - `topology_evolution_3d.mp4`（`ffmpeg/libx264` 可用时生成，否则可能是 `null`）
  - `optimized_cylinder_3d.stl`
  - `optimized_cylinder_3d.stp`（可选，依赖 FreeCAD）
  - `optimization_history_3d.csv`
  - `run_summary_3d.json`
  - `design_strategy_report_3d.md`

## 2. 服务器环境准备

### 2.1 基础工具

```bash
sudo apt-get update
sudo apt-get install -y git tmux htop ffmpeg
```

`ffmpeg` 建议装上，后面 MP4 导出更稳。

### 2.2 安装 Miniconda（如未安装）

```bash
cd ~
wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3.sh -b -p ~/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2.3 创建 conda 环境

```bash
conda create -n mcga_4090 python=3.10 -y
conda activate mcga_4090
```

### 2.4 安装 PyTorch CUDA 版

RTX4090 推荐直接装 CUDA 轮子，不单独装完整 CUDA Toolkit：

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.5 安装仓库依赖

```bash
pip install rl-games gymnasium numpy scipy pyyaml tensorboard matplotlib imageio imageio-ffmpeg
```

`imageio-ffmpeg` 建议一起装，GIF/MP4 导出更稳。

### 2.6 可选：FreeCAD / STEP

如果你需要 `.stp/.step`，再额外准备 FreeCAD CLI。没有 FreeCAD 也能正常训练，区别只是 STP 可能不会生成。  
在真 3D 命令里也可以直接加 `--no-step` 跳过 STEP 导出。
如果 FreeCAD 在 STL -> STEP 转换时长时间占用 CPU，优先停止当前进程并改用 `--no-step`；新版本会用 `--freecad-timeout` 限制等待时间，超时后保留 STL 并继续结束。

### 2.7 验证 GPU

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 3. 拉取代码

```bash
mkdir -p ~/work
cd ~/work
git clone <你的仓库地址> make_cylinder_great_again
cd make_cylinder_great_again
```

如果你在指定分支上工作，再切到对应分支即可。

## 4. 推荐执行顺序

建议按这个顺序跑：

1. 先跑 `train_rl.py --smoke`，确认 GPU、依赖、导出链路都通。
2. 再跑 `optimize_3d.py` 正式优化。
3. `train_rl.py` 的正式 RL 训练命令保留作对照和补充实验。

### 4.1 先跑 smoke

RL 路线 smoke：

```bash
python -u train_rl.py --smoke --experiment-name mcga_phy_drl_4090_smoke
```

真 3D 路线 smoke：

```bash
python -u optimize_3d.py --smoke --no-step --experiment-name mcga_3d_4090_smoke
```

如果只想检查 FreeCAD STEP 导出是否可用，用很短超时单独试：

```bash
python -u optimize_3d.py --smoke --experiment-name freecad_check_step --freecad-timeout 20
```

若 20 秒内未完成，脚本会放弃 STEP、保留 STL/GIF，并把 `stp` 记为 `null`。

### 4.2 再跑 3D 正式优化

```bash
python -u optimize_3d.py \
  --experiment-name mcga_3d_4090 \
  --output-dir outputs/three_d_runs \
  --generations 4 \
  --population-size 16 \
  --axial-modes 4 \
  --circum-modes 2 \
  --thermal-iters 640 \
  --no-step
```

### 4.3 保留 RL 正式训练命令作对照

```bash
python -u train_rl.py \
  --experiment-name mcga_phy_drl_4090 \
  --max-epochs 400 \
  --num-actors 16 \
  --realtime-interval 4 \
  --console-interval 20
```

如果显存紧张，先把 `--num-actors` 降到 8，再看效果。

## 5. 长任务运行方式

### 5.1 用 tmux

```bash
tmux new -s mcga4090
conda activate mcga_4090
cd ~/work/make_cylinder_great_again
python -u optimize_3d.py \
  --experiment-name mcga_3d_4090 \
  --output-dir outputs/three_d_runs \
  --generations 4 \
  --population-size 16 \
  --axial-modes 4 \
  --circum-modes 2 \
  --thermal-iters 640 \
  --no-step
```

断开会话但不中断任务：

```bash
Ctrl+b
d
```

恢复会话：

```bash
tmux attach -t mcga4090
```

查看会话：

```bash
tmux ls
```

### 5.2 用 nohup

```bash
conda activate mcga_4090
cd ~/work/make_cylinder_great_again
nohup python -u optimize_3d.py --experiment-name mcga_3d_4090 --output-dir outputs/three_d_runs --generations 4 --population-size 16 --axial-modes 4 --circum-modes 2 --thermal-iters 640 --no-step > 3d_4090.log 2>&1 &
tail -f 3d_4090.log
```

RL 路线也可以同样方式跑，只是命令换成 `train_rl.py`。

## 6. 日志和产物怎么看

### 6.1 RL 路线

训练日志主要看控制台或你重定向出来的 `train_rl.log`。训练目录和最终评估目录分别是：

- `outputs/rl_runs/<experiment>_<stamp>/`
- `outputs/final_eval/<run>/`

常看文件：

- `outputs/rl_runs/<run>/realtime/training_evolution.gif`
- `outputs/rl_runs/<run>/realtime/training_evolution.mp4`
- `outputs/final_eval/<run>/run_summary.json`
- `outputs/final_eval/<run>/rollout_metrics.csv`
- `outputs/final_eval/<run>/optimized_cylinder.stl`
- `outputs/final_eval/<run>/topology_evolution.gif`
- `outputs/final_eval/<run>/topology_evolution.mp4`

### 6.2 真 3D 路线

输出目录默认类似：

- `outputs/three_d_runs/<experiment>_<stamp>/`

常看文件：

- `outputs/three_d_runs/<run>/topology_evolution_3d.gif`
- `outputs/three_d_runs/<run>/topology_evolution_3d.mp4`
- `outputs/three_d_runs/<run>/optimized_cylinder_3d.stl`
- `outputs/three_d_runs/<run>/optimized_cylinder_3d.stp`
- `outputs/three_d_runs/<run>/optimization_history_3d.csv`
- `outputs/three_d_runs/<run>/run_summary_3d.json`
- `outputs/three_d_runs/<run>/design_strategy_report_3d.md`

## 7. 如果 MP4 没出来

如果 `mp4` 显示 `null`、没生成，先看同目录下的 GIF。  
这通常说明编码器不可用，优先这样处理：

1. 确认已经安装 `ffmpeg`
2. 确认已经安装 `imageio-ffmpeg`
3. 再重跑一次训练或 3D 优化，或者把已有 GIF 后处理成 MP4

真 3D 路线和 RL 路线都遵循同样逻辑：

- GIF 基本一定会有
- MP4 依赖本机视频编码能力，`libx264` 不可用时可能不写出

## 8. 拉回服务器上的产物

建议把回收来的日志和产物都放进本地仓库根目录下的 `RTX4090/`。这个目录只用于**复制回来的结果**，不应该作为代码提交内容。

示例：

```bash
mkdir -p RTX4090
scp -r <user>@<host>:~/work/make_cylinder_great_again/outputs/final_eval/<run> RTX4090/
scp -r <user>@<host>:~/work/make_cylinder_great_again/outputs/three_d_runs/<run> RTX4090/
scp <user>@<host>:~/work/make_cylinder_great_again/3d_4090.log RTX4090/
```

仓库里已经把 `RTX4090/` 加进了忽略列表，正常情况下不会被提交。

## 9. 常见问题

### 9.1 看不到输出

- 运行命令时加 `-u`
- 尽量先 `conda activate`，再直接跑 `python`
- 不要依赖缓冲输出判断程序是否还活着，建议配合 `tail -f`

### 9.2 CUDA 不可用

- 先看 `nvidia-smi`
- 再确认当前环境里的 `torch` 是 CUDA 版
- 若 `torch.cuda.is_available()` 为 `False`，先不要继续训练

### 9.3 4090 显存不够

- 先降 `--num-actors`
- 再考虑减小 3D 路线的 `--population-size`
- RL 路线下 `--max-epochs` 不影响单步显存，但会影响总时长

### 9.4 STEP 没生成

- 这是可选项，不影响核心训练
- 先确认是否装了 FreeCAD CLI
- 如果 FreeCAD 转换卡住，终止当前任务后拉取最新代码，使用 `--freecad-timeout 20` 检查；正式优化继续建议加 `--no-step`
- 没有 FreeCAD 时，保留 STL 即可

## 10. 3D 正式跑仍选回 baseline 怎么看

拉取最新 `xzh` 后，3D 路线已修正两点：

- 3D rated-condition 和 transient 现在使用真实曲面面元面积 `sqrt(r^2(1+(dr/dz)^2)+(dr/dtheta)^2) dz dtheta`，不再把非平直 3D 表面当作平面展开面积。
- `run_summary_3d.json` 和 `design_strategy_report_3d.md` 会写出 archive 诊断，包括 `selection_reason_3d`、`best_nonbaseline_by_score`、`best_feasible_nonbaseline_by_initial_power`、`max_nonbaseline_surface_area_ratio`。

建议先跑一个快速 3D 冒烟确认诊断字段存在：

```bash
python -u optimize_3d.py --smoke --no-step --experiment-name mcga_3d_diag_smoke
```

然后正式跑：

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

跑完重点看：

```bash
python - <<'PY'
import json, pathlib
root = sorted(pathlib.Path("outputs/three_d_runs").glob("mcga_3d_4090_*"))[-1]
s = json.loads((root / "run_summary_3d.json").read_text())
for k in [
    "selected_archive_index",
    "selection_reason_3d",
    "initial_power_ratio_3d",
    "lifetime_ratio_3d",
    "feature_change_ratio_3d",
    "max_nonbaseline_surface_area_ratio",
    "best_feasible_nonbaseline_by_initial_power",
]:
    print(k, "=", s.get(k))
PY
```

如果 `selected_archive_index=0`，现在不再只能说明“结果没变”，而要看 `selection_reason_3d`：

- `no feasible non-baseline candidate improved...`：有 3D 形变，但细网格复评后功率没有超过圆柱。
- `all feasible non-baseline candidates scored below baseline`：可能功率、寿命、温度均匀性或约束综合后输给 baseline。
- `selected feasible archive candidate...`：已选中非基准 3D 候选。
