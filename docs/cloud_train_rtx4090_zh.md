# 云服务器（RTX4090）训练教程（SSH 版）

本教程面向：你将租用一台可 SSH 登录的 RTX4090 Linux 服务器，在其上配置环境、拉取本仓库 `xzh` 分支（含实时进度条/日志），并用进程守护长期运行训练。

## 0. 约定与目标

- 目标：在 **GPU（cuda）** 上运行 `train_rl.py` 的正式训练，且持续输出训练进度（epoch/fps）与实时关键指标（每 N 步打印一次）。
- 产物目录：
  - 训练：`outputs/rl_runs/<experiment>_<stamp>/`
  - 训练快照（实时形状）：`outputs/rl_runs/<experiment>_<stamp>/realtime/`
  - 训练结束自动生成训练演化：`training_evolution.gif/.mp4`（若编码器可用）
  - 最终评估：`outputs/final_eval/<experiment>_<stamp>/`（含 `run_summary.json`、`rollout_metrics.csv`、`optimized_cylinder.stl/.stp`、`topology_evolution.gif/.mp4`）

## 1. 服务器侧准备（一次性）

### 1.1 登录与基础工具

```bash
ssh <user>@<host>
```

建议安装常用工具：

```bash
sudo apt-get update
sudo apt-get install -y git tmux htop
```

### 1.2 驱动与 CUDA

- RTX4090 建议使用较新的 NVIDIA Driver（例如 535+）。
- 只要驱动正常，PyTorch 轮子自带 CUDA runtime，通常**不需要**你单独安装完整 CUDA Toolkit。

验证 GPU：

```bash
nvidia-smi
```

## 2. 拉取代码（xzh 分支）

```bash
mkdir -p ~/work && cd ~/work
git clone <你的仓库地址> make_cylinder_great_again
cd make_cylinder_great_again
git checkout xzh
git pull
```

## 3. 创建 Python 环境（推荐 conda）

### 3.1 安装 Miniconda（若无）

```bash
cd ~
wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3.sh -b -p ~/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3.2 创建环境并安装依赖

```bash
conda create -n mcga_4090 python=3.10 -y
conda activate mcga_4090
```

安装 PyTorch（建议 cu121，适配 4090，更常见）：

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装训练/数值/可视化依赖：

```bash
pip install rl-games gymnasium numpy scipy pyyaml tensorboard matplotlib imageio
```

可选（如果你要导出 STEP/STP）：
- 服务器若无 GUI，建议只导出 STL（`--no-step` 相关参数在优化脚本里；RL final eval 当前默认会尝试导出 STEP，需要 FreeCAD CLI 才会生成 STP）。

验证 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 4. 开始训练（带实时进度条与实时关键指标）

### 4.1 推荐启动命令

- `--console-interval`：每 N 个 RL step 打印一次关键指标（电压/功率/寿命比/可行性）。
- `--realtime-interval`：每 N 个 RL step 保存一次形状快照（训练结束会自动合成训练演化动图）。

示例（正式训练）：

```bash
python -u train_rl.py \
  --experiment-name mcga_phy_drl_4090 \
  --max-epochs 400 \
  --num-actors 16 \
  --realtime-interval 2 \
  --console-interval 10
```

### 4.2 你将看到的实时输出

- rl_games 的周期输出（epoch/fps/frames）类似：
  - `fps step: ... epoch: k/N frames: ...`
- 本仓库新增的实时关键指标输出类似：
  - `[rl] step=10 V*=... P0-3=...W life=... feasible=True/False`
- 若 `tqdm` 可用，还会显示 `[rl] training` 的进度条（按 epoch 递增）。

## 5. 进程守护（推荐 tmux）

### 5.1 使用 tmux（最推荐）

```bash
tmux new -s mcga
conda activate mcga_4090
cd ~/work/make_cylinder_great_again
python -u train_rl.py --experiment-name mcga_phy_drl_4090 --max-epochs 400 --num-actors 16 --realtime-interval 2 --console-interval 10
```

脱离会话（不中断进程）：按 `Ctrl+b` 再按 `d`。

回到会话：

```bash
tmux attach -t mcga
```

查看会话列表：

```bash
tmux ls
```

### 5.2 使用 nohup（备选）

```bash
conda activate mcga_4090
cd ~/work/make_cylinder_great_again
nohup python -u train_rl.py --experiment-name mcga_phy_drl_4090 --max-epochs 400 --num-actors 16 --realtime-interval 2 --console-interval 10 > train_4090.log 2>&1 &
tail -f train_4090.log
```

## 6. 训练结束后检查产物

```bash
ls -lah outputs/rl_runs
ls -lah outputs/final_eval
```

你关心的关键文件：
- `outputs/final_eval/<run>/run_summary.json`
- `outputs/final_eval/<run>/rollout_metrics.csv`
- `outputs/final_eval/<run>/optimized_cylinder.stl`（以及可能的 `.stp`）
- `outputs/final_eval/<run>/topology_evolution.gif/.mp4`
- `outputs/rl_runs/<run>/realtime/training_evolution.gif/.mp4`（训练过程演化）

## 7. 常见问题排查

### 7.1 看不到进度条/输出很慢
- 确保使用 `python -u train_rl.py ...`（关闭缓冲）。
- 若使用 `conda run ...`，部分环境会出现输出缓冲或启动很慢，建议进入环境后直接运行 `python`。

### 7.2 CUDA 不可用
- 先 `nvidia-smi` 看驱动是否正常。
- 再检查 `pip show torch` 与安装的 CUDA 轮子是否正确（cu121）。

### 7.3 4090 上显存不够
- 降低 `--num-actors`（例如 16 -> 8）。
- 也可调整 `config/rl_games_ppo.yaml` 里的 `minibatch_size` / `horizon_length`。

