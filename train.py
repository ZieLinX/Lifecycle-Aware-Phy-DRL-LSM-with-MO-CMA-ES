import torch
from config.cylinder_cfg import CylinderPhysicsCfg
from envs.cylinder_env import CylinderPhysicsEnv
from rl_games.torch_runner import Runner
import yaml

def train_miku():
    # 初始化环境
    cfg = CylinderPhysicsCfg()
    env = CylinderPhysicsEnv(cfg)
    
    # 载入 RL-Games 配置
    with open("config/config.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)

    # 启动训练
    runner = Runner()
    runner.load(train_cfg)
    runner.run({'train': True})

if __name__ == "__main__":
    train_miku()