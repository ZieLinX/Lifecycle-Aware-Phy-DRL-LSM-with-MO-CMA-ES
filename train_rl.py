from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.common.ivecenv import IVecEnv
from rl_games.torch_runner import Runner

from config.cylinder_cfg import make_eval_cfg, make_training_cfg
from envs.cylinder_vec_env import CylinderVecEnv
from utils.animation import build_realtime_animation, export_topology_evolution_animation, save_realtime_frame
from utils.exporter import export_ring_profile_mesh


class ProgressAlgoObserver(DefaultAlgoObserver):
    def __init__(self, total_epochs: int | None = None):
        super().__init__()
        self._total_epochs = int(total_epochs) if total_epochs is not None else None
        self._pbar = None
        self._last_epoch = -1

    def after_init(self, algo):
        super().after_init(algo)
        total = self._total_epochs
        if total is None:
            total = int(getattr(algo, "max_epochs", 0)) or None
        try:
            from tqdm import tqdm  # type: ignore

            self._pbar = tqdm(total=total, desc="[rl] training", unit="epoch", dynamic_ncols=True)
        except Exception:
            self._pbar = None

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        mean_scores = None
        try:
            if getattr(self, "game_scores", None) is not None and self.game_scores.current_size > 0:
                mean_scores = float(self.game_scores.get_mean())
        except Exception:
            mean_scores = None

        if self._pbar is not None:
            inc = int(epoch_num) - int(self._last_epoch)
            if inc > 0:
                self._pbar.update(inc)
                self._last_epoch = int(epoch_num)
            postfix = {"frame": int(frame), "time_s": f"{float(total_time):.0f}"}
            if mean_scores is not None:
                postfix["score_mean"] = f"{mean_scores:.3f}"
            self._pbar.set_postfix(postfix)
        else:
            msg = f"[rl] epoch={int(epoch_num)} frame={int(frame)} time_s={float(total_time):.1f}"
            if mean_scores is not None:
                msg += f" score_mean={mean_scores:.3f}"
            print(msg, flush=True)


class MCGARlGamesVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        # Consume fields meant for this wrapper before forwarding to env creator.
        self._realtime_dir: str | None = kwargs.pop("realtime_dir", None)
        self._realtime_interval: int = int(kwargs.pop("realtime_interval", 4))
        self._global_step = 0
        # Pass remaining kwargs (e.g. cfg) through to the underlying env creator.
        creator = env_configurations.configurations[config_name]["env_creator"]
        self.env = creator(num_envs=num_actors, **kwargs)
        self.action_space = self.env.single_action_space
        self.observation_space = self.env.single_observation_space

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = np.logical_or(terminated, truncated)
        if np.any(dones):
            obs = self.env.reset_done(dones)
        info["time_outs"] = truncated
        self._global_step += 1
        if self._realtime_dir is not None and self._global_step % self._realtime_interval == 0:
            try:
                ring_r = self.env.ring_radius[0].detach().cpu().numpy()
                m = self.env.current_metrics
                metrics_snap = {
                    "rated_voltage_v": float(m["voltage_v"][0].item()),
                    "initial_net_band_power_w": float(m["initial_net_band_power_w"][0].item()),
                    "lifetime_ratio": float(
                        m["lifetime_s"][0].item()
                        / max(float(self.env.baseline_metrics["lifetime_s"][0].item()), 1.0e-9)
                    ),
                    "feasible": bool(m["feasible"][0].item()),
                }
                save_realtime_frame(
                    ring_r,
                    self.env.cfg.height,
                    metrics_snap,
                    self._realtime_dir,
                    self._global_step,
                )
            except Exception:
                pass
        return obs, reward, dones, info

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "state_space": None,
            "use_global_observations": False,
            "agents": 1,
            "value_size": 1,
        }

    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)


def _register_rl_games_env():
    if "MCGA" not in vecenv.vecenv_config:
        vecenv.register("MCGA", lambda config_name, num_actors, **kw: MCGARlGamesVecEnv(config_name, num_actors, **kw))
    if "mcga_cylinder" not in env_configurations.configurations:
        env_configurations.register(
            "mcga_cylinder",
            {
                "vecenv_type": "MCGA",
                "env_creator": lambda num_envs, cfg, **kwargs: CylinderVecEnv(cfg=cfg, num_envs=num_envs),
            },
        )


def _load_config(config_path: Path, args) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = make_training_cfg()
    cfg.device = "cuda:0"
    if args.max_steps is not None:
        cfg.max_steps = int(args.max_steps)
    params = config["params"]
    train_cfg = params["config"]
    train_cfg["features"]["observer"] = ProgressAlgoObserver(total_epochs=args.max_epochs)
    realtime_dir = str(Path(args.train_dir) / args.experiment_name / "realtime") if int(args.realtime_interval) > 0 else None
    train_cfg["env_config"]["cfg"] = cfg
    train_cfg["env_config"]["realtime_dir"] = realtime_dir
    train_cfg["env_config"]["realtime_interval"] = int(args.realtime_interval)
    train_cfg["device"] = "cuda:0"
    train_cfg["name"] = args.experiment_name
    train_cfg["train_dir"] = args.train_dir
    if args.num_actors is not None:
        train_cfg["num_actors"] = int(args.num_actors)
    if args.max_epochs is not None:
        train_cfg["max_epochs"] = int(args.max_epochs)
    if args.smoke:
        train_cfg["num_actors"] = 2
        train_cfg["max_epochs"] = 1
        train_cfg["horizon_length"] = 4
        train_cfg["minibatch_size"] = 8
        cfg.max_steps = min(cfg.max_steps, 2)
        cfg.num_rings = 24
        cfg.num_segments = 32
        cfg.voltage_grid_points = 3
        cfg.voltage_refine_levels = 0
        cfg.voltage_refine_points = 3
        cfg.thermal_max_iters = 32
        cfg.transient_max_time_s = 2.0
        cfg.transient_dt_s = 1.0
    batch_size = int(train_cfg["num_actors"]) * int(train_cfg["horizon_length"])
    train_cfg["minibatch_size"] = min(int(train_cfg.get("minibatch_size", batch_size)), batch_size)
    params["seed"] = int(args.seed)
    return config


def _latest_checkpoint(train_dir: Path, experiment_name: str) -> Path:
    experiment_dirs = sorted(
        [p for p in train_dir.glob(f"{experiment_name}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not experiment_dirs:
        raise FileNotFoundError(f"No rl_games experiment directory found in {train_dir} for prefix {experiment_name}")
    nn_dir = experiment_dirs[-1] / "nn"
    checkpoints = sorted(nn_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {nn_dir}")
    return checkpoints[-1]


def _extract_metric(metrics: dict, idx: int = 0) -> dict:
    extracted = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            item = value[idx]
            if item.ndim == 0:
                extracted[key] = float(item.item()) if item.dtype != torch.bool else bool(item.item())
        else:
            extracted[key] = value
    return extracted


def _build_player_config(base_config: dict, eval_env: CylinderVecEnv) -> dict:
    player_config = deepcopy(base_config)
    env_info = {
        "action_space": eval_env.single_action_space,
        "observation_space": eval_env.single_observation_space,
        "state_space": None,
        "use_global_observations": False,
        "agents": 1,
        "value_size": 1,
    }
    player_config["params"]["config"]["env_info"] = env_info
    player_config["params"]["config"]["vec_env"] = None
    player_config["params"]["config"]["player"] = {"deterministic": True}
    return player_config


def _write_rollout_artifacts(output_dir: Path, history: list[dict], summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    if history:
        csv_path = output_dir / "rollout_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _run_final_evaluation(base_config: dict, checkpoint_path: Path, args):
    coarse_cfg = deepcopy(base_config["params"]["config"]["env_config"]["cfg"])
    coarse_cfg.device = "cuda:0"
    coarse_cfg.max_steps = int(args.eval_steps)
    coarse_env = CylinderVecEnv(coarse_cfg, num_envs=1)
    player_runner = Runner()
    player_config = _build_player_config(base_config, coarse_env)
    player_runner.load(player_config)
    player = player_runner.create_player()
    player.restore(str(checkpoint_path))
    player.reset()

    obs, _ = coarse_env.reset()
    action_history = []
    done = False
    step_idx = 0
    while not done and step_idx < int(coarse_cfg.max_steps):
        obs_tensor = torch.as_tensor(obs[0], dtype=torch.float32, device=player.device)
        action = player.get_action(obs_tensor, is_deterministic=True)
        action_np = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action, dtype=np.float32)
        action_history.append(action_np.reshape(-1).astype(np.float32))
        obs, _, terminated, truncated, _ = coarse_env.step(action_np.reshape(1, -1))
        done = bool(terminated[0] or truncated[0])
        step_idx += 1

    eval_cfg = make_eval_cfg()
    eval_cfg.device = "cuda:0"
    eval_cfg.max_steps = int(args.eval_steps)
    if args.smoke:
        eval_cfg.num_segments = 48
        eval_cfg.num_rings = 48
        eval_cfg.max_steps = min(int(args.eval_steps), 2)
        eval_cfg.voltage_grid_points = 3
        eval_cfg.voltage_refine_levels = 0
        eval_cfg.voltage_refine_points = 3
        eval_cfg.thermal_max_iters = 32
        eval_cfg.transient_max_time_s = 2.0
        eval_cfg.transient_dt_s = 1.0
    eval_env = CylinderVecEnv(eval_cfg, num_envs=1)
    obs, _ = eval_env.reset()
    ring_history = [eval_env.ring_radius[0].detach().cpu().numpy().copy()]
    metrics_history = [_extract_metric(eval_env.current_metrics, 0)]
    rollout_history = []
    step_idx = 0
    done = False
    for action_np in action_history:
        obs, reward, terminated, truncated, info = eval_env.step(action_np.reshape(1, -1))
        done = bool(terminated[0] or truncated[0])
        step_idx += 1
        metrics = _extract_metric(eval_env.current_metrics, 0)
        metrics.update(
            {
                "step": step_idx,
                "reward": float(reward[0]),
                "score": float(info["score"][0]),
                "dwell_time_s": float(info["dwell_time_s"][0]),
                "optimal_transient_time_s": float(info["optimal_transient_time_s"][0]),
                "policy_dwell_time_s": float(info["policy_dwell_time_s"][0]),
                "transient_power_w": float(info["transient_power_w"][0]),
                "transient_mean_power_w": float(info["transient_mean_power_w"][0]),
                "transient_objective": float(info["transient_objective"][0]),
                "feasible": bool(eval_env.current_metrics["feasible"][0].item()),
            }
        )
        rollout_history.append(metrics)
        metrics_history.append(metrics)
        ring_history.append(eval_env.ring_radius[0].detach().cpu().numpy().copy())
        if done:
            break

    output_dir = Path(args.final_eval_dir) / checkpoint_path.parent.parent.name
    export_info = export_ring_profile_mesh(
        ring_radius=ring_history[-1],
        height=eval_cfg.height,
        num_segments=eval_cfg.num_segments,
        output_dir=str(output_dir),
        output_name="optimized_cylinder",
        export_step=True,
        freecad_cmd=getattr(eval_cfg, "freecad_cmd", ""),
    )
    animation_info = export_topology_evolution_animation(
        ring_radius_history=ring_history,
        metrics_history=metrics_history,
        height=eval_cfg.height,
        output_dir=str(output_dir),
        output_name="topology_evolution",
    )
    baseline = metrics_history[0]
    final = metrics_history[-1]
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": "cuda:0",
        "grid_mode": "evaluation",
        "num_segments": int(eval_cfg.num_segments),
        "num_rings": int(eval_cfg.num_rings),
        "steps": int(step_idx),
        "baseline_voltage_v": float(baseline["voltage_v"]),
        "final_voltage_v": float(final["voltage_v"]),
        "baseline_initial_power_w": float(baseline["initial_net_band_power_w"]),
        "final_initial_power_w": float(final["initial_net_band_power_w"]),
        "baseline_average_power_w": float(baseline["average_net_band_power_w"]),
        "final_average_power_w": float(final["average_net_band_power_w"]),
        "baseline_lifetime_s": float(baseline["lifetime_s"]),
        "final_lifetime_s": float(final["lifetime_s"]),
        "baseline_max_temp_k": float(baseline["max_temperature_k"]),
        "final_max_temp_k": float(final["max_temperature_k"]),
        "initial_power_ratio": float(final["initial_net_band_power_w"] / max(float(baseline["initial_net_band_power_w"]), 1.0e-9)),
        "average_power_ratio": float(final["average_net_band_power_w"] / max(float(baseline["average_net_band_power_w"]), 1.0e-9)),
        "lifetime_ratio": float(final["lifetime_s"] / max(float(baseline["lifetime_s"]), 1.0e-9)),
        "feature_change_ratio": float(final["feature_change_ratio"]),
        "volume_change_ratio": float(final["volume_change_ratio"]),
        "final_optimal_transient_time_s": float(final.get("optimal_transient_time_s", 0.0)),
        "final_transient_power_w": float(final.get("transient_power_w", 0.0)),
        "final_transient_mean_power_w": float(final.get("transient_mean_power_w", 0.0)),
        "final_transient_objective": float(final.get("transient_objective", 0.0)),
        "feasible": bool(final["feasible"]),
        "gif": animation_info["gif"],
        "mp4": animation_info["mp4"],
        "stl": export_info["stl"],
        "stp": export_info["stp"],
    }
    _write_rollout_artifacts(output_dir, rollout_history, summary)
    print(f"[eval] final artifacts saved to: {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train Phy-DRL cylinder optimizer with rl_games.")
    parser.add_argument("--config", type=str, default="config/rl_games_ppo.yaml")
    parser.add_argument("--train-dir", type=str, default="outputs/rl_runs")
    parser.add_argument("--experiment-name", type=str, default="mcga_phy_drl")
    parser.add_argument("--num-actors", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--final-eval-dir", type=str, default="outputs/final_eval")
    parser.add_argument("--realtime-interval", type=int, default=4, help="Save realtime shape snapshot every N RL steps (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "必须在独显上训练"
    print(f"[rl] training device: {torch.cuda.get_device_name(0)}", flush=True)

    _register_rl_games_env()
    config = _load_config(Path(args.config), args)
    runner = Runner()
    runner.load(config)
    runner.reset()
    runner.run({"train": True, "play": False, "checkpoint": "", "sigma": None})
    checkpoint_path = _latest_checkpoint(Path(args.train_dir), args.experiment_name)

    # Build realtime animation from shape snapshots captured during training.
    if int(args.realtime_interval) > 0:
        realtime_dir = str(Path(args.train_dir) / args.experiment_name / "realtime")
        realtime_anim = build_realtime_animation(realtime_dir, output_name="training_evolution", fps=6)
        if realtime_anim["gif"]:
            print(f"[rl] realtime animation: {realtime_anim['gif']}", flush=True)
        if realtime_anim["mp4"]:
            print(f"[rl] realtime mp4: {realtime_anim['mp4']}", flush=True)

    _run_final_evaluation(config, checkpoint_path, args)


if __name__ == "__main__":
    main()
