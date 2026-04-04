import numpy as np
import torch
from config.cylinder_cfg import CylinderPhysicsCfg
from envs.cylinder_env import CylinderPhysicsEnv


def pick_best_action(env: CylinderPhysicsEnv):
    """
    Greedy one-step optimizer.
    It searches candidate dent positions and amplitudes, then picks
    the action that yields the lowest next-step free energy.
    """
    points = env.points
    radial = torch.norm(points[:, :2], dim=1)
    radial_disp = radial - env.cfg.radius

    # Focus on outward bulges first: these are best candidates for inward dents.
    candidate_k = min(env.cfg.search_top_k, env.num_points)
    top_indices = torch.topk(radial_disp, k=candidate_k, largest=True).indices.tolist()

    # If shell is already mostly non-bulged, still allow sparse global probing.
    if float(radial_disp.max().item()) < 1e-4:
        coarse = np.linspace(0, env.num_points - 1, num=min(12, env.num_points), dtype=int).tolist()
        top_indices = list(dict.fromkeys(top_indices + coarse))

    depth_grid = list(env.cfg.search_depth_grid)
    sigma_grid = list(env.cfg.search_sigma_grid)

    best_action = None
    best_free_energy = float("inf")
    best_reward = -float("inf")

    for idx in top_indices:
        idx_ratio = idx / max(1, env.num_points - 1)
        for depth in depth_grid:
            for sigma in sigma_grid:
                action = np.array([idx_ratio, depth, sigma], dtype=np.float32)
                reward, _, info = env.evaluate_action(action)
                free_energy = float(info["free_energy"])
                if free_energy < best_free_energy:
                    best_free_energy = free_energy
                    best_reward = float(reward)
                    best_action = action

    if best_action is None:
        best_action = np.array([0.0, 0.0, env.cfg.min_sigma], dtype=np.float32)
    return best_action, best_reward, best_free_energy


def run_simulation():
    """Run optimization policy rollout for cumulative dent placement."""
    cfg = CylinderPhysicsCfg()
    env = CylinderPhysicsEnv(cfg)

    obs = env.reset()
    free_energies = []
    rewards = []
    chosen_actions = []

    for step_idx in range(cfg.max_steps):
        action, _, _ = pick_best_action(env)
        obs, reward, done, info = env.step(action)
        chosen_actions.append(action)
        rewards.append(float(reward))
        free_energies.append(float(info["free_energy"]))
        if (step_idx + 1) % cfg.log_interval == 0 or step_idx == 0:
            print(
                f"[{step_idx + 1:03d}/{cfg.max_steps}] "
                f"F={info['free_energy']:.4f}, "
                f"active_dents={int(info['active_dent_points'])}, "
                f"max_dent={info['max_dent_depth']:.4f}",
                flush=True,
            )
        if done:
            break

    print("Simulation finished.")
    print(f"Observation dim: {obs.shape[0]}")
    print(f"Steps: {len(free_energies)}")
    print(f"Best policy actions sampled: {len(chosen_actions)}")
    print(f"Free energy | min: {min(free_energies):.4f}, max: {max(free_energies):.4f}, final: {free_energies[-1]:.4f}")
    print(f"Reward      | mean: {np.mean(rewards):.4f}, final: {rewards[-1]:.4f}")

if __name__ == "__main__":
    run_simulation()