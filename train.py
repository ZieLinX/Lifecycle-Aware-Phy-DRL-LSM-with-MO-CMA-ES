import numpy as np
import torch
import os
import csv
import time
import argparse
import importlib
from config.cylinder_cfg import CylinderPhysicsCfg
from envs.cylinder_env import CylinderPhysicsEnv
from utils.exporter import export_env_mesh


def frame_viewport_on_prims(prim_paths):
    """Try to focus active viewport on target prims."""
    try:
        omni_usd = importlib.import_module("omni.usd")
        UsdGeom = importlib.import_module("pxr.UsdGeom")
        viewport_util = importlib.import_module("omni.kit.viewport.utility")
        viewport_api = viewport_util.get_active_viewport()
        omni_cmd = importlib.import_module("omni.kit.commands")
        cam_path = None
        if viewport_api is not None:
            try:
                cam_path = viewport_api.camera_path
            except Exception:
                cam_path = None
        if not cam_path:
            print("[viewer] Skip frame: no active viewport camera path.", flush=True)
            return
        stage = omni_usd.get_context().get_stage()
        if stage is None:
            print("[viewer] Skip frame: no valid stage.", flush=True)
            return
        cam_prim = stage.GetPrimAtPath(cam_path)
        if not cam_prim or not cam_prim.IsValid():
            print(f"[viewer] Skip frame: camera prim missing at {cam_path}", flush=True)
            return
        try:
            camera = UsdGeom.Camera(cam_prim)
            camera.GetClippingRangeAttr().Set((0.001, 100000.0))
        except Exception:
            pass
        omni_cmd.execute(
            "FramePrimsCommand",
            prim_to_move=cam_path,
            prims_to_frame=list(prim_paths),
            aspect_ratio=16.0 / 9.0,
            zoom=0.05,
        )
        print(f"[viewer] Framed viewport on: {prim_paths}, camera={cam_path}", flush=True)
        return
    except Exception as exc:
        print(f"[viewer] Could not auto-frame viewport ({exc}). Manual F may be needed.", flush=True)

def pick_best_action(env: CylinderPhysicsEnv):
    """
    Greedy one-step optimizer.
    It searches candidate dent positions and amplitudes, then picks
    the action that yields the lowest next-step free energy.
    """
    points = env.points
    radial = torch.norm(points[:, :2], dim=1)
    radial_disp = radial - env.cfg.radius

    # Focus on outward bulges and thermal hotspots.
    candidate_k = min(env.cfg.search_top_k, env.num_points)
    top_shape = torch.topk(radial_disp, k=candidate_k, largest=True).indices.tolist()
    top_temp = torch.topk(env.temperature, k=candidate_k, largest=True).indices.tolist()
    top_indices = list(dict.fromkeys(top_shape + top_temp))

    # Skip fixed electrode rings so deformation actions affect free surface.
    if env.cfg.keep_electrode_rings_fixed:
        ring_max = int(torch.max(env.ring_index).item())
        movable_mask = (env.ring_index != 0) & (env.ring_index != ring_max)
        top_indices = [i for i in top_indices if bool(movable_mask[i].item())]

    # If shell is already mostly non-bulged, still allow sparse global probing.
    if float(radial_disp.max().item()) < 1e-4:
        coarse = np.linspace(0, env.num_points - 1, num=min(12, env.num_points), dtype=int).tolist()
        if env.cfg.keep_electrode_rings_fixed:
            ring_max = int(torch.max(env.ring_index).item())
            movable_mask = (env.ring_index != 0) & (env.ring_index != ring_max)
            coarse = [i for i in coarse if bool(movable_mask[i].item())]
        top_indices = list(dict.fromkeys(top_indices + coarse))

    # Fallback: if all candidates got filtered out, sample from movable region.
    if not top_indices:
        if env.cfg.keep_electrode_rings_fixed:
            ring_max = int(torch.max(env.ring_index).item())
            movable = torch.where((env.ring_index != 0) & (env.ring_index != ring_max))[0].tolist()
            top_indices = movable[:candidate_k] if movable else [0]
        else:
            top_indices = list(range(candidate_k))

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
                if float(reward) > best_reward:
                    best_free_energy = free_energy
                    best_reward = float(reward)
                    best_action = action

    if best_action is None:
        best_action = np.array([0.0, 0.0, env.cfg.min_sigma], dtype=np.float32)
    return best_action, best_reward, best_free_energy


def run_simulation(
    visualize: bool = False,
    pause: float = 0.02,
    headless: bool = False,
    hold: bool = False,
    max_steps_override: int | None = None,
):
    """Run optimization policy rollout for coupled multiphysics objective."""
    simulation_app = None
    if visualize:
        SimulationApp = importlib.import_module("isaacsim").SimulationApp
        simulation_app = SimulationApp(
            {
                "headless": headless,
                # Use stable lighting renderer instead of RTX real-time
                # to avoid severe ghosting/flickering in this scene.
                "renderer": "RayTracedLighting",
            }
        )
        try:
            carb_settings = importlib.import_module("carb.settings").get_settings()
            # Disable temporal upscaling style options that often cause motion trails.
            carb_settings.set("/rtx/post/dlss/enabled", False)
            carb_settings.set("/rtx/post/aa/op", 0)
            carb_settings.set("/rtx/directLighting/enabled", True)
            print("[viewer] Applied stable render settings (RayTracedLighting, DLSS off).", flush=True)
        except Exception as exc:
            print(f"[viewer] Render settings override skipped: {exc}", flush=True)

    cfg = CylinderPhysicsCfg()
    if max_steps_override is not None:
        cfg.max_steps = int(max_steps_override)
    cfg.use_usd_backend = bool(visualize)
    if visualize:
        # Visualization mode: run full trajectory unless max_steps reached.
        cfg.terminate_on_constraints = False
        # Keep viewer interactive: reduce lookahead search workload.
        cfg.search_top_k = min(cfg.search_top_k, 4)
        cfg.search_depth_grid = (0.35, 0.70)
        cfg.search_sigma_grid = (cfg.min_sigma, 0.5 * (cfg.min_sigma + cfg.max_sigma))
        cfg.log_interval = max(cfg.log_interval, 20)
    env = CylinderPhysicsEnv(cfg)
    if visualize and not env.use_usd:
        if simulation_app is not None:
            simulation_app.close()
        raise RuntimeError(
            "Visualization requested but USD mesh backend is unavailable. "
            "Check terminal logs for '[viewer]' initialization errors."
        )
    if visualize:
        print(
            f"[viewer] visualize mode: use_usd={env.use_usd}, prim_path={cfg.prim_path}",
            flush=True,
        )
        frame_viewport_on_prims([cfg.prim_path])
    print(
        f"[run] effective max_steps={cfg.max_steps}, "
        f"terminate_on_constraints={cfg.terminate_on_constraints}",
        flush=True,
    )

    obs = env.reset()
    free_energies = []
    rewards = []
    chosen_actions = []
    history = []

    for step_idx in range(cfg.max_steps):
        if simulation_app is not None and not simulation_app.is_running():
            break
        action, _, _ = pick_best_action(env)
        obs, reward, done, info = env.step(action)
        if simulation_app is not None:
            simulation_app.update()
            if pause > 0.0:
                time.sleep(pause)
        chosen_actions.append(action)
        rewards.append(float(reward))
        free_energies.append(float(info["free_energy"]))
        history.append(
            {
                "step": int(info["step"]),
                "reward": float(reward),
                "free_energy": float(info["free_energy"]),
                "radiation_power": float(info["radiation_power"]),
                "mean_temp": float(info["mean_temp"]),
                "max_temp": float(info["max_temp"]),
                "mass_loss_rate": float(info["mass_loss_rate"]),
                "feature_change_ratio": float(info["feature_change_ratio"]),
                "volume_change_ratio": float(info["volume_change_ratio"]),
                "active_dent_points": int(info["active_dent_points"]),
                "max_dent_depth": float(info["max_dent_depth"]),
            }
        )
        if (step_idx + 1) % cfg.log_interval == 0 or step_idx == 0:
            print(
                f"[{step_idx + 1:03d}/{cfg.max_steps}] "
                f"R={reward:.4f}, "
                f"F={info['free_energy']:.4f}, "
                f"Tmax={info['max_temp']:.1f}K, "
                f"Prad={info['radiation_power']:.4f}W, "
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

    out_dir = os.path.abspath("outputs")
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(out_dir, "rollout_metrics.csv")
    actions_npy = os.path.join(out_dir, "actions.npy")
    obs_npy = os.path.join(out_dir, "last_observation.npy")

    if history:
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"Saved metrics to: {metrics_csv}")
    np.save(actions_npy, np.asarray(chosen_actions, dtype=np.float32))
    np.save(obs_npy, obs.astype(np.float32))
    print(f"Saved actions to: {actions_npy}")
    print(f"Saved final observation to: {obs_npy}")

    exported = export_env_mesh(
        env=env,
        output_dir=out_dir,
        output_name="optimized_cylinder",
        export_step=True,
    )
    print(f"Exported STL: {exported['stl']}")
    if exported["stp"] is not None:
        print(f"Exported STP: {exported['stp']}")
    else:
        print("STP export skipped (FreeCADCmd not found or conversion failed).")
    if simulation_app is not None and hold and not headless:
        print("Holding viewer open. Close Isaac Sim window to exit.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.02)
    if simulation_app is not None:
        simulation_app.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cylinder optimization training.")
    parser.add_argument("--visualize", action="store_true", help="Run with Isaac Sim window live updates.")
    parser.add_argument("--pause", type=float, default=0.2, help="Sleep seconds between visualized steps.")
    parser.add_argument("--headless", action="store_true", help="When visualizing, run Isaac Sim without viewport.")
    parser.add_argument("--hold", action="store_true", help="Keep viewer open after training until manually closed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max simulation steps from config.")
    args = parser.parse_args()
    run_simulation(
        visualize=args.visualize,
        pause=args.pause,
        headless=args.headless,
        hold=args.hold,
        max_steps_override=args.max_steps,
    )