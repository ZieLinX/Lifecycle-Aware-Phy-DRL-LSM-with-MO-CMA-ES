import json
import os
import csv
import time
import argparse
import importlib

import numpy as np
from config.cylinder_cfg import make_training_cfg
from envs.cylinder_env import CylinderPhysicsEnv
from utils.exporter import export_env_mesh
from utils.planner import plan_action


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

def run_simulation(
    visualize: bool = False,
    pause: float = 0.02,
    headless: bool = False,
    hold: bool = False,
    max_steps_override: int | None = None,
    fast_smoke: bool = False,
):
    """Run offline geometry-optimization rollout for the coupled multiphysics objective."""
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

    cfg = make_training_cfg()
    if max_steps_override is not None:
        cfg.max_steps = int(max_steps_override)
    if fast_smoke:
        cfg.num_segments = 32
        cfg.num_rings = 16
        cfg.search_top_k = min(cfg.search_top_k, 4)
        cfg.search_depth_grid = (0.0, 0.35, 0.70)
        cfg.search_sigma_grid = (cfg.min_sigma, 0.5 * (cfg.min_sigma + cfg.max_sigma))
        cfg.voltage_grid_points = min(cfg.voltage_grid_points, 7)
        cfg.voltage_refine_levels = min(cfg.voltage_refine_levels, 1)
        cfg.max_steps = min(cfg.max_steps, 4)
        cfg.planner_horizon = 1
        cfg.planner_beam_width = min(cfg.planner_beam_width, 2)
        cfg.planner_seed_top_k = min(cfg.planner_seed_top_k, 4)
        cfg.planner_candidate_top_k = min(cfg.planner_candidate_top_k, 2)
        cfg.planner_local_refine_top_k = min(cfg.planner_local_refine_top_k, 1)
    cfg.use_usd_backend = bool(visualize)
    if visualize:
        # Keep constraints unless explicitly disabled in cfg.
        if getattr(cfg, "visualize_disable_constraints", False):
            cfg.terminate_on_constraints = False
        # Keep viewer interactive: reduce lookahead search workload.
        cfg.search_top_k = min(cfg.search_top_k, 4)
        # Include gentle/no-op actions in visualize mode to avoid thermal runaway.
        cfg.search_depth_grid = (0.0, 0.10, 0.25)
        cfg.search_sigma_grid = (cfg.min_sigma, 0.5 * (cfg.min_sigma + cfg.max_sigma))
        cfg.planner_horizon = 1
        cfg.planner_beam_width = 2
        cfg.planner_seed_top_k = min(cfg.planner_seed_top_k, 4)
        cfg.planner_candidate_top_k = min(cfg.planner_candidate_top_k, 2)
        cfg.planner_local_refine_top_k = min(cfg.planner_local_refine_top_k, 1)
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
    baseline_metrics = env.baseline_metrics
    free_energies = []
    rewards = []
    chosen_actions = []
    history = []

    for step_idx in range(cfg.max_steps):
        if simulation_app is not None and not simulation_app.is_running():
            break
        decision = plan_action(env)
        action = decision.action
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
                "score": float(info["score"]),
                "free_energy": float(info["free_energy"]),
                "rated_voltage_v": float(info["rated_voltage_v"]),
                "radiation_power": float(info["radiation_power"]),
                "average_radiation_power": float(info["average_radiation_power"]),
                "mean_temp": float(info["mean_temp"]),
                "max_temp": float(info["max_temp"]),
                "mass_loss_rate": float(info["mass_loss_rate"]),
                "lifetime_s": float(info["lifetime_s"]),
                "lifetime_ratio": float(info["lifetime_ratio"]),
                "band_efficiency": float(info["band_efficiency"]),
                "temperature_uniformity": float(info["temperature_uniformity"]),
                "feature_change_ratio": float(info["feature_change_ratio"]),
                "volume_change_ratio": float(info["volume_change_ratio"]),
                "thermal_iterations": float(info["thermal_iterations"]),
                "thermal_residual_k": float(info["thermal_residual_k"]),
                "thermal_converged": float(info["thermal_converged"]),
                "active_dent_points": int(info["active_dent_points"]),
                "max_dent_depth": float(info["max_dent_depth"]),
            }
        )
        if (step_idx + 1) % cfg.log_interval == 0 or step_idx == 0:
            print(
                f"[{step_idx + 1:03d}/{cfg.max_steps}] "
                f"dScore={reward:.4f}, "
                f"Score={info['score']:.4f}, "
                f"F={info['free_energy']:.4f}, "
                f"V*={info['rated_voltage_v']:.1f}V, "
                f"Tmax={info['max_temp']:.1f}K, "
                f"P0-3={info['radiation_power']:.4f}W, "
                f"Pavg={info['average_radiation_power']:.4f}W, "
                f"life={info['lifetime_ratio']:.3f}, "
                f"lookahead={decision.projected_return:.4f}, "
                f"thermal_iters={int(info['thermal_iterations'])}, "
                f"I={info.get('current_a', 0.0):.1f}A, "
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
    print(f"Score delta  | mean: {np.mean(rewards):.4f}, final: {rewards[-1]:.4f}")

    out_dir = os.path.abspath("outputs")
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(out_dir, "rollout_metrics.csv")
    actions_npy = os.path.join(out_dir, "actions.npy")
    obs_npy = os.path.join(out_dir, "last_observation.npy")
    summary_json = os.path.join(out_dir, "run_summary.json")

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

    final_metrics = env.current_metrics
    if baseline_metrics is not None and final_metrics is not None:
        summary = {
            "device": str(env.device),
            "steps": len(history),
            "baseline_voltage_v": float(baseline_metrics.voltage_v),
            "final_voltage_v": float(final_metrics.voltage_v),
            "baseline_initial_power_w": float(baseline_metrics.initial_net_band_power_w),
            "final_initial_power_w": float(final_metrics.initial_net_band_power_w),
            "baseline_average_power_w": float(baseline_metrics.average_net_band_power_w),
            "final_average_power_w": float(final_metrics.average_net_band_power_w),
            "baseline_lifetime_s": float(baseline_metrics.lifetime_s),
            "final_lifetime_s": float(final_metrics.lifetime_s),
            "baseline_max_temp_k": float(baseline_metrics.max_temperature_k),
            "final_max_temp_k": float(final_metrics.max_temperature_k),
            "initial_power_ratio": float(
                final_metrics.initial_net_band_power_w / max(baseline_metrics.initial_net_band_power_w, 1.0e-9)
            ),
            "average_power_ratio": float(
                final_metrics.average_net_band_power_w / max(baseline_metrics.average_net_band_power_w, 1.0e-9)
            ),
            "lifetime_ratio": float(final_metrics.lifetime_s / max(baseline_metrics.lifetime_s, 1.0e-9)),
            "feature_change_ratio": float(final_metrics.feature_change_ratio),
            "volume_change_ratio": float(final_metrics.volume_change_ratio),
            "feasible": bool(final_metrics.feasible),
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved run summary to: {summary_json}")
        print(
            "[summary] "
            f"P0-3 ratio={summary['initial_power_ratio']:.4f}, "
            f"Pavg ratio={summary['average_power_ratio']:.4f}, "
            f"life ratio={summary['lifetime_ratio']:.4f}, "
            f"feasible={summary['feasible']}",
            flush=True,
        )

    exported = export_env_mesh(
        env=env,
        output_dir=out_dir,
        output_name="optimized_cylinder",
        export_step=True,
    )
    print(f"Exported STL: {exported['stl']}")
    print(f"Export mesh watertight: {exported.get('watertight', False)}")
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
    parser = argparse.ArgumentParser(description="Run offline cylinder geometry optimization.")
    parser.add_argument("--visualize", action="store_true", help="Run with Isaac Sim window live updates.")
    parser.add_argument("--pause", type=float, default=0.2, help="Sleep seconds between visualized steps.")
    parser.add_argument("--headless", action="store_true", help="When visualizing, run Isaac Sim without viewport.")
    parser.add_argument("--hold", action="store_true", help="Keep viewer open after training until manually closed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max simulation steps from config.")
    parser.add_argument("--fast-smoke", action="store_true", help="Run a reduced CPU-friendly smoke configuration.")
    args = parser.parse_args()
    run_simulation(
        visualize=args.visualize,
        pause=args.pause,
        headless=args.headless,
        hold=args.hold,
        max_steps_override=args.max_steps,
        fast_smoke=args.fast_smoke,
    )
