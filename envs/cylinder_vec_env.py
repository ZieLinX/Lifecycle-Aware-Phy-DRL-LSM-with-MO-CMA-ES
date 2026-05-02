from __future__ import annotations

import math
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from utils.feasibility import project_connected_profile_batch
from utils.rated_condition import search_rated_condition_batch, simulate_transient_trajectory, summarize_transient_selection


class CylinderVecEnv:
    """
    GPU-friendly vectorized ring-profile environment.

    This environment follows a physics-informed DRL pattern:
    the policy does not edit geometry point-by-point. Instead it emits low-order
    coefficients for three strategy fields (radiation / evaporation / current),
    which are combined with local physical sensitivities into a radius velocity field.
    A Lagrange-style projection then removes the net volume drift.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg, num_envs: int):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.num_rings = int(cfg.num_rings)
        self.num_basis = 6
        self.action_dim = self.num_basis * 3 + 1
        self.global_dim = 10
        self.single_action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.single_observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_rings * 3 + self.global_dim,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, self.single_observation_space.shape[0]),
            dtype=np.float32,
        )
        self.initial_volume = float(math.pi * cfg.radius * cfg.radius * cfg.height)
        self.rest_ring_radius = torch.full(
            (self.num_envs, self.num_rings),
            float(cfg.radius),
            dtype=torch.float32,
            device=self.device,
        )
        self.ring_radius = self.rest_ring_radius.clone()
        self.recession_depth = torch.zeros_like(self.ring_radius)
        self.temperature = torch.full_like(self.ring_radius, float(cfg.ambient_temp))
        self.current_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.baseline_metrics = None
        self.current_metrics = None
        self.last_score = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._basis = self._build_basis(self.num_rings, self.num_basis, self.device)

    @staticmethod
    def _build_basis(num_rings: int, num_basis: int, device: torch.device) -> torch.Tensor:
        z = torch.linspace(-1.0, 1.0, num_rings, device=device)
        basis = [torch.ones_like(z)]
        if num_basis > 1:
            basis.append(z)
        for order in range(2, num_basis):
            basis.append(2.0 * z * basis[-1] - basis[-2])
        return torch.stack(basis[:num_basis], dim=0)

    def get_number_of_agents(self) -> int:
        return 1

    def _metrics_to_dict(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}

    def _evaluate_geometry(self) -> Dict[str, torch.Tensor]:
        return search_rated_condition_batch(self.cfg, self.ring_radius, self.initial_volume)

    def _strategy_fields(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coeffs = actions[:, :-1].reshape(self.num_envs, 3, self.num_basis)
        alpha = torch.sigmoid(torch.einsum("ebk,kr->ebr", coeffs, self._basis))
        dwell = 0.5 * (actions[:, -1] + 1.0)
        return alpha[:, 0], alpha[:, 1], alpha[:, 2], dwell

    def _physical_sensitivities(self, metrics: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = metrics["ring_temperature_k"]
        recession_rate = metrics["ring_recession_rate_m_s"]
        temp_scale = torch.clamp(temperature - float(self.cfg.ambient_temp), min=0.0)
        temp_scale = temp_scale / torch.clamp(torch.amax(temp_scale, dim=1, keepdim=True), min=1.0)
        rad = temp_scale.pow(4)
        evap = recession_rate / torch.clamp(torch.amax(recession_rate, dim=1, keepdim=True), min=1.0e-12)
        cur = (1.0 / torch.clamp(self.ring_radius, min=float(self.cfg.min_radius)).pow(2))
        cur = cur / torch.clamp(torch.amax(cur, dim=1, keepdim=True), min=1.0e-12)
        return rad, evap, cur

    def _project_volume_preserving_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        dz = float(self.cfg.height) / max(self.num_rings - 1, 1)
        weights = 2.0 * math.pi * self.ring_radius * dz
        projected = velocity.clone()
        projected[:, 0] = 0.0
        projected[:, -1] = 0.0
        lambda_term = torch.sum(weights * projected, dim=1, keepdim=True) / torch.clamp(torch.sum(weights, dim=1, keepdim=True), min=1.0e-12)
        projected = projected - lambda_term
        projected[:, 0] = 0.0
        projected[:, -1] = 0.0
        return projected

    def _apply_action(self, actions: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        alpha_rad, alpha_evap, alpha_cur, dwell = self._strategy_fields(actions)
        sens_rad, sens_evap, sens_cur = self._physical_sensitivities(metrics)
        raw_velocity = alpha_rad * sens_rad - alpha_evap * sens_evap + alpha_cur * sens_cur
        raw_velocity = raw_velocity - torch.mean(raw_velocity, dim=1, keepdim=True)
        projected = self._project_volume_preserving_velocity(raw_velocity)
        scale = float(self.cfg.max_depth) * 0.35
        self.ring_radius = torch.clamp(
            self.ring_radius + scale * projected,
            min=float(self.cfg.min_radius),
        )
        # Electrode rings are fixed at nominal radius.
        self.ring_radius[:, 0] = float(self.cfg.radius)
        self.ring_radius[:, -1] = float(self.cfg.radius)
        # Enforce geometric connectivity: project onto slope-limited connected profile
        # so that no adjacent cross-section ratio can create a physically floating island.
        max_step = float(getattr(self.cfg, "feasibility_area_ratio_max", 5.0)) ** 0.5
        self.ring_radius = project_connected_profile_batch(
            self.ring_radius,
            min_radius=float(self.cfg.min_radius),
            max_step_ratio=max_step,
            fix_endpoints=bool(getattr(self.cfg, "keep_electrode_rings_fixed", True)),
        )
        return dwell

    def _transient_summary(self, metrics: Dict[str, torch.Tensor], dwell: torch.Tensor) -> Dict[str, torch.Tensor]:
        horizon_s = float(getattr(self.cfg, "transient_max_time_s", self.cfg.lifecycle_reference_s))
        policy_dwell_time_s = dwell * horizon_s
        transient = simulate_transient_trajectory(
            cfg=self.cfg,
            ring_radius=self.ring_radius,
            voltage_schedule=metrics["voltage_v"] * float(getattr(self.cfg, "transient_default_voltage_ratio", 1.0)),
            t_max=horizon_s,
            dt=float(self.cfg.transient_dt_s),
        )
        baseline_power = torch.clamp(self.baseline_metrics["initial_net_band_power_w"], min=1.0e-9)
        return summarize_transient_selection(
            self.cfg,
            transient,
            baseline_power_w=baseline_power,
            dwell_time_s=policy_dwell_time_s,
        )

    def _score(self, metrics: Dict[str, torch.Tensor], transient: Dict[str, torch.Tensor]) -> torch.Tensor:
        baseline = self.baseline_metrics
        initial_ratio = metrics["initial_net_band_power_w"] / torch.clamp(baseline["initial_net_band_power_w"], min=1.0e-9)
        average_ratio = metrics["average_net_band_power_w"] / torch.clamp(baseline["average_net_band_power_w"], min=1.0e-9)
        lifetime_ratio = metrics["lifetime_s"] / torch.clamp(baseline["lifetime_s"], min=1.0e-9)
        reward_lifetime_ratio = torch.clamp(
            lifetime_ratio,
            max=float(getattr(self.cfg, "reward_lifetime_ratio_cap", 5.0)),
        )
        score = (
            float(self.cfg.reward_weight_initial_power) * initial_ratio
            + float(self.cfg.reward_weight_average_power) * average_ratio
            + float(self.cfg.reward_weight_lifetime) * reward_lifetime_ratio
            + float(self.cfg.reward_weight_uniformity) * metrics["temperature_uniformity"]
            + float(self.cfg.reward_weight_efficiency) * metrics["band_efficiency"]
            + float(self.cfg.reward_weight_transient_power) * transient["transient_power_ratio"]
            - float(self.cfg.reward_penalty_feasibility) * metrics["feasibility_penalty"]
            - float(self.cfg.reward_weight_thermomech) * metrics["thermo_mech_penalty"]
            - float(self.cfg.reward_penalty_mass_loss) * torch.clamp(metrics["mass_loss_rate_kg_s"] - float(self.cfg.max_mass_loss_rate), min=0.0)
            - float(self.cfg.reward_penalty_temp_violation) * torch.clamp(metrics["max_temperature_k"] - float(self.cfg.max_temp), min=0.0).pow(2)
            - float(self.cfg.reward_penalty_feature_violation) * torch.clamp(metrics["feature_change_ratio"] - float(self.cfg.feature_fail_ratio), min=0.0)
            - float(self.cfg.reward_penalty_volume_change) * metrics["volume_change_ratio"]
        )
        return score

    def _build_obs(self) -> np.ndarray:
        metrics = self.current_metrics
        baseline = self.baseline_metrics
        radial_disp = (self.ring_radius - float(self.cfg.radius)) / max(float(self.cfg.radius), 1.0e-12)
        temp_norm = (self.temperature - float(self.cfg.ambient_temp)) / max(float(self.cfg.max_temp - self.cfg.ambient_temp), 1.0)
        ablation_norm = self.recession_depth / max(float(self.cfg.feature_fail_ratio * self.cfg.radius), 1.0e-12)
        lifetime_ratio_obs = metrics["lifetime_s"] / torch.clamp(baseline["lifetime_s"], min=1.0e-9)
        lifetime_ratio_obs = torch.clamp(
            lifetime_ratio_obs,
            max=float(getattr(self.cfg, "observation_lifetime_ratio_cap", 5.0)),
        )
        global_obs = torch.stack(
            [
                metrics["voltage_v"] / max(float(self.cfg.max_voltage), 1.0),
                metrics["initial_net_band_power_w"] / torch.clamp(baseline["initial_net_band_power_w"], min=1.0e-9),
                metrics["average_net_band_power_w"] / torch.clamp(baseline["average_net_band_power_w"], min=1.0e-9),
                metrics["max_temperature_k"] / max(float(self.cfg.max_temp), 1.0),
                lifetime_ratio_obs,
                metrics["view_factor_proxy"],
                metrics["feature_change_ratio"],
                metrics["volume_change_ratio"],
                metrics["feasibility_penalty"],
                metrics["thermo_mech_penalty"],
            ],
            dim=1,
        )
        obs = torch.cat([radial_disp, temp_norm, ablation_norm, global_obs], dim=1)
        return obs.detach().cpu().numpy().astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.ring_radius = self.rest_ring_radius.clone()
        self.recession_depth.zero_()
        self.temperature.fill_(float(self.cfg.ambient_temp))
        self.current_step.zero_()
        self.done.zero_()
        self.current_metrics = self._evaluate_geometry()
        self.baseline_metrics = self._metrics_to_dict(self.current_metrics)
        self.last_score = self._score(
            self.current_metrics,
            {
                "transient_power_ratio": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            },
        )
        self.temperature = self.current_metrics["ring_temperature_k"].clone()
        self.recession_depth = self.current_metrics["ring_recession_rate_m_s"].clone() * float(self.cfg.ablation_observation_horizon_s)
        return self._build_obs(), {}

    def reset_done(self, done_mask: np.ndarray | torch.Tensor):
        mask = torch.as_tensor(done_mask, dtype=torch.bool, device=self.device)
        if not bool(torch.any(mask).item()):
            return self._build_obs()
        self.ring_radius[mask] = float(self.cfg.radius)
        self.recession_depth[mask] = 0.0
        self.temperature[mask] = float(self.cfg.ambient_temp)
        self.current_step[mask] = 0
        self.done[mask] = False
        refreshed = self._evaluate_geometry()
        for key, value in refreshed.items():
            if isinstance(value, torch.Tensor):
                self.current_metrics[key][mask] = value[mask]
                self.baseline_metrics[key][mask] = value[mask]
        self.last_score[mask] = 0.0
        self.temperature[mask] = self.current_metrics["ring_temperature_k"][mask]
        self.recession_depth[mask] = self.current_metrics["ring_recession_rate_m_s"][mask] * float(self.cfg.ablation_observation_horizon_s)
        return self._build_obs()

    def step(self, actions):
        action_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device).reshape(self.num_envs, self.action_dim)
        previous_score = self.last_score.clone()
        prev_metrics = self.current_metrics
        dwell = self._apply_action(action_t, prev_metrics)
        self.current_metrics = self._evaluate_geometry()
        self.temperature = self.current_metrics["ring_temperature_k"].clone()
        self.recession_depth = self.current_metrics["ring_recession_rate_m_s"].clone() * float(self.cfg.ablation_observation_horizon_s)
        transient = self._transient_summary(self.current_metrics, dwell)
        score = self._score(self.current_metrics, transient)
        reward = score - previous_score
        self.last_score = score
        self.current_step += 1

        life_ratio = self.current_metrics["lifetime_s"] / torch.clamp(self.baseline_metrics["lifetime_s"], min=1.0e-9)
        fail = (
            (self.current_metrics["feature_change_ratio"] >= float(self.cfg.feature_fail_ratio))
            | (self.current_metrics["max_temperature_k"] > float(self.cfg.max_temp))
            | (self.current_metrics["volume_change_ratio"] > float(self.cfg.volume_tolerance_ratio))
            | (life_ratio < float(self.cfg.minimum_lifetime_ratio))
        )
        self.done = self.done | fail | (self.current_step >= int(self.cfg.max_steps))
        info = {
            "score": score.detach().cpu().numpy(),
            "reward_delta": reward.detach().cpu().numpy(),
            "rated_voltage_v": self.current_metrics["voltage_v"].detach().cpu().numpy(),
            "transient_power_w": transient["transient_power_w"].detach().cpu().numpy(),
            "dwell_time_s": transient["dwell_time_s"].detach().cpu().numpy(),
            "optimal_transient_time_s": transient["optimal_transient_time_s"].detach().cpu().numpy(),
            "policy_dwell_time_s": transient["policy_dwell_time_s"].detach().cpu().numpy(),
            "transient_mean_power_w": transient["transient_mean_power_w"].detach().cpu().numpy(),
            "transient_objective": transient["transient_objective"].detach().cpu().numpy(),
            "feasibility_penalty": self.current_metrics["feasibility_penalty"].detach().cpu().numpy(),
            "thermo_mech_penalty": self.current_metrics["thermo_mech_penalty"].detach().cpu().numpy(),
        }
        return self._build_obs(), reward.detach().cpu().numpy().astype(np.float32), self.done.detach().cpu().numpy(), np.zeros(self.num_envs, dtype=bool), info
