from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from envs.cylinder_env import CylinderPhysicsEnv


@dataclass
class ActionCandidate:
    action: np.ndarray
    reward: float
    score: float
    free_energy: float
    feasible: bool
    info: Dict[str, float]


@dataclass
class PlannerDecision:
    action: np.ndarray
    immediate_reward: float
    projected_return: float
    info: Dict[str, float]


def _normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    vmin = torch.min(values)
    vmax = torch.max(values)
    if float((vmax - vmin).item()) < 1.0e-12:
        return torch.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _movable_mask(env: "CylinderPhysicsEnv") -> torch.Tensor:
    if not env.cfg.keep_electrode_rings_fixed:
        return torch.ones(env.num_points, dtype=torch.bool, device=env.device)
    ring_max = int(torch.max(env.ring_index).item())
    return (env.ring_index != 0) & (env.ring_index != ring_max)


def _candidate_point_score(env: "CylinderPhysicsEnv") -> torch.Tensor:
    radial = torch.norm(env.points[:, :2], dim=1)
    shape = torch.clamp(radial - env.cfg.radius, min=0.0)
    temp = torch.clamp(env.temperature - env.cfg.ambient_temp, min=0.0)
    ablation = torch.clamp(env.ablation_depth, min=0.0)
    score = (
        env.cfg.planner_weight_shape * _normalize(shape)
        + env.cfg.planner_weight_temp * _normalize(temp)
        + env.cfg.planner_weight_ablation * _normalize(ablation)
    )
    score = score * _movable_mask(env).float()
    return score


def _coarse_probe_indices(env: "CylinderPhysicsEnv", max_count: int) -> list[int]:
    movable = torch.where(_movable_mask(env))[0].tolist()
    if not movable:
        return [0]
    if len(movable) <= max_count:
        return movable
    picks = np.linspace(0, len(movable) - 1, num=max_count, dtype=int)
    return [movable[i] for i in picks.tolist()]


def _top_candidate_indices(env: "CylinderPhysicsEnv") -> list[int]:
    point_score = _candidate_point_score(env)
    candidate_k = min(int(env.cfg.planner_seed_top_k), env.num_points)
    top_dynamic = torch.topk(point_score, k=candidate_k, largest=True).indices.tolist()
    coarse = _coarse_probe_indices(env, max_count=max(4, candidate_k // 2))
    merged = []
    seen = set()
    for idx in top_dynamic + coarse:
        if idx not in seen and bool(_movable_mask(env)[idx].item()):
            seen.add(idx)
            merged.append(int(idx))
    return merged[: max(candidate_k, 1)]


def _depth_candidates(env: "CylinderPhysicsEnv", severity: float) -> list[float]:
    positive_depths = [float(v) for v in env.cfg.search_depth_grid if float(v) > 0.0]
    if not positive_depths:
        return [0.0]
    if severity < 0.35:
        return positive_depths[:2]
    if severity < 0.70:
        return positive_depths[1:3] if len(positive_depths) >= 3 else positive_depths[:2]
    return positive_depths[-2:]


def _sigma_candidates(env: "CylinderPhysicsEnv", idx: int) -> list[float]:
    sigma_grid = [float(v) for v in env.cfg.search_sigma_grid]
    if len(sigma_grid) <= 2:
        return sigma_grid
    temp_norm = float(
        torch.clamp(
            (env.temperature[idx] - env.cfg.ambient_temp) / max(env.cfg.max_temp - env.cfg.ambient_temp, 1.0),
            0.0,
            1.0,
        ).item()
    )
    ablation_norm = float(
        torch.clamp(
            env.ablation_depth[idx] / max(env.cfg.feature_fail_ratio * env.cfg.radius, 1.0e-12),
            0.0,
            1.0,
        ).item()
    )
    if temp_norm + ablation_norm > 0.8:
        return sigma_grid[-2:]
    return sigma_grid[:2]


def _coarse_action_pool(env: "CylinderPhysicsEnv") -> list[np.ndarray]:
    actions: list[np.ndarray] = [np.asarray([0.0, 0.0, env.cfg.min_sigma], dtype=np.float32)]
    point_score = _candidate_point_score(env)
    for idx in _top_candidate_indices(env):
        idx_ratio = idx / max(env.num_points - 1, 1)
        severity = float(torch.clamp(point_score[idx], 0.0, 1.0).item())
        for depth in _depth_candidates(env, severity):
            for sigma in _sigma_candidates(env, idx):
                actions.append(np.asarray([idx_ratio, depth, sigma], dtype=np.float32))
    return actions


def _action_key(action: np.ndarray) -> tuple[float, float, float]:
    return (round(float(action[0]), 6), round(float(action[1]), 6), round(float(action[2]), 7))


def _is_feasible(env: "CylinderPhysicsEnv", info: Dict[str, float]) -> bool:
    return (
        float(info["max_temp"]) <= float(env.cfg.max_temp)
        and float(info["lifetime_ratio"]) >= float(env.cfg.minimum_lifetime_ratio)
        and float(info["volume_change_ratio"]) <= float(env.cfg.volume_tolerance_ratio)
        and float(info["feature_change_ratio"]) < float(env.cfg.feature_fail_ratio)
    )


def _evaluate_candidates(
    env: "CylinderPhysicsEnv",
    actions: Iterable[np.ndarray],
) -> list[ActionCandidate]:
    evaluated: list[ActionCandidate] = []
    seen = set()
    for action in actions:
        key = _action_key(action)
        if key in seen:
            continue
        seen.add(key)
        reward, _, info = env.evaluate_action(action)
        evaluated.append(
            ActionCandidate(
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                score=float(info["score"]),
                free_energy=float(info["free_energy"]),
                feasible=_is_feasible(env, info),
                info={k: float(v) for k, v in info.items()},
            )
        )
    evaluated.sort(
        key=lambda item: (
            1 if item.feasible else 0,
            item.reward,
            item.score,
            -item.free_energy,
        ),
        reverse=True,
    )
    return evaluated


def _neighbor_indices(env: "CylinderPhysicsEnv", base_idx: int, span: int) -> list[int]:
    base_ring = base_idx // env.cfg.num_segments
    base_seg = base_idx % env.cfg.num_segments
    ring_min = 1 if env.cfg.keep_electrode_rings_fixed else 0
    ring_max = env.cfg.num_rings - 2 if env.cfg.keep_electrode_rings_fixed else env.cfg.num_rings - 1
    neighbors = []
    for ring_offset in range(-span, span + 1):
        new_ring = min(max(base_ring + ring_offset, ring_min), ring_max)
        for seg_offset in range(-span, span + 1):
            new_seg = (base_seg + seg_offset) % env.cfg.num_segments
            neighbors.append(new_ring * env.cfg.num_segments + new_seg)
    deduped = []
    seen = set()
    for idx in neighbors:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def _refined_action_pool(env: "CylinderPhysicsEnv", seed: ActionCandidate) -> list[np.ndarray]:
    base = seed.action
    base_idx = int(round(float(base[0]) * max(env.num_points - 1, 1)))
    base_depth = float(base[1])
    base_sigma = float(base[2])
    depth_cap = max(float(v) for v in env.cfg.search_depth_grid)
    actions = [base]
    for idx in _neighbor_indices(env, base_idx, int(env.cfg.planner_refine_neighbor_span)):
        idx_ratio = idx / max(env.num_points - 1, 1)
        for depth_scale in env.cfg.planner_depth_scale_factors:
            depth = float(np.clip(base_depth * float(depth_scale), 0.0, depth_cap))
            for sigma_scale in env.cfg.planner_sigma_scale_factors:
                sigma = float(np.clip(base_sigma * float(sigma_scale), env.cfg.min_sigma, env.cfg.max_sigma))
                actions.append(np.asarray([idx_ratio, depth, sigma], dtype=np.float32))
    return actions


def propose_actions(env: "CylinderPhysicsEnv") -> list[ActionCandidate]:
    coarse_candidates = _evaluate_candidates(env, _coarse_action_pool(env))
    coarse_top = coarse_candidates[: max(int(env.cfg.planner_candidate_top_k), 1)]
    refined_candidates = list(coarse_top)
    for seed in coarse_top[: max(int(env.cfg.planner_local_refine_top_k), 0)]:
        refined_candidates.extend(_evaluate_candidates(env, _refined_action_pool(env, seed)))
    deduped: dict[tuple[float, float, float], ActionCandidate] = {}
    for cand in refined_candidates:
        key = _action_key(cand.action)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = cand
            continue
        if (
            (1 if cand.feasible else 0, cand.reward, cand.score, -cand.free_energy)
            > (1 if existing.feasible else 0, existing.reward, existing.score, -existing.free_energy)
        ):
            deduped[key] = cand
    merged = list(deduped.values())
    merged.sort(
        key=lambda item: (
            1 if item.feasible else 0,
            item.reward,
            item.score,
            -item.free_energy,
        ),
        reverse=True,
    )
    return merged[: max(int(env.cfg.planner_beam_width), 1)]


def _search(env: "CylinderPhysicsEnv", horizon: int) -> PlannerDecision:
    candidates = propose_actions(env)
    if not candidates:
        action = np.asarray([0.0, 0.0, env.cfg.min_sigma], dtype=np.float32)
        reward, _, info = env.evaluate_action(action)
        return PlannerDecision(
            action=action,
            immediate_reward=float(reward),
            projected_return=float(reward),
            info={k: float(v) for k, v in info.items()},
        )

    best_action = candidates[0].action
    best_immediate = candidates[0].reward
    best_projected = -float("inf")
    best_info = candidates[0].info

    for candidate in candidates[: max(int(env.cfg.planner_beam_width), 1)]:
        snapshot = env.get_state()
        try:
            _, reward, done, info = env.step(candidate.action)
            projected = float(reward)
            if horizon > 1 and not done:
                projected += _search(env, horizon - 1).projected_return
            if projected > best_projected:
                best_projected = projected
                best_action = candidate.action
                best_immediate = candidate.reward
                best_info = {k: float(v) for k, v in info.items()}
        finally:
            env.set_state(snapshot)

    return PlannerDecision(
        action=np.asarray(best_action, dtype=np.float32),
        immediate_reward=float(best_immediate),
        projected_return=float(best_projected),
        info=best_info,
    )


def plan_action(env: "CylinderPhysicsEnv") -> PlannerDecision:
    snapshot = env.get_state()
    try:
        return _search(env, max(int(env.cfg.planner_horizon), 1))
    finally:
        env.set_state(snapshot)
