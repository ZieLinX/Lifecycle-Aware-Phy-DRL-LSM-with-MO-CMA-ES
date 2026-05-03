from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


OBJECTIVE_KEYS = [
    "P0_escape_0_3um_w",
    "lifetime_s_full3d",
    "lifecycle_avg_escape_0_3um_w",
    "escape_visibility_factor",
]


class StrategyActor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def _load_rewards(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for row in rows:
        if not all(key in row for key in OBJECTIVE_KEYS):
            continue
        values.append([float(row.get(key, 0.0)) for key in OBJECTIVE_KEYS])
    if not values:
        raise ValueError(f"No objective rows found in {path}")
    arr = np.asarray(values, dtype=np.float32)
    return arr / (np.max(arr, axis=0, keepdims=True) + 1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SAC-style policy scaffold from a true-physics archive.")
    parser.add_argument("--archive", type=str, required=True, help="Path to pareto_archive_full3d.json")
    parser.add_argument("--output", type=str, default="policy_train_metrics.json")
    parser.add_argument("--action-dim", type=int, default=196)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    rewards = _load_rewards(Path(args.archive))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    actor = StrategyActor(latent_dim=OBJECTIVE_KEYS.__len__(), action_dim=int(args.action_dim)).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    for _ in range(max(int(args.updates), 1)):
        weights = torch.rand_like(reward_tensor)
        weights = weights / torch.clamp(torch.sum(weights, dim=1, keepdim=True), min=1.0e-6)
        scalar = torch.sum(weights * reward_tensor, dim=1, keepdim=True)
        action = actor(weights)
        entropy_proxy = torch.mean(torch.square(action))
        loss = -torch.mean(scalar) + 1.0e-3 * entropy_proxy
        opt.zero_grad()
        loss.backward()
        opt.step()
    metrics = {
        "enabled": True,
        "status": "SAC-style actor scaffold initialized from archive; produced actions must be verified by true physics",
        "archive_samples": int(rewards.shape[0]),
        "objective_keys": OBJECTIVE_KEYS,
        "action_dim": int(args.action_dim),
        "true_physics_verification_required": True,
    }
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
