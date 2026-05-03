from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0e300, neginf=0.0)
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    return (arr / (np.max(arr, axis=0, keepdims=True) + 1.0e-6)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SAC-style policy scaffold from a true-physics archive.")
    parser.add_argument("--archive", type=str, required=True, help="Path to pareto_archive_full3d.json")
    parser.add_argument("--output", type=str, default="policy_train_metrics.json")
    parser.add_argument("--action-dim", type=int, default=196)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()

    rewards = _load_rewards(Path(args.archive))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    actor = StrategyActor(latent_dim=OBJECTIVE_KEYS.__len__(), action_dim=int(args.action_dim)).to(device)
    if bool(args.compile) and hasattr(torch, "compile"):
        actor = torch.compile(actor)
    opt = torch.optim.AdamW(actor.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(reward_tensor), batch_size=max(int(args.batch_size), 1), shuffle=True)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for _ in range(max(int(args.updates), 1)):
        for (batch_rewards,) in loader:
            weights = torch.rand_like(batch_rewards)
            weights = weights / torch.clamp(torch.sum(weights, dim=1, keepdim=True), min=1.0e-6)
            scalar = torch.sum(weights * batch_rewards, dim=1, keepdim=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                action = actor(weights)
                entropy_proxy = torch.mean(torch.square(action))
                loss = -torch.mean(scalar) + 1.0e-3 * entropy_proxy
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    metrics = {
        "enabled": True,
        "status": "SAC-style actor scaffold initialized from archive; produced actions must be verified by true physics",
        "archive_samples": int(rewards.shape[0]),
        "objective_keys": OBJECTIVE_KEYS,
        "action_dim": int(args.action_dim),
        "true_physics_verification_required": True,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "amp": bool(use_amp),
        "compiled": bool(args.compile),
    }
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
