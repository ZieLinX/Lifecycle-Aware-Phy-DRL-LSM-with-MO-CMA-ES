from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


TARGET_KEYS = [
    "P0_escape_0_3um_w",
    "lifetime_s_full3d",
    "lifecycle_avg_escape_0_3um_w",
    "escape_visibility_factor",
    "max_temperature_k",
]


FEATURE_KEYS = [
    "action_norm_full3d",
    "voltage_v",
    "electrode_contact_area_m2",
    "surface_area_ratio",
    "effective_radiating_area_m2",
    "axial_resistance_shape_factor_m_inv",
]


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    x_rows = []
    y_rows = []
    for row in rows:
        if not all(key in row for key in TARGET_KEYS):
            continue
        x_rows.append([float(row.get(key, 0.0)) for key in FEATURE_KEYS])
        y_rows.append([float(row.get(key, 0.0)) for key in TARGET_KEYS])
    if not x_rows:
        raise ValueError(f"No usable rows found in {path}")
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the full3d archive surrogate scaffold.")
    parser.add_argument("--archive", type=str, required=True, help="Path to pareto_archive_full3d.json")
    parser.add_argument("--output", type=str, default="surrogate_train_metrics.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()

    x_np, y_np = _load_archive(Path(args.archive))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    x_safe = np.nan_to_num(x_np, nan=0.0, posinf=1.0e300, neginf=-1.0e300)
    y_safe = np.nan_to_num(y_np, nan=0.0, posinf=1.0e300, neginf=-1.0e300)
    y_safe = np.sign(y_safe) * np.log1p(np.abs(y_safe))
    x_mean = x_safe.mean(axis=0, keepdims=True)
    x_std = x_safe.std(axis=0, keepdims=True) + 1.0e-6
    y_mean = y_safe.mean(axis=0, keepdims=True)
    y_std = y_safe.std(axis=0, keepdims=True) + 1.0e-6
    x = torch.as_tensor((x_safe - x_mean) / x_std, dtype=torch.float32, device=device)
    y = torch.as_tensor((y_safe - y_mean) / y_std, dtype=torch.float32, device=device)
    model = SurrogateMLP(x.shape[1], y.shape[1]).to(device)
    if bool(args.compile) and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=max(int(args.batch_size), 1), shuffle=True)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for _ in range(max(int(args.epochs), 1)):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xb)
                loss = torch.mean((pred - yb) ** 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    with torch.no_grad():
        pred = model(x)
        mae = torch.mean(torch.abs(pred - y), dim=0).detach().cpu().numpy()
    metrics = {
        "enabled": True,
        "status": "trained_on_true_physics_archive_for_candidate_prescreening",
        "samples": int(x_np.shape[0]),
        "feature_keys": FEATURE_KEYS,
        "target_keys": TARGET_KEYS,
        "normalized_mae": {key: float(value) for key, value in zip(TARGET_KEYS, mae)},
        "surrogate_only_final_results_allowed": False,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "amp": bool(use_amp),
        "compiled": bool(args.compile),
    }
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
