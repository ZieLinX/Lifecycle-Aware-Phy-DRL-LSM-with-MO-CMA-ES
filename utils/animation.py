from __future__ import annotations

from pathlib import Path
import io

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def export_topology_evolution_animation(
    ring_radius_history: list[np.ndarray],
    metrics_history: list[dict],
    height: float,
    output_dir: str,
    output_name: str = "topology_evolution",
) -> dict[str, str | None]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gif_path = output_path / f"{output_name}.gif"
    mp4_path = output_path / f"{output_name}.mp4"

    frames = []
    num_rings = len(ring_radius_history[0])
    z_mm = np.linspace(-0.5 * height, 0.5 * height, num_rings) * 1.0e3
    max_radius_mm = max(float(np.max(profile)) for profile in ring_radius_history) * 1.0e3 * 1.15

    for idx, radius_profile in enumerate(ring_radius_history):
        radius_mm = np.asarray(radius_profile, dtype=np.float64) * 1.0e3
        metrics = metrics_history[min(idx, len(metrics_history) - 1)] if metrics_history else {}
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=120)
        ax.fill_between(z_mm, radius_mm, -radius_mm, alpha=0.28)
        ax.plot(z_mm, radius_mm, linewidth=2.0)
        ax.plot(z_mm, -radius_mm, linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlim(float(z_mm[0]), float(z_mm[-1]))
        ax.set_ylim(-max_radius_mm, max_radius_mm)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("radius (mm)")
        ax.set_title("Topology Evolution")
        stats_text = "\n".join(
            [
                f"step={metrics.get('step', idx)}",
                f"V*={metrics.get('rated_voltage_v', 0.0):.2f} V",
                f"P0-3={metrics.get('radiation_power', metrics.get('initial_net_band_power_w', 0.0)):.3f} W",
                f"life={metrics.get('lifetime_ratio', 1.0):.3f}",
                f"feasible={metrics.get('feasible', True)}",
            ]
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    imageio.mimsave(gif_path, frames, duration=0.7)
    mp4_written = None
    try:
        imageio.mimsave(mp4_path, frames, fps=2)
        mp4_written = str(mp4_path)
    except Exception:
        mp4_written = None
    return {"gif": str(gif_path), "mp4": mp4_written}
