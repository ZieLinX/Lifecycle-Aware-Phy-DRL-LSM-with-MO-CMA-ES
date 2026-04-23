from __future__ import annotations

from pathlib import Path
import io
import math

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_MB = 16


def _align_frame(img: np.ndarray, block: int = _MB) -> np.ndarray:
    """Pad image height/width to the nearest multiple of `block` for video codecs."""
    h, w = img.shape[:2]
    nh = math.ceil(h / block) * block
    nw = math.ceil(w / block) * block
    if nh == h and nw == w:
        return img
    padded = np.zeros((nh, nw, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
    padded[:h, :w] = img[..., None] if img.ndim == 2 else img
    return padded


def _render_frame(
    z_mm: np.ndarray,
    radius_mm: np.ndarray,
    max_radius_mm: float,
    metrics: dict,
    step_idx: int,
    figsize: tuple = (8, 5),
    dpi: int = 120,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.fill_between(z_mm, radius_mm, -radius_mm, alpha=0.30, color="#4C9BE8")
    ax.plot(z_mm, radius_mm, linewidth=2.0, color="#1565C0")
    ax.plot(z_mm, -radius_mm, linewidth=2.0, color="#1565C0")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlim(float(z_mm[0]), float(z_mm[-1]))
    ax.set_ylim(-max_radius_mm * 1.08, max_radius_mm * 1.08)
    ax.set_xlabel("Axial position z (mm)")
    ax.set_ylabel("Radius (mm)")
    ax.set_title("Cylinder Topology Evolution", fontsize=12)
    stats_text = "\n".join(
        [
            f"step = {metrics.get('step', step_idx)}",
            f"V*   = {metrics.get('rated_voltage_v', 0.0):.2f} V",
            f"P₀₋₃ = {metrics.get('radiation_power', metrics.get('initial_net_band_power_w', 0.0)):.3f} W",
            f"life = {metrics.get('lifetime_ratio', 1.0):.3f}",
            f"feasible = {metrics.get('feasible', True)}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    frame = imageio.imread(buffer)
    return frame


def export_topology_evolution_animation(
    ring_radius_history: list[np.ndarray],
    metrics_history: list[dict],
    height: float,
    output_dir: str,
    output_name: str = "topology_evolution",
    fps: int = 3,
) -> dict[str, str | None]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gif_path = output_path / f"{output_name}.gif"
    mp4_path = output_path / f"{output_name}.mp4"

    num_rings = len(ring_radius_history[0])
    z_mm = np.linspace(-0.5 * height, 0.5 * height, num_rings) * 1.0e3
    max_radius_mm = max(float(np.max(profile)) for profile in ring_radius_history) * 1.0e3

    frames = []
    for idx, radius_profile in enumerate(ring_radius_history):
        radius_mm = np.asarray(radius_profile, dtype=np.float64) * 1.0e3
        metrics = metrics_history[min(idx, len(metrics_history) - 1)] if metrics_history else {}
        frame = _render_frame(z_mm, radius_mm, max_radius_mm, metrics, idx)
        frames.append(frame)

    gif_duration = max(1.0 / fps, 0.1)
    imageio.mimsave(gif_path, frames, duration=gif_duration, loop=0)

    mp4_written = None
    try:
        aligned = [_align_frame(f) for f in frames]
        with imageio.get_writer(str(mp4_path), fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
            for f in aligned:
                writer.append_data(f)
        mp4_written = str(mp4_path)
    except Exception as exc:
        try:
            with imageio.get_writer(str(mp4_path), fps=fps, macro_block_size=_MB) as writer:
                for f in frames:
                    writer.append_data(f)
            mp4_written = str(mp4_path)
        except Exception:
            mp4_written = None

    return {"gif": str(gif_path), "mp4": mp4_written}


def save_realtime_frame(
    ring_radius: np.ndarray,
    height: float,
    metrics: dict,
    output_dir: str,
    step_idx: int,
) -> str:
    """Save a single cross-section snapshot during training for live monitoring."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    num_rings = len(ring_radius)
    z_mm = np.linspace(-0.5 * height, 0.5 * height, num_rings) * 1.0e3
    radius_mm = np.asarray(ring_radius, dtype=np.float64) * 1.0e3
    max_r = float(np.max(radius_mm))
    frame_path = output_path / f"step_{step_idx:05d}.png"
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.fill_between(z_mm, radius_mm, -radius_mm, alpha=0.30, color="#4C9BE8")
    ax.plot(z_mm, radius_mm, linewidth=2.0, color="#1565C0")
    ax.plot(z_mm, -radius_mm, linewidth=2.0, color="#1565C0")
    ax.set_xlim(float(z_mm[0]), float(z_mm[-1]))
    ax.set_ylim(-max_r * 1.25, max_r * 1.25)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("radius (mm)")
    title_info = (
        f"step={step_idx}  "
        f"V*={metrics.get('rated_voltage_v', 0.0):.1f}V  "
        f"P0-3={metrics.get('initial_net_band_power_w', 0.0):.2f}W  "
        f"life={metrics.get('lifetime_ratio', 1.0):.3f}  "
        f"feasible={metrics.get('feasible', True)}"
    )
    ax.set_title(title_info, fontsize=9)
    fig.tight_layout()
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)
    return str(frame_path)


def build_realtime_animation(realtime_dir: str, output_name: str = "realtime_evolution", fps: int = 4) -> dict[str, str | None]:
    """Assemble all realtime PNG snapshots into a gif/mp4."""
    realtime_path = Path(realtime_dir)
    png_files = sorted(realtime_path.glob("step_*.png"))
    if not png_files:
        return {"gif": None, "mp4": None}
    frames = [imageio.imread(str(p)) for p in png_files]
    return export_topology_evolution_animation(
        ring_radius_history=[],
        metrics_history=[],
        height=0.015,
        output_dir=realtime_dir,
        output_name=output_name,
        fps=fps,
    ) if False else _write_frames(frames, realtime_path, output_name, fps)


def _write_frames(frames: list, output_path: Path, name: str, fps: int) -> dict[str, str | None]:
    gif_path = output_path / f"{name}.gif"
    mp4_path = output_path / f"{name}.mp4"
    imageio.mimsave(gif_path, frames, duration=1.0 / fps, loop=0)
    mp4_written = None
    try:
        aligned = [_align_frame(f) for f in frames]
        with imageio.get_writer(str(mp4_path), fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
            for f in aligned:
                writer.append_data(f)
        mp4_written = str(mp4_path)
    except Exception:
        pass
    return {"gif": str(gif_path), "mp4": mp4_written}
