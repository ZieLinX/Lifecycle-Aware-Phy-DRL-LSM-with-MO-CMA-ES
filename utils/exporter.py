import os
import shutil
import subprocess
import tempfile
import importlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh


def build_cylinder_surface_faces(num_segments: int, num_rings: int) -> np.ndarray:
    """Build triangular side faces for a ring-structured cylinder lattice."""
    faces = []
    for r_idx in range(num_rings - 1):
        ring0 = r_idx * num_segments
        ring1 = (r_idx + 1) * num_segments
        for s_idx in range(num_segments):
            s_next = (s_idx + 1) % num_segments
            i0 = ring0 + s_idx
            i1 = ring0 + s_next
            i2 = ring1 + s_idx
            i3 = ring1 + s_next
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
    return np.asarray(faces, dtype=np.int64)


def _run_freecad_stl_to_step(stl_path: str, step_path: str, freecad_cmd: str = "FreeCADCmd") -> bool:
    """
    Convert STL mesh to STEP using FreeCAD command-line.
    Returns True on success, False otherwise.
    """
    if shutil.which(freecad_cmd) is None:
        return False

    script = f"""
import Mesh
import Part

stl_path = r"{stl_path}"
step_path = r"{step_path}"

mesh = Mesh.Mesh(stl_path)
shape = Part.Shape()
shape.makeShapeFromMesh(mesh.Topology, 0.01)
solid = Part.makeSolid(shape)
Part.export([solid], step_path)
print("STEP exported:", step_path)
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(script)
        tmp_script = tmp.name

    try:
        proc = subprocess.run(
            [freecad_cmd, tmp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and os.path.exists(step_path)
    finally:
        try:
            os.remove(tmp_script)
        except OSError:
            pass


def export_mesh_files(
    points: np.ndarray,
    num_segments: int,
    num_rings: int,
    output_dir: str,
    output_name: str,
    export_step: bool = True,
    freecad_cmd: str = "FreeCADCmd",
) -> Dict[str, Optional[str]]:
    """
    Export current cylinder mesh to STL and optionally STEP/STP.
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    stl_path = str(output_path / f"{output_name}.stl")
    stp_path = str(output_path / f"{output_name}.stp")

    faces = build_cylinder_surface_faces(num_segments=num_segments, num_rings=num_rings)
    mesh = trimesh.Trimesh(vertices=np.asarray(points, dtype=np.float64), faces=faces, process=False)
    mesh.export(stl_path)

    step_written = False
    if export_step:
        step_written = _run_freecad_stl_to_step(stl_path=stl_path, step_path=stp_path, freecad_cmd=freecad_cmd)

    return {
        "stl": stl_path,
        "stp": stp_path if step_written else None,
    }


def export_env_mesh(env, output_dir: str, output_name: str = "optimized_cylinder", export_step: bool = True):
    """Convenience wrapper: export mesh directly from env state."""
    points_np = env.points.detach().cpu().numpy()
    return export_mesh_files(
        points=points_np,
        num_segments=env.cfg.num_segments,
        num_rings=env.cfg.num_rings,
        output_dir=output_dir,
        output_name=output_name,
        export_step=export_step,
    )


async def export_results_from_usd(usd_path: str, output_name: str):
    """Export STL from USD via Isaac converter; STEP conversion remains optional."""
    try:
        converter = importlib.import_module("isaaclab.kit.asset_converter")
    except Exception:
        raise RuntimeError("isaaclab.kit.asset_converter is not available in current environment.")

    conv = converter.get_instance()
    settings = converter.AssetConverterSettings()
    stl_path = os.path.abspath(f"{output_name}.stl")
    task = conv.create_converter_task(usd_path, stl_path, None, settings)
    await task.wait_until_finished()
    print(f"STL exported from USD: {stl_path}")