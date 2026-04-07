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


def build_cylinder_cap_faces(num_segments: int, num_rings: int, lower_center: int, upper_center: int) -> np.ndarray:
    faces = []
    lower_ring = 0
    upper_ring = (num_rings - 1) * num_segments
    for s_idx in range(num_segments):
        s_next = (s_idx + 1) % num_segments
        faces.append([lower_center, lower_ring + s_next, lower_ring + s_idx])
        faces.append([upper_center, upper_ring + s_idx, upper_ring + s_next])
    return np.asarray(faces, dtype=np.int64)


def _candidate_freecad_commands(freecad_cmd: str = "") -> list[str]:
    candidates: list[str] = []
    executable_names = ("FreeCADCmd.exe", "FreeCADcmd.exe", "FreeCADCmd", "FreeCADcmd")

    def push(candidate: str | None) -> None:
        if not candidate:
            return
        cleaned = str(candidate).strip().strip('"')
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    push(freecad_cmd)
    push(os.environ.get("FREECAD_CMD"))
    push(os.environ.get("FREECADCMD"))

    for env_name in ("FREECAD_HOME", "FREECAD_BIN"):
        base = os.environ.get(env_name)
        if not base:
            continue
        base_path = Path(base)
        for exe_name in executable_names:
            push(str(base_path / exe_name))
            push(str(base_path / "bin" / exe_name))

    for exe_name in executable_names:
        push(shutil.which(exe_name))

    common_roots = [
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
        Path.home() / "AppData" / "Local" / "Programs",
    ]
    for root in common_roots:
        if not root.exists():
            continue
        for exe_name in ("FreeCADCmd.exe", "FreeCADcmd.exe"):
            for match in root.glob(f"FreeCAD*/bin/{exe_name}"):
                push(str(match))

    resolved = []
    for candidate in candidates:
        if os.path.isfile(candidate):
            resolved.append(candidate)
        elif shutil.which(candidate) is not None:
            resolved.append(candidate)
    return resolved


def _resolve_freecad_command(freecad_cmd: str = "") -> Optional[str]:
    matches = _candidate_freecad_commands(freecad_cmd=freecad_cmd)
    return matches[0] if matches else None


def _run_freecad_stl_to_step(stl_path: str, step_path: str, freecad_cmd: str = "FreeCADCmd") -> bool:
    """
    Convert STL mesh to STEP using FreeCAD command-line.
    Returns True on success, False otherwise.
    """
    resolved_cmd = _resolve_freecad_command(freecad_cmd)
    if resolved_cmd is None:
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
            [resolved_cmd, tmp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(f"[export] FreeCAD STEP conversion failed: {proc.stderr.strip()}", flush=True)
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

    vertices = np.asarray(points, dtype=np.float64)
    lower_center = np.mean(vertices[:num_segments], axis=0, keepdims=True)
    upper_center = np.mean(vertices[-num_segments:], axis=0, keepdims=True)
    vertices = np.vstack([vertices, lower_center, upper_center])
    lower_center_idx = vertices.shape[0] - 2
    upper_center_idx = vertices.shape[0] - 1

    side_faces = build_cylinder_surface_faces(num_segments=num_segments, num_rings=num_rings)
    cap_faces = build_cylinder_cap_faces(
        num_segments=num_segments,
        num_rings=num_rings,
        lower_center=lower_center_idx,
        upper_center=upper_center_idx,
    )
    faces = np.vstack([side_faces, cap_faces])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=False)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    if not mesh.is_watertight:
        print("[export] Mesh is not watertight after repair; STEP conversion may fail.", flush=True)
    mesh.export(stl_path)

    step_written = False
    if export_step:
        step_written = _run_freecad_stl_to_step(stl_path=stl_path, step_path=stp_path, freecad_cmd=freecad_cmd)

    return {
        "stl": stl_path,
        "stp": stp_path if step_written else None,
        "watertight": bool(mesh.is_watertight),
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
        freecad_cmd=getattr(env.cfg, "freecad_cmd", ""),
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
