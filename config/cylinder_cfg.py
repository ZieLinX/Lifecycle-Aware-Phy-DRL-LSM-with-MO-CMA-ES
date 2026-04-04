import importlib
from importlib import util as importlib_util

def _safe_find_spec(module_name: str):
    try:
        return importlib_util.find_spec(module_name)
    except ModuleNotFoundError:
        return None


_isaaclab_utils = importlib.import_module("isaaclab.utils") if _safe_find_spec("isaaclab.utils") else None
configclass = _isaaclab_utils.configclass if _isaaclab_utils is not None else (lambda cls: cls)

@configclass
class CylinderPhysicsCfg:
    prim_path = "/World/Cylinder"
    device = "cuda:0"
    radius = 0.5
    height = 1.0

    # Discretization
    num_segments = 32
    num_rings = 16

    # Physics terms
    dt = 0.01
    mass = 0.02
    damping = 0.6
    k_spring = 12.0
    k_bend = 3.5
    k_input = 8.0

    # Action scaling and safety constraints
    max_depth = 0.08
    min_sigma = 0.02
    max_sigma = 0.20
    min_radius = 0.15
    dent_decay = 0.01
    max_total_dent = 0.12
    dent_active_threshold = 0.003

    # Thermodynamics (free-energy objective)
    temp_k = 600.0
    kb = 1.38e-23
    hbar = 1.054e-34

    # Rollout settings
    max_steps = 200
    num_actions = 3

    # Greedy search budget (larger = better but slower)
    search_top_k = 8
    search_depth_grid = (0.20, 0.45, 0.70, 0.90)
    search_sigma_grid = (0.02, 0.11, 0.20)
    log_interval = 10