def configclass(cls):
    return cls


@configclass
class CylinderPhysicsCfg:
    # Runtime
    device = "cuda:0"
    grid_mode = "training"

    # Baseline geometry: diameter 5 mm, height 15 mm.
    radius = 2.5e-3
    height = 15e-3
    num_segments = 96
    num_rings = 48
    eval_num_segments = 160
    eval_num_rings = 151

    # Tungsten material constants.
    density = 19350.0
    k_ref = 120.0
    k_temp_coeff = -0.025
    rho_elec_ref = 5.6e-8
    rho_elec_temp_coeff = 3.5e-3

    # Voltage is an upper bound for rated-condition search, not a fixed load.
    min_voltage = 0.01
    max_voltage = 100.0
    voltage_grid_spacing = "log"
    voltage_grid_points = 11
    voltage_refine_levels = 2
    voltage_refine_points = 7
    voltage_refine_spacing = "log"
    voltage_focus_ratio = 0.18
    min_resistance = 1.0e-6
    max_current = 1.0e9
    external_series_resistance = 0.0

    # Thermal and radiative environment.
    ambient_temp = 300.0
    max_temp = 3273.15
    stefan_boltzmann = 5.670374419e-8
    band_emissivity = 0.35
    out_of_band_emissivity = 0.15
    radiative_cooling_scale = 1.0
    in_band_upper_um = 3.0

    # Surface evaporation model: Ye = A * exp(B / T) [g/(cm^2*s)].
    evap_A = 3.9e8
    evap_B = -1.023e5
    latent_heat_evap = 4.5e6

    # Steady thermal solver.
    thermal_relaxation = 0.45
    thermal_max_iters = 160
    thermal_tol_k = 0.05

    # Failure and optimization constraints.
    feature_fail_ratio = 0.20
    minimum_lifetime_ratio = 0.30
    min_radius = 8.0e-4
    max_depth = 1.6e-4

    # Full closed-geometry 3D backend.
    full3d_cap_rings = 8
    full3d_cap_max_displacement_m = 4.0e-4
    full3d_volume_tolerance_ratio = 1.0e-5
    full3d_electrode_tolerance_m = 2.0e-6
    full3d_fixed_voltage_v = None
    full3d_thermal_sink_temperature_k = 300.0
    full3d_thermal_residual_tol_w = 1.0e-3
    full3d_thermal_max_delta_k = 1200.0
    full3d_lifecycle_reference_s = 1.0e27
    full3d_lifetime_recession_floor_m_s = 1.0e-300
    full3d_lifetime_cap_s = 1.0e300
    full3d_sphere_temperature_k = 0.0
    full3d_sphere_emissivity = 1.0
    full3d_use_neural_policy = True

    # Export settings.
    freecad_cmd = ""
    freecad_timeout_s = 90.0


def make_training_cfg() -> CylinderPhysicsCfg:
    cfg = CylinderPhysicsCfg()
    cfg.grid_mode = "training"
    return cfg


def make_eval_cfg() -> CylinderPhysicsCfg:
    cfg = CylinderPhysicsCfg()
    cfg.grid_mode = "evaluation"
    cfg.num_segments = int(cfg.eval_num_segments)
    cfg.num_rings = int(cfg.eval_num_rings)
    return cfg
