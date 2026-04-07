def configclass(cls):
    return cls


@configclass
class CylinderPhysicsCfg:
    # Runtime / viewer
    prim_path = "/World/Cylinder"
    device = "cuda:0"
    use_usd_backend = False
    viewer_scale = 400.0
    viewer_lift_z = 0.5
    viewer_subdivision_scheme = "catmullClark"

    # Design loop: one step = one offline geometry edit + one full evaluation.
    dt = 1.0
    max_steps = 40
    terminate_on_constraints = True
    visualize_disable_constraints = False

    # Baseline geometry (competition statement: dia=5 mm, h=15 mm)
    radius = 2.5e-3
    height = 15e-3

    # Surface discretization
    num_segments = 96
    num_rings = 48
    keep_electrode_rings_fixed = True

    # Material constants (pure tungsten)
    density = 19350.0
    cp_ref = 140.0
    cp_temp_coeff = 0.02
    k_ref = 120.0
    k_temp_coeff = -0.025
    rho_elec_ref = 5.6e-8
    rho_elec_temp_coeff = 3.5e-3

    # Voltage is an upper bound for rated-condition search, not a fixed always-on load.
    min_voltage = 5.0
    max_voltage = 100.0
    voltage_grid_points = 11
    voltage_refine_levels = 2
    voltage_refine_points = 7
    voltage_focus_ratio = 0.18
    min_resistance = 1e-6
    max_current = 5.0e3
    external_series_resistance = 0.08

    # Thermal / radiative environment
    ambient_temp = 300.0
    max_temp = 3273.15
    stefan_boltzmann = 5.670374419e-8
    # Ignore emissivity-temperature coupling as required by the statement.
    # Use fixed spectral emissivity: 0-3 um -> 0.35, other bands -> 0.15.
    band_emissivity = 0.35
    out_of_band_emissivity = 0.15
    radiative_cooling_scale = 1.0
    convective_cooling_coeff = 0.0

    # Surface evaporation model: Ye = A * exp(B / T) [g/(cm^2*s)]
    evap_A = 3.9e8
    evap_B = -1.023e5
    latent_heat_evap = 4.5e6

    # Offline rated-condition solver
    thermal_relaxation = 0.45
    thermal_pseudo_dt = 0.02
    thermal_max_iters = 160
    thermal_tol_k = 0.05
    in_band_upper_um = 3.0
    band_fraction_min_temp = 300.0
    band_fraction_max_temp = 4200.0
    band_fraction_lut_size = 384
    min_view_factor = 0.45
    shadow_slope_coeff = 0.18
    shadow_roughness_coeff = 0.12
    lifecycle_reference_s = 3600.0
    ablation_observation_horizon_s = 600.0

    # Mechanics / shape control
    mass_lumped = 0.02
    damping = 0.0
    k_spring = 8.0
    k_bend = 2.0
    k_input = 4.5
    max_depth = 1.6e-4
    min_sigma = 1.2e-4
    max_sigma = 1.0e-3
    min_radius = 8.0e-4
    dent_decay = 0.0
    max_total_dent = 5.0e-4
    max_total_bulge = 5.0e-4
    dent_active_threshold = 5.0e-6
    use_strict_shape_projection = False
    shape_projection_alpha = 0.0
    dent_polygon_sides = 256
    compensation_exclusion_sigma = 1.35
    compensation_cool_bias = 0.35

    # Constraints / failure
    feature_fail_ratio = 0.20
    minimum_lifetime_ratio = 0.30
    volume_tolerance_ratio = 0.02
    max_mass_loss_rate = 1.5e-6

    # Objective weights
    rated_weight_initial_power = 1.0
    rated_weight_average_power = 0.65
    rated_weight_uniformity = 0.15
    rated_penalty_mass_loss = 2.0e5
    rated_penalty_temp_violation = 2.5e-1
    rated_penalty_feature_violation = 120.0
    rated_penalty_volume_change = 80.0

    reward_weight_initial_power = 1.15
    reward_weight_average_power = 1.00
    reward_weight_lifetime = 0.45
    reward_weight_uniformity = 0.20
    reward_weight_efficiency = 0.10
    reward_penalty_temp_violation = 2.5e-1
    reward_penalty_mass_loss = 2.0e5
    reward_penalty_feature_violation = 120.0
    reward_penalty_volume_change = 80.0
    reward_penalty_free_energy = 0.02

    # Legacy aliases kept for minimal downstream breakage
    applied_voltage = max_voltage
    penalty_mass_loss = reward_penalty_mass_loss
    penalty_temp_violation = reward_penalty_temp_violation
    penalty_feature_violation = reward_penalty_feature_violation
    penalty_volume_change = reward_penalty_volume_change
    reward_scale_radiation = reward_weight_initial_power

    # Action space
    num_actions = 3

    # Model-based planner settings
    search_top_k = 12
    search_depth_grid = (0.0, 0.18, 0.35, 0.55, 0.80)
    search_sigma_grid = (1.2e-4, 3.0e-4, 6.5e-4, 1.0e-3)
    planner_horizon = 2
    planner_beam_width = 3
    planner_seed_top_k = 8
    planner_candidate_top_k = 4
    planner_local_refine_top_k = 2
    planner_refine_neighbor_span = 1
    planner_depth_scale_factors = (0.70, 1.00, 1.30)
    planner_sigma_scale_factors = (0.70, 1.00, 1.35)
    planner_weight_shape = 0.20
    planner_weight_temp = 0.45
    planner_weight_ablation = 0.35

    # Export settings
    freecad_cmd = ""
    log_interval = 1
