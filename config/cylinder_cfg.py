def configclass(cls):
    return cls

@configclass
class CylinderPhysicsCfg:
    # USD / runtime
    prim_path = "/World/Cylinder"
    device = "cuda:0"
    use_usd_backend = False
    viewer_scale = 400.0
    viewer_lift_z = 0.5
    dt = 0.002
    max_steps = 30000

    # Baseline geometry (competition statement: dia=5mm, h=15mm)
    radius = 2.5e-3
    height = 15e-3

    # Surface discretization
    num_segments = 48
    num_rings = 24

    # Material constants (pure tungsten)
    density = 19350.0                 # kg/m^3
    cp_ref = 140.0                    # J/(kg*K), rough high-temp fit anchor
    cp_temp_coeff = 0.02              # J/(kg*K^2), simple linearized fit
    k_ref = 120.0                     # W/(m*K) near room/high-T simplified
    k_temp_coeff = -0.025             # W/(m*K^2), conductivity drops with T
    rho_elec_ref = 5.6e-8             # Ohm*m (300K)
    rho_elec_temp_coeff = 4.5e-3      # 1/K

    # Electrical loading
    applied_voltage = 30.0            # V
    min_resistance = 1e-6             # Ohm safety lower bound
    max_current = 3.0e3               # A clamp to avoid blow-up

    # Thermal / radiative environment
    ambient_temp = 300.0              # K
    max_temp = 3273.15                # 3000C
    stefan_boltzmann = 5.670374419e-8 # W/(m^2*K^4)
    emissivity_low = 0.30
    emissivity_high = 0.55
    emissivity_transition_temp = 2200.0

    # Surface evaporation model: Ye = A * exp(B / T) [g/(cm^2*s)]
    evap_A = 3.9e8
    evap_B = -1.023e5
    latent_heat_evap = 4.5e6          # J/kg, simplified effective latent heat

    # Mechanics / shape control
    mass_lumped = 0.02
    damping = 0.75
    k_spring = 8.0
    k_bend = 2.0
    k_input = 4.5
    max_depth = 1.6e-4
    min_sigma = 1.2e-4
    max_sigma = 1.0e-3
    min_radius = 8.0e-4
    dent_decay = 0.01
    max_total_dent = 5.0e-4
    dent_active_threshold = 5.0e-6

    # Failure / constraints
    terminate_on_constraints = True
    feature_fail_ratio = 0.20         # |Li(t)-Li(0)|/Li(0) >= 20%
    max_mass_loss_rate = 1.5e-6       # kg/s (soft constraint)
    keep_electrode_rings_fixed = True

    # Reward shaping
    reward_scale_radiation = 1.0
    penalty_mass_loss = 2.0e5
    penalty_temp_violation = 2.0e-2
    penalty_feature_violation = 120.0
    penalty_volume_change = 80.0

    # Action space
    num_actions = 3

    # Greedy search policy settings
    search_top_k = 10
    search_depth_grid = (0.20, 0.45, 0.70, 0.90)
    search_sigma_grid = (1.2e-4, 4.5e-4, 1.0e-3)
    log_interval = 10