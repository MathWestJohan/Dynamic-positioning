from dataclasses import dataclass

# R/V Gunnerus — NTNU research vessel
# LOA 31.25 m, beam 9.9 m, draft 3.6 m, displacement ~500 t
# 2x Rolls-Royce PM azimuth thrusters (retrofit 2015), Vmax ≈ 13 kn
# 1x Brunvoll bow tunnel thruster
# Sources: ntnu.edu/gunnerus, VesselFinder IMO 9371361, SINTEF propulsion report

@dataclass
class VesselParams:
    mass: float = 450_000.0          # ~500 t displacement, reduced slightly as AGX hydro adds mass
    Iz:   float = 30.0e6            # yaw inertia [kg·m²], approx m·(L/4)²
    Xu:   float = 20_000.0          # surge linear damping [N·s/m] — low; AGX hydro adds more
    Yv:   float = 50_000.0          # sway linear damping
    Nr:   float = 15_000_000.0      # yaw damping [N·m·s/rad] — measured from AGX hydro

    half_length: float = 15.6       # LOA/2 ≈ 31.25/2
    half_width: float = 4.95        # beam/2 ≈ 9.9/2
    half_height: float = 0.9
    cm_shift_x: float = -0.2

    thruster_z_offset: float = -3.0  # thrusters below waterline, draft ~3.6 m
    stern_x_offset: float = None

    thr_port_x: float = -14.0       # ~1.5 m forward of stern (stern at -15.6)
    thr_port_y: float = +3.0        # wider spacing on 9.9 m beam
    thr_star_x: float = -14.0
    thr_star_y: float = -3.0

    Tmax_thruster: float = 150_000.0  # ~15 t bollard pull per thruster, realistic for PM azimuth
    alloc_bias_Fy: float = 0.0

    thr_bow_x: float = 12.0          # bow tunnel thruster, ~3m from bow
    thr_bow_y: float = 0.0           # centerline
    Tmax_bow: float = 60_000.0       # Brunvoll tunnel thruster, ~6t lateral only

@dataclass
class SceneParams:
    wave_height: float = 0.0

    # Reference filter
    ref_pos_vmax: float = 3.0       # ~6 kn maneuvering — leaves 80% thrust for sway/yaw
    ref_pos_wn: float = 0.10        # slightly above heading wn (0.08) so ship arrives before needing to turn
    ref_head_rmax: float = 0.03     # ~1.7 deg/s — max sustainable against AGX hydro damping
    ref_head_wn: float = 0.08       # fits within yaw torque budget (Iz·α + Nr·r < 700 kNm)

    # Controller — 3 bandwidth params replace 9 PID gains
    wn_pos: float = 0.25            # position control bandwidth [rad/s]
    wn_psi: float = 0.15            # heading control bandwidth [rad/s]
    sway_ratio: float = 0.3         # sway gains = surge gains * this

    tau_max: float = 300_000.0       # matches 2x150 kN thruster budget

    obs_L_eta: float = 0.15
    obs_L_nu_xy: float = 0.2
    obs_L_nu_psi: float = 0.2
    obs_filter_alpha: float = 0.03
    goal_tol: float = 8.0


def compute_pid_gains(v: VesselParams, s: SceneParams):
    """Derive PID gains from physics + bandwidth parameters."""
    # Surge: Kp = M*wn², Kd = 2*M*wn - D, Ki = Kp*wn/10
    kp_x = v.mass * s.wn_pos ** 2
    kd_x = max(0.0, 2.0 * v.mass * s.wn_pos - v.Xu)
    ki_x = kp_x * s.wn_pos / 10.0

    # Sway: same formulas scaled down
    kp_y = kp_x * s.sway_ratio
    kd_y = kd_x * s.sway_ratio
    ki_y = ki_x * s.sway_ratio

    # Yaw: same formulas with Iz and Nr
    kp_psi = v.Iz * s.wn_psi ** 2
    kd_psi = max(0.0, 2.0 * v.Iz * s.wn_psi - v.Nr)
    ki_psi = kp_psi * s.wn_psi / 10.0

    return dict(
        kp_x=kp_x, kd_x=kd_x, ki_x=ki_x,
        kp_y=kp_y, kd_y=kd_y, ki_y=ki_y,
        kp_psi=kp_psi, kd_psi=kd_psi, ki_psi=ki_psi,
    )


@dataclass
class DPRoute:
    waypoints: tuple = ((-200.0, 0.0), (-80.0, 40.0), (0.0, -30.0), (120.0, 50.0), (200.0, -20.0))
    psi_d:    float = None

@dataclass
class Disturbance:
    time: float = 9999.0              # effectively disabled
    duration: float = 0.0
    force_x: float = 0.0
    force_y: float = 0.0

@dataclass
class GNSSNoise:
    sigma_pos: float = 0.30
    sigma_psi: float = 0.01
    disable_noise: bool = True

vessel = VesselParams()
scene  = SceneParams()
route  = DPRoute()
gnss   = GNSSNoise()
disturbance = Disturbance()
gains  = compute_pid_gains(vessel, scene)
