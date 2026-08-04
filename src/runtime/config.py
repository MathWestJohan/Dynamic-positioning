from dataclasses import dataclass

@dataclass
class VesselParams:
    mass: float = 350_000.0
    Iz:   float = 20.0e6
    Xu:   float = 50_000.0
    Yv:   float = 80_000.0
    Nr:   float = 2_000_000.0
    
    half_length: float = 10.0
    half_width: float = 3.75
    half_height: float = 0.9
    cm_shift_x: float = -0.2
    
    thruster_z_offset: float = -2.5
    stern_x_offset: float = None
    
    thr_port_x: float = -10.0
    thr_port_y: float = +2.76
    thr_star_x: float = -10.0
    thr_star_y: float = -2.76
    
    Tmax_thruster: float = 500_000.0
    alloc_bias_Fy: float = 0.0

@dataclass
class SceneParams:
    wave_height: float = 1.5
    
    # Reference filter
    ref_pos_wn: float = 0.08
    ref_pos_zeta: float = 1.0
    ref_pos_Ki: float = 0.1
    ref_pos_vmax: float = 0.3
    ref_head_wn: float = 0.10
    ref_head_zeta: float = 1.0
    ref_head_Ki: float = 0.08
    ref_head_rmax: float = 0.05
    
    # Gains for PID+FF controller
    kp_x: float = 20_000.0
    kp_y: float = 20_000.0
    kd_x: float = 100_000.0
    kd_y: float = 100_000.0
    ki_x: float = 200.0
    ki_y: float = 200.0
    
    kp_psi: float = 500_000.0
    kd_psi: float = 2_000_000.0
    ki_psi: float = 1_000.0
    
    tau_max: float = 400_000.0
    
    obs_L_eta: float = 0.15
    obs_L_nu_xy: float = 0.2
    obs_L_nu_psi: float = 0.2
    obs_filter_alpha: float = 0.03
    goal_tol: float = 5.0
    
    # Wave-adaptive gain scaling
    wave_gain_scale_low: float = 1.2
    wave_gain_scale_high: float = 0.6

@dataclass
class DPRoute:
    waypoints: tuple = ((0.0, 0.0), (30.0, 15.0), (50.0, -10.0), (60.0, 20.0))
    psi_d:    float = None

@dataclass
class Disturbance:
    time: float = 40.0
    duration: float = 15.0
    force_x: float = 0.0
    force_y: float = 80_000.0

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