import math
import random
import csv, os, atexit
from datetime import datetime
import numpy as np
from typing import Tuple
import agx
from agxPythonModules.utils.environment import simulation, application
from agxPythonModules.utils.callbacks import StepEventCallback as Sec
from agxPythonModules.utils.callbacks import KeyboardCallback as Input

from agx_wrap.world import create_ocean, create_waypoint_marker, create_force_arrow, create_trail_dot
from modeling.vessel import Ship
from control.reference import ReferenceFilter, PosRefParams, HeadRefParams
from control.observer import SimpleObserver, ObsGains
from control.controller import PIDFFController, PIDGains, ThrusterGeometry
from control.allocation import TwoThrusterAllocator, Geometry2Thrusters
from runtime.config import vessel as VCFG, scene as SCFG, route as RCFG, gnss as NCFG, disturbance as DCFG, gains as GCFG

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(_LOG_DIR, f"dp_log_{_timestamp}.csv")
_log_file = open(log_path, "w", newline="")
_log_writer = csv.writer(_log_file)
_log_writer.writerow([
    "t", "x", "y", "psi", "xr", "yr", "psir",
    "ex", "ey", "epsi", "ex_body", "ey_body", "pos_err",
    "tau_x", "tau_y", "tau_psi",
    "Fx1", "Fy1", "Fx2", "Fy2", "Fy_bow",
    "alloc_scale", "sigma_x", "sigma_y", "sigma_psi",
    "disturbance",
])

def _angle_of_line(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.atan2(y1 - y0, x1 - x0)

def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def _world_to_body(psi: float, vx: float, vy: float) -> Tuple[float, float]:
    c, s = math.cos(psi), math.sin(psi)
    return c * vx + s * vy, -s * vx + c * vy

def build_scene_and_start():
    create_ocean(height=SCFG.wave_height)

    waypoints = list(RCFG.waypoints)
    spawn = waypoints[0]
    goals = list(waypoints[1:])

    ship = Ship(
        mass_kg=VCFG.mass,
        half_length=VCFG.half_length,
        half_width=VCFG.half_width,
        half_height=VCFG.half_height,
        cm_shift_x=VCFG.cm_shift_x,
        thruster_z_offset=VCFG.thruster_z_offset,
        stern_x_offset=VCFG.stern_x_offset,
        thr_port_x=VCFG.thr_port_x,
        thr_port_y=VCFG.thr_port_y,
        thr_star_x=VCFG.thr_star_x,
        thr_star_y=VCFG.thr_star_y,
        thr_bow_x=VCFG.thr_bow_x,
        thr_bow_y=VCFG.thr_bow_y,
    )
    ship.setPosition(agx.Vec3(spawn[0], spawn[1], 0.0))
    ship.ship_body.setMotionControl(agx.RigidBody.KINEMATICS)
    simulation().add(ship)

    # Waypoint markers
    wp_markers = []
    wp_colors = [(0.0, 0.9, 0.0, 1.0), (1.0, 0.6, 0.0, 1.0), (0.9, 0.0, 0.0, 1.0)]
    for i, (wx, wy) in enumerate(goals):
        rgba = wp_colors[i % len(wp_colors)]
        marker = create_waypoint_marker(wx, wy, z=3.0, rgba=rgba)
        wp_markers.append(marker)

    # --- Keyboard waypoint control ---
    selected_wp = [0]
    WP_STEP = 10.0

    def _move_wp(dx, dy):
        idx = selected_wp[0]
        if idx >= len(goals):
            return
        ox, oy = goals[idx]
        goals[idx] = (ox + dx, oy + dy)
        pole, top = wp_markers[idx]
        pole.setPosition(agx.Vec3(ox + dx, oy + dy, 3.0))
        top.setPosition(agx.Vec3(ox + dx, oy + dy, 8.0))

    def _select_wp(data):
        if not data.isKeyDown:
            return
        selected_wp[0] = (selected_wp[0] + 1) % len(goals)
        print(f"Selected waypoint {selected_wp[0]+1}/{len(goals)}: {goals[selected_wp[0]]}")

    def _wp_north(data):
        if data.down:
            _move_wp(WP_STEP, 0)
    def _wp_south(data):
        if data.down:
            _move_wp(-WP_STEP, 0)
    def _wp_east(data):
        if data.down:
            _move_wp(0, -WP_STEP)
    def _wp_west(data):
        if data.down:
            _move_wp(0, WP_STEP)

    Input.bind(name="select_wp", key="n", callback=_select_wp)
    Input.bind(name="wp_north",  key=Input.KEY_Up,    callback=_wp_north)
    Input.bind(name="wp_south",  key=Input.KEY_Down,  callback=_wp_south)
    Input.bind(name="wp_east",   key=Input.KEY_Right, callback=_wp_east)
    Input.bind(name="wp_west",   key=Input.KEY_Left,  callback=_wp_west)

    ref = ReferenceFilter(
        pos_params=PosRefParams(
            omega=SCFG.ref_pos_wn, vmax=SCFG.ref_pos_vmax
        ),
        head_params=HeadRefParams(
            omega=SCFG.ref_head_wn, rmax=SCFG.ref_head_rmax
        )
    )
    ref.reset(psi_now=ship.get_xy_psi()[2])

    obs = SimpleObserver(ObsGains(
        L_eta=getattr(SCFG, "obs_L_eta", 1.0),
        L_nu_xy=getattr(SCFG, "obs_L_nu_xy", 1.0),
        L_nu_psi=getattr(SCFG, "obs_L_nu_psi", 1.0)
    ))
    x0, y0, psi0 = ship.get_xy_psi()
    obs.reset(x0, y0, psi0)

    lx1, ly1 = float(ship.thruster_port_local.x()), float(ship.thruster_port_local.y())
    lx2, ly2 = float(ship.thruster_star_local.x()), float(ship.thruster_star_local.y())
    lx_bow = float(ship.thruster_bow_local.x())
    ly_bow = float(ship.thruster_bow_local.y())
    geom = Geometry2Thrusters(lx1=lx1, ly1=ly1, lx2=lx2, ly2=ly2,
                              lx_bow=lx_bow, ly_bow=ly_bow, biasFy=VCFG.alloc_bias_Fy)
    alloc = TwoThrusterAllocator(geom, Tmax=VCFG.Tmax_thruster, Tmax_bow=VCFG.Tmax_bow)

    thr_geom = ThrusterGeometry(lx1=lx1, ly1=ly1, lx2=lx2, ly2=ly2)

    M = [VCFG.mass, VCFG.mass, VCFG.Iz]
    D = [VCFG.Xu,   VCFG.Yv,   VCFG.Nr]
    ctl = PIDFFController(
        M_diag=M, D_diag=D,
        gains=PIDGains(
            Kp_x=GCFG['kp_x'], Kd_x=GCFG['kd_x'], Ki_x=GCFG['ki_x'],
            Kp_y=GCFG['kp_y'], Kd_y=GCFG['kd_y'], Ki_y=GCFG['ki_y'],
            Kp_psi=GCFG['kp_psi'], Kd_psi=GCFG['kd_psi'], Ki_psi=GCFG['ki_psi'],
            tau_max=SCFG.tau_max
        ),
        thruster_geom=thr_geom
    )

    # Multi-waypoint state
    wp_idx = [0]
    xA, yA = spawn
    xB, yB = goals[0]
    phi = [_angle_of_line(xA, yA, xB, yB)]
    L_path = [math.hypot(xB - xA, yB - yA)]
    seg_origin = [spawn]

    psi_d = RCFG.psi_d if RCFG.psi_d else phi[0]

    # Thruster force arrows
    arrow_port_geom, arrow_port_node = create_force_arrow(rgba=(1.0, 0.3, 0.0, 1.0))
    arrow_star_geom, arrow_star_node = create_force_arrow(rgba=(1.0, 0.3, 0.0, 1.0))
    arrow_bow_geom, arrow_bow_node = create_force_arrow(rgba=(0.2, 0.8, 1.0, 1.0))

    # Trail state
    trail_dots = []
    trail_max = 300
    trail_step_counter = [0]

    sd = application().getSceneDecorator()
    sd.setText(1, "DP: initializing...")
    sd.setText(2, "Thrusters [Fx1,Fy1,Fx2,Fy2] (kN)")
    sd.setText(3, "Commanded tau [X,Y,N]")
    sd.setText(4, "")
    sd.setText(6, "[N] cycle WP  [arrows] move selected WP")

    mode = {"state": "TRANSIT"}
    goal_tol = getattr(SCFG, "goal_tol", 5.0)
    last_tau = (0.0, 0.0, 0.0)
    t_sim = 0.0
    settled = [False]
    SETTLE_TIME = 0.5

    def _advance_waypoint():
        wp_idx[0] += 1
        if wp_idx[0] >= len(goals):
            mode["state"] = "HOLD"
            return

        prev_goal = goals[wp_idx[0] - 1]
        next_goal = goals[wp_idx[0]]
        seg_origin[0] = prev_goal
        xA_new, yA_new = prev_goal
        xB_new, yB_new = next_goal
        phi[0] = _angle_of_line(xA_new, yA_new, xB_new, yB_new)
        L_path[0] = math.hypot(xB_new - xA_new, yB_new - yA_new)

        ref.reset(psi_now=ship.get_xy_psi()[2])

    _arrow_state = {}

    def _update_force_arrow(arrow_geom, fx, fy, thr_body):
        q = ship.ship_body.getRotation()
        pos = ship.ship_body.getPosition()
        world_thr = pos + q * thr_body

        f_mag = math.sqrt(fx*fx + fy*fy)

        if f_mag < 2000.0:
            arrow_geom.setPosition(agx.Vec3(0, 0, -100))
            _arrow_state.pop(id(arrow_geom), None)
            return

        f_world = ship.ship_force_to_world(fx, fy)
        f_dir_x = float(f_world.x())
        f_dir_y = float(f_world.y())
        angle = math.atan2(f_dir_y, f_dir_x)
        scale = min(f_mag / VCFG.Tmax_thruster * 15.0, 18.0)

        target_x = float(world_thr.x()) + f_dir_x / f_mag * scale * 0.5
        target_y = float(world_thr.y()) + f_dir_y / f_mag * scale * 0.5
        target_z = float(world_thr.z()) + 4.0

        key = id(arrow_geom)
        alpha = 0.3
        if key in _arrow_state:
            prev_x, prev_y, prev_z, prev_angle = _arrow_state[key]
            target_x = prev_x + alpha * (target_x - prev_x)
            target_y = prev_y + alpha * (target_y - prev_y)
            target_z = prev_z + alpha * (target_z - prev_z)
            da = math.atan2(math.sin(angle - prev_angle), math.cos(angle - prev_angle))
            angle = prev_angle + alpha * da

        _arrow_state[key] = (target_x, target_y, target_z, angle)

        arrow_geom.setPosition(agx.Vec3(target_x, target_y, target_z))
        arrow_geom.setRotation(agx.EulerAngles(0, math.pi/2, angle - math.pi/2))

    def dp_step(_time: float):
        nonlocal last_tau, t_sim
        dt = simulation().getTimeStep()
        t_sim += dt

        if not settled[0]:
            if t_sim >= SETTLE_TIME:
                ship.ship_body.setMotionControl(agx.RigidBody.DYNAMICS)
                ship.ship_body.setVelocity(agx.Vec3(0, 0, 0))
                ship.ship_body.setAngularVelocity(agx.Vec3(0, 0, 0))
                settled[0] = True
            else:
                return

        x, y, psi = ship.get_xy_psi()
        if getattr(NCFG, "disable_noise", False):
            x_m, y_m, psi_m = x, y, psi
        else:
            x_m   = x   + random.gauss(0.0, getattr(NCFG, "sigma_pos", 0.0))
            y_m   = y   + random.gauss(0.0, getattr(NCFG, "sigma_pos", 0.0))
            psi_m = _wrap_pi(psi + random.gauss(0.0, getattr(NCFG, "sigma_psi", 0.0)))

        cur_goal = goals[min(wp_idx[0], len(goals) - 1)]
        xB_cur, yB_cur = cur_goal

        dN, dE = (xB_cur - x), (yB_cur - y)
        rem_alng = max(0.0, math.cos(phi[0]) * dN + math.sin(phi[0]) * dE)
        progress = L_path[0] - rem_alng

        if mode["state"] == "TRANSIT" and rem_alng <= goal_tol:
            _advance_waypoint()

        # Re-read after possible waypoint advance
        origin = seg_origin[0]
        xA_cur, yA_cur = origin

        if mode["state"] == "TRANSIT":
            # LOS guidance: turn the bow toward the path instead of pushing sideways
            dx_path = x - xA_cur
            dy_path = y - yA_cur
            e_crosstrack = -dx_path * math.sin(phi[0]) + dy_path * math.cos(phi[0])
            DELTA_LOS = 80.0
            psi_los = phi[0] + math.atan2(-e_crosstrack, DELTA_LOS)

            # Turn anticipation: blend heading toward next leg before reaching WP
            ANTICIPATION_DIST = 50.0
            if rem_alng < ANTICIPATION_DIST and wp_idx[0] < len(goals) - 1:
                next_goal = goals[wp_idx[0] + 1]
                phi_next = _angle_of_line(xB_cur, yB_cur, next_goal[0], next_goal[1])
                blend = 1.0 - rem_alng / ANTICIPATION_DIST
                psi_los = _wrap_pi(psi_los + blend * _wrap_pi(phi_next - psi_los))

            psi_target = RCFG.psi_d if RCFG.psi_d else psi_los

            # Speed-heading coordination: slow position reference when
            # ship heading hasn't caught up to the path direction.
            hdg_err = _wrap_pi(psi_target - psi)
            speed_factor = max(0.0, math.cos(hdg_err))

            pr, vr, ar, rr, alphar, psir = ref.step(dt, pd=L_path[0], psi_d=psi_target,
                                                     speed_factor=speed_factor)
            xr = xA_cur + pr * math.cos(phi[0])
            yr = yA_cur + pr * math.sin(phi[0])
        else:
            final_goal = goals[-1]
            psi_hold = RCFG.psi_d if RCFG.psi_d else phi[0]
            pr, vr, ar, rr, alphar, psir = ref.step(dt, pd=L_path[0], psi_d=psi_hold)
            xr, yr = final_goal

        (xh, yh, psih), (uh, vh, rh) = obs.step(
            dt,
            meas_x=x_m, meas_y=y_m, meas_psi=psi_m,
            tau_x=last_tau[0], tau_y=last_tau[1], tau_n=last_tau[2],
            M=M, D=D
        )

        ur_world, vr_world = vr * math.cos(phi[0]), vr * math.sin(phi[0])
        ur_body, vr_body   = _world_to_body(psih, ur_world, vr_world)

        ar_world_x, ar_world_y = ar * math.cos(phi[0]), ar * math.sin(phi[0])
        udotr_body, vdotr_body = _world_to_body(psih, ar_world_x, ar_world_y)

        etar   = (xr, yr, psir)
        nur    = (ur_body, vr_body, rr)
        nudotr = (udotr_body, vdotr_body, alphar)

        taux, tauy, taun = ctl.step(
            dt,
            eta_r=etar, nu_r=nur, nudot_r=nudotr,
            eta_hat=(xh, yh, psih), nu_hat=(uh, vh, rh)
        )

        Fx1, Fy1, Fx2, Fy2, Fy_bow, alloc_scale = alloc.allocate(taux, tauy, taun)
        ship.apply_thruster_forces(Fx1, Fy1, Fx2, Fy2, Fy_bow)
        last_tau = (taux, tauy, taun)

        # Disturbance
        dist_active = 0
        if DCFG.time <= t_sim <= DCFG.time + DCFG.duration:
            ship.ship_body.addForce(agx.Vec3(DCFG.force_x, DCFG.force_y, 0))
            dist_active = 1

        if getattr(VCFG, "enable_drag", False):
            du = uh; dv = vh; dr = rh
            Fx_drag = -(VCFG.drag_lin_u * du + VCFG.drag_quad_u * abs(du) * du)
            Fy_drag = -(VCFG.drag_lin_v * dv + VCFG.drag_quad_v * abs(dv) * dv)
            Nz_drag = -(VCFG.drag_lin_r * dr + VCFG.drag_quad_r * abs(dr) * dr)
            ship.apply_body_drag(Fx_drag, Fy_drag, Nz_drag)

        # Update force arrows
        _update_force_arrow(arrow_port_geom, Fx1, Fy1, ship.thruster_port_body)
        _update_force_arrow(arrow_star_geom, Fx2, Fy2, ship.thruster_star_body)
        _update_force_arrow(arrow_bow_geom, 0.0, Fy_bow, ship.thruster_bow_body)

        # Trail
        trail_step_counter[0] += 1
        if trail_step_counter[0] % 20 == 0:
            if len(trail_dots) >= trail_max:
                old = trail_dots.pop(0)
                old.setPosition(agx.Vec3(0, 0, -100))
            trail_dots.append(create_trail_dot(x, y, z=2.0))

        # HUD
        wp_str = f"WP {wp_idx[0]+1}/{len(goals)}" if mode["state"] == "TRANSIT" else "HOLD"
        dist_str = " | DISTURBANCE" if dist_active else ""
        ex = xr - xh
        ey = yr - yh
        epsi = _wrap_pi(psir - psih)
        c_h, s_h = math.cos(psih), math.sin(psih)
        ex_body = c_h * ex + s_h * ey
        ey_body = -s_h * ex + c_h * ey
        pos_err = math.hypot(ex, ey)

        speed = math.hypot(uh, vh)
        speed_kn = speed * 1.944
        sd.setText(1, f"{wp_str} | progress {progress:6.1f}/{L_path[0]:.1f} m | rem {rem_alng:5.1f} m | {speed:.1f} m/s ({speed_kn:.1f} kn){dist_str}")
        sd.setText(2, f"stern[{Fx1/1000:5.0f},{Fy1/1000:5.0f},{Fx2/1000:5.0f},{Fy2/1000:5.0f}] bow[{Fy_bow/1000:4.0f}] kN  alloc {alloc_scale:.0%}")
        sd.setText(3, f"tau [{taux/1000:6.1f}, {tauy/1000:6.1f}, {taun/1000:6.1f}]")
        sd.setText(4, f"psi {math.degrees(psih):+6.1f} ref {math.degrees(psir):+6.1f} err {math.degrees(epsi):+6.1f} deg | pos_err {pos_err:.1f} m")
        sd.setText(5, f"body err  surge {ex_body:+7.1f}  sway {ey_body:+7.1f} m | integ [{ctl.sigma_x/1000:.1f}, {ctl.sigma_y/1000:.1f}, {ctl.sigma_psi/1000:.1f}] kN")

        # Log (extended columns)
        _log_writer.writerow([
            t_sim, xh, yh, psih, xr, yr, psir,
            ex, ey, epsi, ex_body, ey_body, pos_err,
            taux, tauy, taun,
            Fx1, Fy1, Fx2, Fy2, Fy_bow,
            alloc_scale, ctl.sigma_x, ctl.sigma_y, ctl.sigma_psi,
            dist_active,
        ])

    Sec.preCallback(lambda t: dp_step(t))

    def close_log():
        try:
            _log_file.close()
            print(f"Log saved to {log_path}")
        except:
            pass
    atexit.register(close_log)

    cam = application().getCameraData()
    cam.eye    = agx.Vec3(spawn[0] - 30.0, spawn[1] - 80.0, 45.0)
    cam.center = agx.Vec3(spawn[0], spawn[1], 5.0)
    cam.up     = agx.Vec3(0.0, 0.0, 1.0)
    cam.nearClippingPlane = 0.1
    cam.farClippingPlane  = 5000.0
    application().applyCameraData(cam)

    simulation().setTimeStep(simulation().getTimeStep() * 4.0)
