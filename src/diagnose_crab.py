"""
Crab-motion diagnostic for DP controller.

Reads the latest log and produces a diagnostic plot that shows WHY
the ship crabs instead of turning. Run with:
    python3 src/diagnose_crab.py [optional_csv_path]
"""
import csv, math, sys, os, glob
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"

# Ensure we import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

def find_latest_log():
    pattern = str(_LOG_DIR / "dp_log_*.csv")
    logs = sorted(glob.glob(pattern))
    return logs[-1] if logs else None

def load_log(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def diagnose(csv_path=None):
    if csv_path is None:
        csv_path = find_latest_log()
        if csv_path is None:
            print("No log file found.")
            return

    rows = load_log(csv_path)
    n = len(rows)
    dt_avg = (rows[-1]['t'] - rows[0]['t']) / max(1, n - 1)

    from runtime.config import vessel as VCFG, scene as SCFG, route as RCFG

    print(f"Log: {Path(csv_path).name}  ({n} steps, {rows[-1]['t']:.0f}s, dt={dt_avg*1000:.1f}ms)")
    print("=" * 80)

    # --- 1. Thruster capacity vs demanded force ---
    max_surge_thrust = 2 * VCFG.Tmax_thruster  # both stern thrusters aligned
    max_accel = max_surge_thrust / VCFG.mass
    time_to_vmax = SCFG.ref_pos_vmax / max_accel

    print("\n[1] THRUSTER CAPACITY vs REFERENCE DEMANDS")
    print(f"  Max total thrust (2x stern):   {max_surge_thrust/1000:.0f} kN")
    print(f"  Ship mass:                     {VCFG.mass/1000:.0f} t")
    print(f"  Max surge acceleration:        {max_accel:.2f} m/s²")
    print(f"  Ref filter vmax:               {SCFG.ref_pos_vmax:.1f} m/s")
    print(f"  Time to reach vmax (ideal):    {time_to_vmax:.1f} s")
    print(f"  Sway tau cap (controller):     {SCFG.tau_max * 0.3 / 1000:.0f} kN")
    print(f"  Bow thruster max:              {VCFG.Tmax_bow/1000:.0f} kN")

    sway_accel = (SCFG.tau_max * 0.3 + VCFG.Tmax_bow) / VCFG.mass
    print(f"  Max sway acceleration:         {sway_accel:.3f} m/s²")
    print(f"  --> Sway authority is {sway_accel/max_accel*100:.0f}% of surge authority")

    # --- 2. Heading bandwidth vs position bandwidth ---
    print("\n[2] BANDWIDTH MISMATCH (heading vs position)")
    print(f"  Position ref ωn:     {SCFG.ref_pos_wn:.2f} rad/s  (rise time ~{1.8/SCFG.ref_pos_wn:.0f}s)")
    print(f"  Heading  ref ωn:     {SCFG.ref_head_wn:.2f} rad/s  (rise time ~{1.8/SCFG.ref_head_wn:.0f}s)")
    print(f"  Heading max rate:    {math.degrees(SCFG.ref_head_rmax):.1f} deg/s")
    ratio = SCFG.ref_pos_wn / SCFG.ref_head_wn
    print(f"  Position/Heading ωn ratio: {ratio:.1f}x  ", end="")
    if ratio > 2.0:
        print("⚠ Position ref is much faster than heading ref!")
    elif ratio > 1.5:
        print("⚠ Position runs ahead of heading at turns")
    else:
        print("OK")

    # --- 3. Path geometry analysis ---
    waypoints = list(RCFG.waypoints)
    print("\n[3] PATH GEOMETRY & TURN ANALYSIS")
    legs = []
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        angle = math.atan2(y1 - y0, x1 - x0)
        length = math.hypot(x1 - x0, y1 - y0)
        legs.append((angle, length))
        print(f"  Leg {i}: angle={math.degrees(angle):+.1f}°  length={length:.0f}m")

    for i in range(len(legs) - 1):
        turn = math.atan2(math.sin(legs[i+1][0] - legs[i][0]),
                         math.cos(legs[i+1][0] - legs[i][0]))
        turn_deg = abs(math.degrees(turn))
        turn_time = turn_deg / math.degrees(SCFG.ref_head_rmax)
        distance_during_turn = SCFG.ref_pos_vmax * turn_time
        print(f"  WP{i+1} turn: {math.degrees(turn):+.0f}°  "
              f"time@max_rate={turn_time:.0f}s  "
              f"ref_advances={distance_during_turn:.0f}m during turn  "
              f"(leg={legs[i+1][1]:.0f}m)")
        if distance_during_turn > legs[i+1][1] * 0.5:
            print(f"       ⚠ Reference travels >{int(distance_during_turn/legs[i+1][1]*100)}% of next leg before turn completes!")

    # --- 4. Time-series analysis ---
    print("\n[4] LOG ANALYSIS — CRAB MOTION TIMELINE")

    # Detect waypoint transitions (large heading ref changes)
    transitions = []
    for i in range(1, n):
        dpsi_r = rows[i]['psir'] - rows[i-1]['psir']
        dpsi_r = math.atan2(math.sin(dpsi_r), math.cos(dpsi_r))
        if i > 1 and abs(dpsi_r) > 0.01:
            if not transitions or rows[i]['t'] - transitions[-1] > 2.0:
                transitions.append(rows[i]['t'])

    # Compute per-timestep metrics
    crab_phases = []
    in_crab = False
    crab_start = 0

    for i, r in enumerate(rows):
        total_f = math.sqrt(r['tau_x']**2 + r['tau_y']**2)
        if total_f > 1000:
            crab_ratio = abs(r['tau_y']) / total_f
        else:
            crab_ratio = 0

        if not in_crab and crab_ratio > 0.4 and abs(r['ey_body']) > 10:
            in_crab = True
            crab_start = r['t']
        elif in_crab and (crab_ratio < 0.1 or abs(r['ey_body']) < 3):
            in_crab = False
            crab_phases.append((crab_start, r['t']))

    if in_crab:
        crab_phases.append((crab_start, rows[-1]['t']))

    if crab_phases:
        print("  Detected crab-motion phases:")
        for start, end in crab_phases:
            print(f"    t={start:.0f}s to {end:.0f}s  (duration={end-start:.0f}s)")
    else:
        print("  No significant crab-motion phases detected")

    # Heading error at sway saturation
    sway_cap = SCFG.tau_max * 0.3
    sat_moments = [(r['t'], math.degrees(r['epsi']), r['ey_body'], r['pos_err'])
                   for r in rows if abs(r['tau_y']) >= sway_cap * 0.99]
    if sat_moments:
        hdg_errs = [abs(m[1]) for m in sat_moments]
        sway_errs = [abs(m[2]) for m in sat_moments]
        print(f"  During sway saturation ({len(sat_moments)} steps, {len(sat_moments)*dt_avg:.0f}s):")
        print(f"    Heading error: mean={sum(hdg_errs)/len(hdg_errs):.1f}°  max={max(hdg_errs):.1f}°")
        print(f"    Sway error:    mean={sum(sway_errs)/len(sway_errs):.1f}m   max={max(sway_errs):.1f}m")

    # --- 5. Allocator utilization ---
    print("\n[5] ALLOCATOR UTILIZATION")
    saturated = sum(1 for r in rows if r['alloc_scale'] < 0.99)
    min_scale = min(r['alloc_scale'] for r in rows)
    print(f"  Saturated: {saturated}/{n} steps ({100*saturated/n:.0f}%)")
    print(f"  Min alloc scale: {min_scale:.3f}")
    if saturated / n > 0.3:
        print(f"  ⚠ Thrusters are saturated {100*saturated/n:.0f}% of the time!")
        print(f"    → Controller demands exceed physical capability")

    # --- 6. System ID sensitivity ---
    print("\n[6] SYSTEM ID SENSITIVITY")
    print(f"  Using: mass={VCFG.mass/1000:.0f}t  Iz={VCFG.Iz/1e6:.0f}e6  Xu={VCFG.Xu/1000:.0f}kN·s/m  Yv={VCFG.Yv/1000:.0f}kN·s/m  Nr={VCFG.Nr/1e6:.1f}e6")

    yaw_torque_budget = VCFG.Tmax_thruster * abs(VCFG.thr_port_y - VCFG.thr_star_y)
    max_yaw_accel = yaw_torque_budget / VCFG.Iz
    print(f"  Max yaw torque from differential stern thrust: {yaw_torque_budget/1000:.0f} kNm")
    print(f"  Max yaw acceleration (with current Iz):       {math.degrees(max_yaw_accel):.2f} deg/s²")

    # What if Iz is 2x larger?
    max_yaw_accel_2x = yaw_torque_budget / (VCFG.Iz * 2)
    print(f"  If Iz were 2x larger:                         {math.degrees(max_yaw_accel_2x):.2f} deg/s²")
    print(f"  → Iz uncertainty directly scales heading response")

    # --- 7. Root cause summary ---
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    issues = []

    if ratio > 1.5:
        issues.append(
            "POSITION REFERENCE OUTRUNS HEADING: The position ref filter (ωn={:.2f}) is {:.1f}x "
            "faster than heading (ωn={:.2f}). When the path turns at a waypoint, the position "
            "reference advances along the new leg while the heading is still turning. This creates "
            "a large cross-track error that drives sway force → crab motion.".format(
                SCFG.ref_pos_wn, ratio, SCFG.ref_head_wn))

    if any(abs(math.degrees(math.atan2(math.sin(legs[i+1][0]-legs[i][0]),
           math.cos(legs[i+1][0]-legs[i][0])))) > 30 for i in range(len(legs)-1)):
        issues.append(
            "LARGE HEADING CHANGES AT WAYPOINTS: Turns of 60-80° require 20-27s at the "
            "3 deg/s max rate. During this time, the position reference can advance 120-160m, "
            "far exceeding the ship's ability to track laterally.")

    if saturated / n > 0.3:
        issues.append(
            "PERSISTENT THRUSTER SATURATION: Thrusters saturated {:.0f}% of the time. "
            "The controller demands forces the ship physically cannot produce. The position "
            "error keeps growing because the ship can't accelerate fast enough.".format(
                100 * saturated / n))

    for i, issue in enumerate(issues, 1):
        print(f"\n  {i}. {issue}")

    print("\n" + "-" * 80)
    print("RECOMMENDATIONS (in order of impact)")
    print("-" * 80)
    print("""
  1. COORDINATE POSITION AND HEADING (primary fix):
     Don't advance the position reference until the heading has turned enough.
     Add a speed reduction factor: v_ref *= cos(heading_error).
     When heading error > 30°, the position ref should nearly stop.

  2. SLOW DOWN POSITION OR SPEED UP HEADING:
     Either reduce ref_pos_vmax (currently {vmax:.1f} m/s) or increase
     ref_head_wn (currently {hwn:.2f}) / ref_head_rmax (currently {rmax:.1f}°/s).
     A good rule: heading should settle in less time than it takes to travel
     1/3 of the shortest leg.

  3. WAYPOINT SPACING:
     Current turns ({turns}) are large. Add intermediate waypoints to keep
     turns under 20-30° — the ship can handle those without significant crab.

  4. SYSTEM ID (for fine-tuning, not the root cause):
     The M, D, Iz values are estimates. Wrong Iz shifts yaw bandwidth.
     Wrong D means feedforward doesn't cancel drag properly.
     But even perfect sysID won't fix issues 1-3 — the architecture needs
     the coordination logic.
""".format(
        vmax=SCFG.ref_pos_vmax,
        hwn=SCFG.ref_head_wn,
        rmax=math.degrees(SCFG.ref_head_rmax),
        turns="60° and 80°",
    ))

    try:
        _plot_diagnostic(rows, csv_path, dt_avg)
    except Exception as e:
        print(f"  (Plotting skipped: {e})")

def _plot_diagnostic(rows, csv_path, dt_avg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r['t'] for r in rows]

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Crab-Motion Diagnostic — {Path(csv_path).name}", fontsize=13)

    # 1. Position error decomposed into surge and sway (body frame)
    ax = axes[0]
    ax.plot(t, [r['ex_body'] for r in rows], label='surge error (along bow)', linewidth=1.2)
    ax.plot(t, [r['ey_body'] for r in rows], label='sway error (sideways)', linewidth=1.2, color='red')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('Body-frame error [m]')
    ax.set_title('Position error decomposed: surge (ahead/behind) vs sway (sideways = crab)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    # 2. Heading error vs sway force (dual axis)
    ax = axes[1]
    epsi_deg = [math.degrees(r['epsi']) for r in rows]
    ax.plot(t, epsi_deg, label='heading error', color='blue', linewidth=1.2)
    ax.set_ylabel('Heading error [deg]', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    ax2 = ax.twinx()
    ax2.plot(t, [r['tau_y']/1000 for r in rows], label='tau_sway', color='red', alpha=0.7, linewidth=1.0)
    ax2.set_ylabel('Sway force [kN]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_title('Heading error vs Sway force — correlated = crab caused by heading lag')
    ax.grid(True, alpha=0.2)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # 3. Crab index: |tau_y| / |tau_total|
    ax = axes[2]
    crab_idx = []
    for r in rows:
        total_f = math.sqrt(r['tau_x']**2 + r['tau_y']**2)
        crab_idx.append(abs(r['tau_y']) / total_f * 100 if total_f > 500 else 0)
    ax.fill_between(t, crab_idx, alpha=0.4, color='orange')
    ax.plot(t, crab_idx, color='darkorange', linewidth=0.8)
    ax.axhline(30, color='red', linestyle='--', alpha=0.5, label='30% threshold')
    ax.set_ylabel('Crab index [%]')
    ax.set_title('Crab Index (% of control effort going sideways) — should be <20% in transit')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    # 4. Control forces
    ax = axes[3]
    ax.plot(t, [r['tau_x']/1000 for r in rows], label='tau_surge', linewidth=1.0)
    ax.plot(t, [r['tau_y']/1000 for r in rows], label='tau_sway', linewidth=1.0)
    ax.plot(t, [r['tau_psi']/1000 for r in rows], label='tau_yaw', linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('Control [kN / kNm]')
    ax.set_title('Commanded generalized forces')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # 5. Allocator scale + heading tracking
    ax = axes[4]
    ax.plot(t, [r['alloc_scale']*100 for r in rows], 'k-', linewidth=1.0, label='alloc scale')
    ax.fill_between(t, [r['alloc_scale']*100 for r in rows], 100,
                    where=[r['alloc_scale'] < 0.99 for r in rows],
                    alpha=0.2, color='red', label='saturated')
    ax.set_ylabel('Alloc scale [%]')
    ax.set_ylim(80, 105)
    ax.set_xlabel('Time [s]')
    ax.set_title('Allocator utilization (red = thrusters maxed out)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = str(_LOG_DIR / (Path(csv_path).stem + "_crab_diag.png"))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Diagnostic plot saved: {out_path}")

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    diagnose(csv_path=csv_arg)
