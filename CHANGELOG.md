# Changelog — Dynamic Positioning → Waypoint Tracking Overhaul

Everything below documents the changes made on **2026-08-07** to transform the DP simulation from a basic, unstable controller demo into a working waypoint-tracking system for the R/V Gunnerus in AGX Dynamics.

> **Important clarification:** What we have now is **waypoint tracking / path following**, not true Dynamic Positioning (station-keeping). True DP — holding position against environmental disturbances — is the next phase.

---

## Table of Contents

1. [The Problem: Crab Motion](#1-the-problem-crab-motion)
2. [Vessel Parameters (System ID)](#2-vessel-parameters-system-id)
3. [PID Gain Computation (Bandwidth-Based)](#3-pid-gain-computation-bandwidth-based)
4. [Reference Filter Redesign](#4-reference-filter-redesign)
5. [LOS Guidance](#5-los-guidance)
6. [Turn Anticipation](#6-turn-anticipation)
7. [Speed-Heading Coordination](#7-speed-heading-coordination)
8. [Thruster Allocation Overhaul](#8-thruster-allocation-overhaul)
9. [Bow Thruster Integration](#9-bow-thruster-integration)
10. [Feedforward Controller](#10-feedforward-controller)
11. [Force Arrow Rendering](#11-force-arrow-rendering)
12. [Simulation & Infrastructure](#12-simulation--infrastructure)
13. [Plotting & Logging](#13-plotting--logging)
14. [Keyboard Waypoint Control](#14-keyboard-waypoint-control)
15. [Performance Results](#15-performance-results)
16. [Known Limitations](#16-known-limitations)

---

## 1. The Problem: Crab Motion

The ship was **moving sideways instead of turning** — classic "crab motion." Diagnosis (via `src/diagnose_crab.py`) revealed:

- The **position reference filter** was advancing 2–3x faster than the heading reference could turn the ship.
- The allocator was **saturated 75% of the time**, leaving no room for yaw torque.
- With thruster budget consumed by surge force (`tau_x`), sway and yaw got starved.

**Root cause:** Decoupled position and heading reference channels with mismatched bandwidths + unrealistic thrust limits.

---

## 2. Vessel Parameters (System ID)

All vessel parameters were updated to match the real R/V Gunnerus specifications. These are estimates — formal system identification has not been done yet.

**File:** `src/runtime/config.py` — `VesselParams`

| Parameter | Before | After | Unit | Why |
|-----------|--------|-------|------|-----|
| `mass` | 350,000 | **450,000** | kg | ~500t displacement, reduced slightly because AGX hydro adds virtual mass |
| `Iz` | 20e6 | **30e6** | kg·m² | Approximated as `m·(L/4)²` for a 31.25m hull |
| `Xu` (surge damping) | 50,000 | **20,000** | N·s/m | Reduced — AGX WindAndWater adds its own surge drag |
| `Yv` (sway damping) | 80,000 | **50,000** | N·s/m | Sway drag reduced; AGX hydro contributes additional damping |
| `Nr` (yaw damping) | 2,000,000 | **15,000,000** | N·m·s/rad | Measured from AGX hydro behavior — the ship resists yaw strongly |
| `half_length` | 10.0 | **15.6** | m | LOA/2 = 31.25/2 (Gunnerus specs) |
| `half_width` | 3.75 | **4.95** | m | Beam/2 = 9.9/2 |
| `thr_port_x` | -10.0 | **-14.0** | m | ~1.5m forward of stern (stern at -15.6m) |
| `thr_port_y` | +2.76 | **+3.0** | m | Wider spacing on 9.9m beam |
| `thr_star_x` | -10.0 | **-14.0** | m | Mirrors port thruster |
| `thr_star_y` | -2.76 | **-3.0** | m | Mirrors port thruster |
| `Tmax_thruster` | 500,000 | **150,000** | N | ~15t bollard pull per PM azimuth thruster (realistic for Gunnerus) |

**New parameters added:**

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `thr_bow_x` | 12.0 | m | Bow tunnel thruster, ~3m from bow |
| `thr_bow_y` | 0.0 | m | Centerline |
| `Tmax_bow` | 60,000 | N | Brunvoll tunnel thruster, ~6t lateral only |

**Removed:** `wave_gain_scale_low`, `wave_gain_scale_high` (wave-adaptive gain scaling was unused).

---

## 3. PID Gain Computation (Bandwidth-Based)

**Before:** 9 manually-tuned PID gains hardcoded in `SceneParams`.

**After:** Gains are **derived from physics** via 3 intuitive bandwidth parameters. The function `compute_pid_gains()` in `config.py` computes them:

```
Kp = M * wn²
Kd = max(0, 2*M*wn - D)
Ki = Kp * wn / 10
```

For sway, gains = surge gains × `sway_ratio` (0.3), since sway authority is limited.

**File:** `src/runtime/config.py` — `SceneParams` + `compute_pid_gains()`

| Bandwidth Parameter | Value | Unit | Description |
|---------------------|-------|------|-------------|
| `wn_pos` | 0.25 | rad/s | Position control bandwidth |
| `wn_psi` | 0.15 | rad/s | Heading control bandwidth |
| `sway_ratio` | 0.3 | — | Sway gains = surge gains × this |
| `tau_max` | 300,000 | N | Matches 2×150kN stern thruster budget |

**Resulting PID gains:**

| Gain | Surge (x) | Sway (y) | Yaw (psi) | Unit |
|------|-----------|----------|-----------|------|
| Kp | 28,125 | 8,437.5 | 675,000 | N/m or N·m/rad |
| Kd | 205,000 | 61,500 | 0* | N·s/m or N·m·s/rad |
| Ki | 703.1 | 210.9 | 10,125 | N/(m·s) or N·m/(rad·s) |

*`Kd_psi = 0` because `Nr` (15M) already exceeds `2·Iz·wn_psi` (9M). The physical yaw damping is strong enough that no additional derivative action is needed.*

---

## 4. Reference Filter Redesign

The 2nd-order reference filter generates smooth position and heading trajectories for the controller to track.

**File:** `src/control/reference.py`

**Changes:**
- Removed `zeta` and `Ki` parameters from `PosRefParams` and `HeadRefParams` — they were unused or caused issues. The filter now uses `omega` (natural frequency) and velocity/rate limits only.
- Added `speed_factor` parameter to `step()` method for speed-heading coordination (see §7).

| Ref Filter Parameter | Before | After | Unit | Why |
|----------------------|--------|-------|------|-----|
| `ref_pos_vmax` | 0.3 | **3.0** | m/s | ~6kn maneuvering speed — leaves 80% thrust for sway/yaw |
| `ref_pos_wn` | 0.08 | **0.10** | rad/s | Slightly above heading wn (0.08) so ship arrives before needing to turn |
| `ref_head_wn` | 0.10 | **0.08** | rad/s | Reduced to fit within yaw torque budget |
| `ref_head_rmax` | 0.05 | **0.03** | rad/s | ~1.7°/s max yaw rate — sustainable against AGX hydro damping |

**Key constraint:** The heading bandwidth must keep peak yaw demand (`Iz·α + Nr·r`) below the available yaw torque budget (~700 kNm from 2 stern thrusters at 14m lever arm). At `wn=0.15`, peak demand was ~1125 kNm (saturated). At `wn=0.08`, peak is ~640 kNm (fits).

---

## 5. LOS Guidance

**Before:** The heading reference was simply the angle of the current path segment (`phi`). The ship pointed toward the next waypoint but had no way to correct cross-track error.

**After:** Line-of-Sight (LOS) guidance steers the bow toward the path using the cross-track error:

```python
e_crosstrack = -dx * sin(phi) + dy * cos(phi)
psi_los = phi + atan2(-e_crosstrack, DELTA_LOS)
```

**File:** `src/runtime/runner.py`, lines 287–292

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `DELTA_LOS` | 80 | m | Lookahead distance (~2.5× ship length). Larger = smoother path, smaller = tighter tracking |

**How it works:** When the ship drifts off the path, `e_crosstrack` grows, and the LOS angle rotates the heading command back toward the path. The lookahead distance `DELTA_LOS` controls how aggressively it corrects — too small causes oscillation, too large lets the ship cut corners.

---

## 6. Turn Anticipation

**Before:** The ship waited until it reached each waypoint before turning, causing aggressive snap-turns that heeled the vessel and saturated thrusters.

**After:** The heading reference blends toward the next leg's direction when approaching a waypoint:

```python
ANTICIPATION_DIST = 50.0
if rem_alng < ANTICIPATION_DIST and wp_idx < len(goals) - 1:
    phi_next = angle_of_line(current_wp, next_wp)
    blend = 1.0 - rem_alng / ANTICIPATION_DIST
    psi_los = wrap(psi_los + blend * wrap(phi_next - psi_los))
```

**File:** `src/runtime/runner.py`, lines 295–300

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `ANTICIPATION_DIST` | 50 | m | Distance before waypoint where heading starts blending toward next leg |

**Note:** We tried 80m but it caused drifting on 106–144m legs (ship started turning 55–75% through each leg, cutting corners). 50m is the sweet spot for the current waypoint geometry.

---

## 7. Speed-Heading Coordination

This was the **key fix for crab motion**. The position reference now slows down when the ship hasn't turned to face the path yet:

```python
hdg_err = wrap_pi(psi_target - psi)
speed_factor = max(0.0, cos(hdg_err))
ref.step(dt, pd=L_path, psi_d=psi_target, speed_factor=speed_factor)
```

Inside the reference filter:
```python
vmax_eff = self.pp.vmax * max(0.0, min(1.0, speed_factor))
```

**File:** `src/runtime/runner.py` lines 306–307, `src/control/reference.py` line 122

**How it works:**
- When heading error = 0° → `cos(0) = 1.0` → full speed
- When heading error = 45° → `cos(45°) ≈ 0.7` → 70% speed
- When heading error = 90° → `cos(90°) = 0.0` → reference stops advancing

This prevents the position reference from running ahead of the ship when it needs to turn, which was the root cause of crab motion.

---

## 8. Thruster Allocation Overhaul

**Before:** Simple pseudoinverse with per-axis box saturation. 4-column matrix (2 thrusters × 2 DOF each). No priority scheme — when saturated, all axes were clipped equally, often destroying yaw torque.

**After:** Pseudoinverse with **yaw-priority binary search**. 5-column matrix (2 stern × 2 DOF + 1 bow lateral). When the full demand exceeds limits, yaw torque is preserved and surge/sway are scaled down.

**File:** `src/control/allocation.py`

**Allocation algorithm:**
1. Compute `f = T_pinv @ tau` (5 forces from 3 demands)
2. If feasible (all forces within limits) → done, scale = 1.0
3. If even pure yaw exceeds limits → scale everything proportionally
4. Otherwise → binary search (12 iterations) for maximum surge/sway scaling that keeps yaw at full while staying within limits

**Returns:** `(Fx1, Fy1, Fx2, Fy2, Fy_bow, alloc_scale)` — the 6th value is the allocation scale (1.0 = no saturation, <1.0 = surge/sway were reduced).

**Configuration matrix T:**
```
T = | 1    0    1    0    0     |   <- Fx (surge)
    | 0    1    0    1    1     |   <- Fy (sway)
    | -ly1 lx1  -ly2 lx2  lx_bow |   <- N  (yaw moment)
```

**Known issue:** Fy1 ≡ Fy2 always. Columns 2 and 4 of T are identical (`[0, 1, -14]` for both stern thrusters at the same x-position), so the pseudoinverse distributes lateral force equally. A proper azimuth-angle-based allocator would fix this.

---

## 9. Bow Thruster Integration

Added a Brunvoll bow tunnel thruster as a third actuator.

**Files:** `src/modeling/vessel.py`, `src/runtime/runner.py`, `src/control/allocation.py`, `src/runtime/config.py`

| Property | Value | Description |
|----------|-------|-------------|
| Position x | 12.0 m | ~3m from bow |
| Position y | 0.0 m | Centerline |
| Max force | 60,000 N | Lateral only (tunnel thruster) |
| Visual arrow color | Cyan (0.2, 0.8, 1.0) | Distinct from orange stern arrows |

The bow thruster contributes lateral force and yaw moment (`12m × Fy_bow`). It's included in the pseudoinverse allocation matrix.

---

## 10. Feedforward Controller

The PID+FF controller computes: `tau = M·nudot_r + D·nu_r + Kp·e_pos + Kd·e_vel + integral`

**File:** `src/control/controller.py`

**Key aspects:**
- Position errors are transformed to **body frame** before applying Kp gains
- Sway tau is capped at `tau_max × 0.3` (90 kN) to prevent sway from eating the entire thruster budget
- Yaw tau is capped at `tau_max × 3.0` (900 kNm)
- Anti-windup with **back-calculation**: when saturated, integral decays at 10%/s (50%/s for yaw)
- Integral accumulator limits: surge 30% of tau_max, sway 30% of sway_max, yaw 20% of yaw_max

**Feedforward is NOT double-counting AGX hydro damping.** The physics equation is `M·ν̇ = τ_thrusters - D·ν` (AGX applies drag, thrusters must overcome it). So `τ = M·ν̇r + D·νr` is correct — the thrusters need to produce enough force to both accelerate the vessel AND compensate for the drag that AGX is applying.

---

## 11. Force Arrow Rendering

**Before:** Arrows teleported between frames when force direction changed. Used body-local coordinates incorrectly (didn't account for mesh alignment quaternion). Hide threshold was 500 N.

**After:**
- **Low-pass filter smoothing** (alpha=0.3) on position and angle — arrows glide instead of jumping
- Force direction uses `ship_force_to_world()` which accounts for the Gunnerus mesh alignment quaternion
- Hide threshold raised to **2 kN** (arrows below this disappear)
- Rotation fix: `angle - π/2` corrects the arrow geometry orientation

**File:** `src/runtime/runner.py`, `_update_force_arrow()` function

---

## 12. Simulation & Infrastructure

| Change | Before | After | Why |
|--------|--------|-------|-----|
| Sim speed | Real-time | **4× speed** (`simulation().setTimeStep(dt * 4.0)`) | Faster iteration during tuning |
| Spawn position | (0, 0) center of ocean | **(-200, 0)** south edge | More room for waypoints on 500×500m ocean |
| Ship z-spawn | 2.0 (dropped from air) | **0.0** + KINEMATICS settle | No splash/bounce at start |
| Settle phase | None | **0.5s KINEMATICS** then switch to DYNAMICS | Ship stabilizes before control starts |
| Log directory | `src/runtime/` | **`logs/`** at project root | Cleaner project structure |
| Disturbance | Enabled (80kN lateral at t=40s) | **Disabled** (time=9999) | Focus on path following first |
| Waves | height=1.5 | **height=0.0** | Calm water for tuning |
| Goal tolerance | 5.0 m | **8.0 m** | Prevents overshooting at waypoints with the larger ship |

**Waypoints:**

| Before | After |
|--------|-------|
| `(0, 0), (30, 15), (50, -10), (60, 20)` | `(-200, 0), (-80, 40), (0, -30), (120, 50), (200, -20)` |

Legs are now 106–144m long (vs 18–36m before), appropriate for a 31m vessel maneuvering at 3 m/s.

---

## 13. Plotting & Logging

**Logging** (`runner.py`): Extended from 18 columns to 26 columns:

| New Columns | Description |
|-------------|-------------|
| `ex_body`, `ey_body` | Position error in body frame (surge/sway) |
| `pos_err` | Euclidean position error magnitude |
| `Fy_bow` | Bow thruster force |
| `alloc_scale` | Allocator utilization (1.0 = no saturation) |
| `sigma_x`, `sigma_y`, `sigma_psi` | PID integral accumulators |

**Plotting** (`plot_log.py`): Upgraded from basic line plots to a 4×3 grid:

| Panel | What it shows |
|-------|---------------|
| Row 0, left | Position tracking (x, y vs reference) over time |
| Row 0, right | 2D trajectory with heading arrows |
| Row 1, left | Heading tracking (psi vs reference) |
| Row 1, right | Body-frame errors (surge, sway, heading) |
| Row 2, left | Commanded generalized forces (tau_x, tau_y, tau_psi) |
| Row 2, right | Individual thruster forces (Fx1, Fy1, Fx2, Fy2, Fy_bow) |
| Row 3, left | Allocator utilization (% with red fill when saturated) |
| Row 3, center | PID integral accumulators |
| Row 3, right | Position error magnitude with goal tolerance line |

Disturbance periods are shaded red across all time-series panels.

---

## 14. Keyboard Waypoint Control

Added runtime waypoint editing via keyboard:

| Key | Action |
|-----|--------|
| `N` | Cycle through waypoints (select next) |
| `↑` Arrow | Move selected waypoint +10m in X |
| `↓` Arrow | Move selected waypoint -10m in X |
| `→` Arrow | Move selected waypoint -10m in Y |
| `←` Arrow | Move selected waypoint +10m in Y |

**File:** `src/runtime/runner.py`, lines 81–118

Waypoint markers update visually in real-time when moved.

---

## 15. Performance Results

From the latest run (`dp_log_20260807_214114`, duration 220s):

| Metric | Before (estimated) | After | Unit |
|--------|-------------------|-------|------|
| Position error mean | ~41 | **5.4** | m |
| Position error max | >50 | **11.6** | m |
| Sway error max | ~60 | **10.6** | m |
| Heading error max | >26 | **14.5** | deg |
| Allocator saturated | ~75% | **12%** | of time |
| Min allocator scale | <0.3 | **0.95** | — |

---

## 16. Known Limitations

1. **Not true DP yet** — this is waypoint tracking (path following). True DP (station-keeping against disturbances) has not been tested.
2. **System ID not done** — M, D, Iz, Nr are estimates. Nr=15M in particular was "measured" from AGX sim behavior, not formally identified.
3. **Fy1 ≡ Fy2 always** — pseudoinverse can't differentiate between symmetric stern thrusters. Needs an azimuth-angle-based allocator.
4. **Feedforward D values unvalidated** — the drag coefficients in the controller should ideally match what AGX's WindAndWaterController actually applies.
5. **No wave/current disturbance** — waves and disturbance force are disabled for clean tuning.
6. **`Kd_psi = 0`** — yaw derivative gain is zero because Nr (15M) > 2·Iz·wn_psi (9M). The physical damping provides all the derivative action needed, but this should be verified with system ID.

---

## Files Changed Summary

| File | What changed |
|------|-------------|
| `src/runtime/config.py` | Vessel params for Gunnerus, bandwidth-based PID gains, new waypoints, bow thruster params |
| `src/runtime/runner.py` | LOS guidance, turn anticipation, speed-heading coordination, bow thruster, 4× sim speed, settle phase, keyboard waypoint control, extended logging, force arrow smoothing |
| `src/control/reference.py` | `speed_factor` parameter, simplified constructor (removed zeta/Ki) |
| `src/control/controller.py` | Sway/yaw saturation limits, back-calculation anti-windup |
| `src/control/allocation.py` | 5-actuator pseudoinverse, yaw-priority binary search, bow thruster column |
| `src/modeling/vessel.py` | Bow thruster position, `ship_force_to_world()` method, mesh alignment fixes |
| `src/plot_log.py` | 4×3 grid layout, allocator/integral panels, body-frame error display |
| `src/agx_wrap/world.py` | Minor: force arrow and waypoint marker updates |
| `src/diagnose_crab.py` | **New file** — diagnostic script for analyzing crab motion from logs |
