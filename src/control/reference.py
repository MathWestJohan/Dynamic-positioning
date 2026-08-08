# control/reference.py
import math
from dataclasses import dataclass

"""
1D 2nd-order tracking filter for distance along a line and heading.

This class implements two decoupled 2nd-order reference systems:

- Position channel: generates a smoothed position reference `pr` along a line
  and the corresponding velocity `vr`, given a desired distance `pd`. The output
  velocity is saturated to a maximum allowed value to prevent excessive acceleration.
  
- Heading channel: generates a smoothed heading reference `psir` and yaw-rate `rr`
  given a desired heading `psi_d`. The yaw-rate is limited to a maximum value.

Constructor
-----------
ReferenceFilter(pos_params: PosRefParams, head_params: HeadRefParams)

Methods
-------
reset(psi_now: float = 0.0)
    Resets the internal states (position, velocity, heading, yaw-rate) to zero
    and sets the heading reference to `psi_now`.
    
step(dt: float, pd: float, psi_d: float) -> Tuple[float, float, float, float]
    Advances the filter by `dt` seconds using the desired distance `pd` and
    desired heading `psi_d`. Returns smoothed references:
        - pr : filtered distance along the line (m)
        - vr : filtered velocity along the line (m/s)
        - rr : filtered yaw-rate (rad/s)
        - psir : filtered heading reference (rad)

Inputs and units
----------------
- pd : desired scalar distance along the line (meters)
- psi_d : desired heading (radians)
- dt : time step in seconds
- pr : smoothed position reference (meters)
- vr : smoothed velocity reference (m/s)
- rr : smoothed yaw-rate reference (rad/s)
- psir : smoothed heading reference (rad)
"""

def sat(val, vmin, vmax):
    return max(vmin, min(vmax, val))

@dataclass
class PosRefParams:
    omega: float = 0.3      # ω_r,p
    zeta: float  = 1.0      # ζ_r,p
    Ki: float    = 0.1     # integral gain for xi_p
    vmax: float  = 2.0      # m/s cap

@dataclass
class HeadRefParams:
    omega: float = 0.5
    zeta: float  = 1.0
    Ki: float    = 0.2
    rmax: float  = 0.5      # rad/s cap

class ReferenceFilter:
    """
    1D 2nd-order tracking filter for distance along a line and a separate heading filter.

    This class implements two decoupled 2nd-order reference systems:
    - Position channel: produces pr (distance), vr (speed along the line) and an
      internal integrator xi_p used to track a desired distance pd.
    - Heading channel: produces psi_r (heading reference) and rr (yaw-rate) using a
      separate integrator xi_psi to track the desired heading psi_d.

    Constructor
    -----------
    ReferenceFilter(pos_params: PosRefParams, head_params: HeadRefParams)

    Methods
    -------
    reset(psi_now: float = 0.0)
        Reset internal states to zeros and set psi reference to psi_now.
    step(dt: float, pd: float, psi_d: float) -> Tuple[float, float, float, float]
        Advance the filter by dt seconds using desired distance pd and desired heading
        psi_d. Returns (pr, vr, rr, psi_r).

    Inputs - Outputs and units
    ----------------
    Inputs:
    - pd : desired scalar distance (m) along the line (1D command).
    - psi_d : desired heading (rad).
    - dt : time step in seconds.
    
    Outputs:
    - pr : produced reference distance in meters.
    - vr : produced reference speed in m/s.
    - rr : produced reference yaw-rate in rad/s.
    - psi_r : produced reference heading in rad.
    """
    def __init__(self, pos_params: PosRefParams, head_params: HeadRefParams):
        self.pp = pos_params
        self.hp = head_params
        # position channel internal states
        self.pr = 0.0      # distance (m)
        self.vr = 0.0      # speed along line (m/s)
        # heading channel internal states
        self.psir = 0.0    # heading reference
        self.rr = 0.0      # yaw rate

    @staticmethod
    def _wrap_pi(a):
        return math.atan2(math.sin(a), math.cos(a))

    def reset(self, psi_now=0.0):
        self.pr = 0.0
        self.vr = 0.0
        self.psir = psi_now
        self.rr = 0.0

    def step(self, dt, pd, psi_d, speed_factor=1.0):
        # --- Position channel ---
        # Speed-heading coordination: scale velocity limit by speed_factor
        # so the position reference slows down when the ship hasn't turned yet.
        vmax_eff = self.pp.vmax * max(0.0, min(1.0, speed_factor))

        # Compute acceleration
        ar = self.pp.omega**2 * (pd - self.pr) - 2 * self.pp.zeta * self.pp.omega * self.vr

        # Update velocity with limit
        vr_old = self.vr
        self.vr += ar * dt
        self.vr = sat(self.vr, -vmax_eff, vmax_eff)

        # Effective acceleration after velocity clamping (for feedforward)
        ar = (self.vr - vr_old) / dt if dt > 0 else 0.0

        # Update position
        self.pr += self.vr * dt
        
        # Clamp pr to not overshoot pd
        if pd >= 0 and self.pr > pd:
            self.pr = pd
            self.vr = 0.0
        elif pd < 0 and self.pr < pd:
            self.pr = pd
            self.vr = 0.0

        # --- Heading channel ---
        epsi = self._wrap_pi(psi_d - self.psir)
        
        # Compute angular acceleration
        alpha = self.hp.omega**2 * epsi - 2 * self.hp.zeta * self.hp.omega * self.rr

        # Update yaw rate with limit
        rr_old = self.rr
        self.rr += alpha * dt
        self.rr = sat(self.rr, -self.hp.rmax, self.hp.rmax)

        # Effective angular acceleration after rate clamping (for feedforward)
        alpha = (self.rr - rr_old) / dt if dt > 0 else 0.0

        # Update heading
        self.psir = self._wrap_pi(self.psir + self.rr * dt)

        return self.pr, self.vr, ar, self.rr, alpha, self.psir
