# control/controller.py
"""
PID + Feedforward controller for a simplified 3-DOF vessel model (surge, sway, yaw).

This module provides:
- PIDGains: Dataclass holding controller gains and limits with detailed explanations.
- PIDFFController: A combined feedforward + PID controller for a vessel modeled in
  surge (x), sway (y) and yaw (psi). The controller computes control efforts (tau)
  that are suitable as generalized forces/torques for an AGX vessel model.

Conceptual model (variables):
- eta = [x, y, psi]  : pose in world frame (position x,y and yaw psi)
- nu  = [u, v, r]    : body-fixed velocities (surge, sway, yaw-rate)
- nudot = [udot, vdot, rdot] : accelerations in body frame (feedforward)
- tau = control generalized force vector [tau_x, tau_y, tau_psi]

Controller law implemented (component-wise):
  tau = tau_ff + Kp * e_p + Kd * e_v + sigma
where
  tau_ff = M * nudot_r + D * nu_r   (simple diagonal inertia + damping feedforward)
  e_p = eta_r - eta_hat            (position/orientation error; psi wrapped)
  e_v = nu_r - nu_hat              (velocity error)
  sigma = integral term (anti-windup applied)

Notes:
- Position errors for x,y are assumed expressed in the world frame (runner must ensure consistent frames).
- Yaw error is angle-wrapped into [-pi, pi] before use.
- Integral anti-windup: integrator increments only when control is not saturated.
- This controller uses diagonal approximations of M and D (vectors of three elements).
"""
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import math


@dataclass
class PIDGains:
    """
    Holds PID gains and saturation limits used by PIDFFController.

    Attributes
    ----------
    Kp_x, Kp_y, Kp_psi : float
        Proportional gains for surge (x), sway (y) and yaw (psi).
        Units: [N/m] for translation terms if position in meters and tau in Newtons,
               [Nm/rad] for yaw if psi in radians and tau_psi in Newton-meters.
    Kd_x, Kd_y, Kd_psi : float
        Derivative gains for velocity errors. Units consistent with Kp above divided by seconds.
    Ki_x, Ki_y, Ki_psi : float
        Integral gains. Units consistent with Kp * 1/s.
        Default zero: no integral action unless explicitly set.
    tau_max : float
        Symmetric scalar saturation limit applied component-wise to each tau output.
        Typical value must be chosen according to thruster capacity (N or Nm).
    """
    Kp_x: float = 15_000.0
    Kp_y: float = 15_000.0
    Kp_psi: float = 500_000.0
    Kd_x: float = 50_000.0
    Kd_y: float = 50_000.0
    Kd_psi: float = 2_000_000.0
    Ki_x: float = 200.0
    Ki_y: float = 200.0
    Ki_psi: float = 1_000.0
    tau_max: float = 150_000.0


@dataclass
class ThrusterGeometry:
    """
    Thruster positions in body frame for decoupling calculations.
    
    Note: Decoupling is currently disabled due to instability issues.
    This class is kept for future use with proper allocation-aware decoupling.
    """
    lx1: float = -10.0  # Thruster 1 x-position (port)
    ly1: float = 2.76   # Thruster 1 y-position
    lx2: float = -10.0  # Thruster 2 x-position (starboard)
    ly2: float = -2.76  # Thruster 2 y-position


class PIDFFController:
    """
    PID + Feedforward controller for 3-DOF vessel dynamics.

    Initialization
    --------------
    PIDFFController(M_diag, D_diag, gains, thruster_geom=None)

    Parameters
    ----------
    M_diag : Sequence[float]
        Diagonal inertia vector [m_x, m_y, I_z] used for feedforward:
        tau_ff_component = M_diag[i] * nudot_r[i]
    D_diag : Sequence[float]
        Diagonal damping vector [Xu, Yv, Nr] used in feedforward:
        tau_ff_component += D_diag[i] * nu_r[i]
    gains : PIDGains
        Gains and saturation limits.
    thruster_geom : ThrusterGeometry, optional
        Thruster geometry for potential future decoupling (currently disabled).

    Internal state
    --------------
    sigma_x, sigma_y, sigma_psi : floats
        Integral accumulator (anti-windup handling implemented).

    Behavior
    --------
    - Computes tau = tau_ff + Kp*e_pos + Kd*e_vel + sigma
    - Saturates each tau component to [-tau_max, tau_max]
    - Anti-windup: integrator increments only when the pre-saturated control is within limits.
    """

    def __init__(self, M_diag: Sequence[float], D_diag: Sequence[float], gains: PIDGains,
                 thruster_geom: Optional[ThrusterGeometry] = None):
        self.M = list(M_diag)
        self.D = list(D_diag)
        self.g = gains
        
        # Integral accumulators
        self.sigma_x = 0.0
        self.sigma_y = 0.0
        self.sigma_psi = 0.0
        
        # Store geometry but disable decoupling (caused instability)
        self.geom = None  # Decoupling disabled
        self.sway_to_yaw_coupling = 0.0
        self.yaw_moment_arm = 1.0

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap angle to [-pi, pi] using trig identity"""
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _sat(val: float, limit: float) -> float:
        """Symmetric saturation."""
        return max(-limit, min(limit, val))

    def reset(self):
        """Reset integral accumulators."""
        self.sigma_x = 0.0
        self.sigma_y = 0.0
        self.sigma_psi = 0.0

    def step(
        self,
        dt: float,
        eta_r: Sequence[float],
        nu_r: Sequence[float],
        nudot_r: Sequence[float],
        eta_hat: Sequence[float],
        nu_hat: Sequence[float],
    ) -> Tuple[float, float, float]:
        """
        Compute control forces/torques for one time step.
        
        Parameters
        ----------
        dt : float
            Time step in seconds.
        eta_r : Sequence[float]
            Reference pose [x_r, y_r, psi_r].
        nu_r : Sequence[float]
            Reference body velocities [u_r, v_r, r_r].
        nudot_r : Sequence[float]
            Reference body accelerations [udot_r, vdot_r, rdot_r].
        eta_hat : Sequence[float]
            Estimated pose [x, y, psi].
        nu_hat : Sequence[float]
            Estimated body velocities [u, v, r].
            
        Returns
        -------
        Tuple[float, float, float]
            Control forces/torques (tau_x, tau_y, tau_psi) in body frame.
        """
        g = self.g
        tau_max = g.tau_max
        
        # Unpack references and estimates
        x_r, y_r, psi_r = eta_r
        u_r, v_r, r_r = nu_r
        udot_r, vdot_r, rdot_r = nudot_r
        
        x_hat, y_hat, psi_hat = eta_hat
        u_hat, v_hat, r_hat = nu_hat
        
        # Position/orientation errors (world frame for x,y; wrapped for psi)
        e_x = x_r - x_hat
        e_y = y_r - y_hat
        e_psi = math.atan2(math.sin(psi_r - psi_hat), math.cos(psi_r - psi_hat))
        
        # Velocity errors (body frame)
        e_u = u_r - u_hat
        e_v = v_r - v_hat
        e_r = r_r - r_hat
        
        # Transform position errors to body frame for surge/sway control
        c = math.cos(psi_hat)
        s = math.sin(psi_hat)
        e_x_body = c * e_x + s * e_y
        e_y_body = -s * e_x + c * e_y
        
        # Feedforward: tau_ff = M * nudot_r + D * nu_r
        tau_ff_x = self.M[0] * udot_r + self.D[0] * u_r
        tau_ff_y = self.M[1] * vdot_r + self.D[1] * v_r
        tau_ff_psi = self.M[2] * rdot_r + self.D[2] * r_r
        
        tau_psi_max = tau_max * 3.0
        
        # PID terms
        # Surge (x)
        tau_x_unsaturated = (tau_ff_x +
                             g.Kp_x * e_x_body +
                             g.Kd_x * e_u +
                             self.sigma_x)
        
        # Sway (y)
        tau_y_unsaturated = (tau_ff_y +
                             g.Kp_y * e_y_body +
                             g.Kd_y * e_v +
                             self.sigma_y)
        
        # Yaw (psi)
        tau_psi_unsaturated = (tau_ff_psi +
                               g.Kp_psi * e_psi +
                               g.Kd_psi * e_r +
                               self.sigma_psi)
        
        tau_sway_max = tau_max * 0.3

        tau_x = self._sat(tau_x_unsaturated, tau_max)
        tau_y = self._sat(tau_y_unsaturated, tau_sway_max)
        tau_psi = self._sat(tau_psi_unsaturated, tau_psi_max)
        
        # Anti-windup
        if abs(tau_x_unsaturated) < tau_max:
            self.sigma_x += g.Ki_x * e_x_body * dt
        else:
            # Back-calculation: decay integral toward zero when saturated
            self.sigma_x *= (1.0 - 0.1 * dt)
        self.sigma_x = self._sat(self.sigma_x, tau_max * 0.3)
        
        # Sway
        if abs(tau_y_unsaturated) < tau_sway_max:
            self.sigma_y += g.Ki_y * e_y_body * dt
        else:
            self.sigma_y *= (1.0 - 0.1 * dt)
        self.sigma_y = self._sat(self.sigma_y, tau_sway_max * 0.3)
        
        # Yaw
        if abs(tau_psi_unsaturated) < tau_psi_max:
            self.sigma_psi += g.Ki_psi * e_psi * dt
        else:
            # Aggressive back-calculation when yaw is saturated to prevent limit cycling
            self.sigma_psi *= (1.0 - 0.5 * dt)
        self.sigma_psi = self._sat(self.sigma_psi, tau_psi_max * 0.2)
        
        return tau_x, tau_y, tau_psi