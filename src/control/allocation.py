# control/allocation.py
import numpy as np
from dataclasses import dataclass


@dataclass
class Geometry2Thrusters:
    lx1: float
    ly1: float
    lx2: float
    ly2: float
    lx_bow: float = 12.0
    ly_bow: float = 0.0
    biasFy: float = 0.0


class TwoThrusterAllocator:
    """
    Thruster allocator for 2 stern azimuth thrusters + 1 bow tunnel thruster.

    Maps tau = [Fx, Fy, N] to f = [Fx1, Fy1, Fx2, Fy2, Fy_bow] via pseudoinverse.

    Yaw-priority allocation: when the full demand exceeds thruster limits,
    yaw torque is preserved at full and surge/sway are scaled down until
    the allocation fits. This makes the ship slow down to turn rather than
    trying to maintain speed while crabbing sideways.
    """

    def __init__(self, geom: Geometry2Thrusters, Tmax=150_000.0, Tmax_bow=60_000.0):
        self.g = geom
        self.Tmax = float(Tmax)
        self.Tmax_bow = float(Tmax_bow)

        self.T = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [-self.g.ly1, self.g.lx1, -self.g.ly2, self.g.lx2, self.g.lx_bow]
        ], dtype=float)
        self.Tpinv = np.linalg.pinv(self.T)

        self._limits = np.array([Tmax, Tmax, Tmax, Tmax, Tmax_bow])

    def _feasible(self, f):
        return np.all(np.abs(f) <= self._limits * 1.001)

    def allocate(self, tau_x, tau_y, tau_n):
        tau = np.array([tau_x, tau_y, tau_n], dtype=float)
        f = self.Tpinv @ tau

        if self._feasible(f):
            return float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), 1.0

        # Yaw priority: check if yaw alone fits
        f_yaw_only = self.Tpinv @ np.array([0.0, 0.0, tau_n])
        if not self._feasible(f_yaw_only):
            # Even pure yaw exceeds limits — scale everything proportionally
            worst = np.max(np.abs(f) / self._limits)
            scale = 1.0 / worst
            f *= scale
            return float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), scale

        # Yaw fits — binary search for max surge/sway scale that keeps it feasible
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) * 0.5
            f_try = self.Tpinv @ np.array([tau_x * mid, tau_y * mid, tau_n])
            if self._feasible(f_try):
                lo = mid
            else:
                hi = mid

        f = self.Tpinv @ np.array([tau_x * lo, tau_y * lo, tau_n])
        f = np.clip(f, -self._limits, self._limits)
        return float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), lo
