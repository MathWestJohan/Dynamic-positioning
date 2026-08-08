import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"

def find_latest_log():
    pattern = str(_LOG_DIR / "dp_log_*.csv")
    logs = sorted(glob.glob(pattern))
    if logs:
        return logs[-1]
    # Fallback: old location
    old_pattern = str(Path(__file__).parent / "runtime" / "dp_log_*.csv")
    old_logs = sorted(glob.glob(old_pattern))
    if old_logs:
        return old_logs[-1]
    return None

def plot_dp_log(csv_path=None, out_path=None):
    if csv_path is None:
        csv_path = find_latest_log()
        if csv_path is None:
            print("No log file found.")
            return

    if out_path is None:
        out_path = str(_LOG_DIR / (Path(csv_path).stem + "_plot.png"))

    df = pd.read_csv(csv_path)
    has_dist = "disturbance" in df.columns
    has_body_err = "ex_body" in df.columns
    has_alloc = "alloc_scale" in df.columns
    has_integral = "sigma_x" in df.columns

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"DP Log: {Path(csv_path).name}", fontsize=13, y=0.98)

    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # --- Row 0: Position tracking + 2D trajectory ---
    ax_pos = fig.add_subplot(gs[0, 0:2])
    ax_pos.plot(df.t, df.x, label="x", linewidth=1.2)
    ax_pos.plot(df.t, df.xr, "--", label="x_ref", alpha=0.7)
    ax_pos.plot(df.t, df.y, label="y", linewidth=1.2)
    ax_pos.plot(df.t, df.yr, "--", label="y_ref", alpha=0.7)
    ax_pos.set_ylabel("Position [m]")
    ax_pos.legend(loc="upper left", fontsize=8)
    ax_pos.set_title("Position Tracking")
    ax_pos.grid(True, alpha=0.2)

    ax_traj = fig.add_subplot(gs[0, 2])
    ax_traj.plot(df.x, df.y, "b-", linewidth=1.2, label="Actual", alpha=0.8)
    ax_traj.plot(df.xr, df.yr, "g--", linewidth=1.0, label="Reference", alpha=0.6)
    ax_traj.plot(df.x.iloc[0], df.y.iloc[0], "bs", markersize=8, label="Start")
    ax_traj.plot(df.x.iloc[-1], df.y.iloc[-1], "r^", markersize=8, label="End")
    n = len(df)
    skip = max(1, n // 15)
    for i in range(0, n, skip):
        dx = np.cos(df.psi.iloc[i]) * 3
        dy = np.sin(df.psi.iloc[i]) * 3
        ax_traj.annotate("", xy=(df.x.iloc[i]+dx, df.y.iloc[i]+dy),
                         xytext=(df.x.iloc[i], df.y.iloc[i]),
                         arrowprops=dict(arrowstyle="->", color="blue", alpha=0.4, lw=0.8))
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.set_title("2D Trajectory")
    ax_traj.set_aspect("equal")
    ax_traj.legend(loc="best", fontsize=7)
    ax_traj.grid(True, alpha=0.2)

    # --- Row 1: Heading + Position/Heading error ---
    ax_hdg = fig.add_subplot(gs[1, 0:2])
    psi_deg = np.degrees(np.unwrap(df.psi))
    psir_deg = np.degrees(np.unwrap(df.psir))
    ax_hdg.plot(df.t, psi_deg, label="psi", linewidth=1.2)
    ax_hdg.plot(df.t, psir_deg, "--", label="psi_ref", alpha=0.7)
    ax_hdg.set_ylabel("Heading [deg]")
    ax_hdg.legend(loc="upper left", fontsize=8)
    ax_hdg.set_title("Heading Tracking")
    ax_hdg.grid(True, alpha=0.2)

    ax_err = fig.add_subplot(gs[1, 2])
    if has_body_err:
        ax_err.plot(df.t, df.ex_body, label="surge err", linewidth=1.0)
        ax_err.plot(df.t, df.ey_body, label="sway err", linewidth=1.0)
    else:
        ax_err.plot(df.t, df.ex, label="e_x", linewidth=1.0)
        ax_err.plot(df.t, df.ey, label="e_y", linewidth=1.0)
    epsi_deg = np.degrees(df.epsi)
    ax_err2 = ax_err.twinx()
    ax_err2.plot(df.t, epsi_deg, "r-", label="heading err", linewidth=1.0, alpha=0.7)
    ax_err2.set_ylabel("Heading error [deg]", color="r", fontsize=8)
    ax_err.set_ylabel("Position error [m]")
    ax_err.set_title("Body-frame Errors")
    ax_err.legend(loc="upper left", fontsize=7)
    ax_err2.legend(loc="upper right", fontsize=7)
    ax_err.grid(True, alpha=0.2)

    # --- Row 2: Control effort + Thruster forces ---
    ax_tau = fig.add_subplot(gs[2, 0:2])
    ax_tau.plot(df.t, df.tau_x / 1000, label="tau_x")
    ax_tau.plot(df.t, df.tau_y / 1000, label="tau_y")
    ax_tau.plot(df.t, df.tau_psi / 1000, label="tau_psi")
    ax_tau.set_ylabel("Control [kN / kNm]")
    ax_tau.legend(loc="upper left", fontsize=8)
    ax_tau.set_title("Commanded Generalized Forces")
    ax_tau.grid(True, alpha=0.2)

    ax_thr = fig.add_subplot(gs[2, 2])
    ax_thr.plot(df.t, df.Fx1 / 1000, label="Fx1", linewidth=0.8)
    ax_thr.plot(df.t, df.Fy1 / 1000, label="Fy1", linewidth=0.8)
    ax_thr.plot(df.t, df.Fx2 / 1000, label="Fx2", linewidth=0.8)
    ax_thr.plot(df.t, df.Fy2 / 1000, label="Fy2", linewidth=0.8)
    if "Fy_bow" in df.columns:
        ax_thr.plot(df.t, df.Fy_bow / 1000, label="Fy_bow", linewidth=0.8, linestyle="--")
    ax_thr.set_ylabel("Force [kN]")
    ax_thr.set_title("Individual Thruster Forces")
    ax_thr.legend(loc="upper left", fontsize=7)
    ax_thr.grid(True, alpha=0.2)

    # --- Row 3: Allocator saturation + Integral windup ---
    ax_alloc = fig.add_subplot(gs[3, 0])
    if has_alloc:
        ax_alloc.plot(df.t, df.alloc_scale * 100, "k-", linewidth=1.0)
        ax_alloc.axhline(100, color="g", linestyle="--", alpha=0.4, label="No saturation")
        ax_alloc.fill_between(df.t, df.alloc_scale * 100, 100,
                              where=df.alloc_scale < 1.0, alpha=0.2, color="red")
        ax_alloc.set_ylabel("Alloc scale [%]")
        ax_alloc.set_title("Allocator Utilization")
        ax_alloc.set_ylim(0, 110)
        ax_alloc.legend(fontsize=7)
    else:
        ax_alloc.text(0.5, 0.5, "No alloc data", ha="center", va="center", transform=ax_alloc.transAxes)
    ax_alloc.set_xlabel("Time [s]")
    ax_alloc.grid(True, alpha=0.2)

    ax_int = fig.add_subplot(gs[3, 1])
    if has_integral:
        ax_int.plot(df.t, df.sigma_x / 1000, label="sigma_x")
        ax_int.plot(df.t, df.sigma_y / 1000, label="sigma_y")
        ax_int.plot(df.t, df.sigma_psi / 1000, label="sigma_psi")
        ax_int.set_ylabel("Integral [kN / kNm]")
        ax_int.set_title("PID Integral Accumulator")
        ax_int.legend(loc="upper left", fontsize=7)
    else:
        ax_int.text(0.5, 0.5, "No integral data", ha="center", va="center", transform=ax_int.transAxes)
    ax_int.set_xlabel("Time [s]")
    ax_int.grid(True, alpha=0.2)

    ax_perr = fig.add_subplot(gs[3, 2])
    if "pos_err" in df.columns:
        ax_perr.plot(df.t, df.pos_err, "b-", linewidth=1.0)
        ax_perr.axhline(5.0, color="r", linestyle="--", alpha=0.5, label="Goal tol")
        ax_perr.set_ylabel("Distance [m]")
        ax_perr.set_title("Position Error Magnitude")
        ax_perr.legend(fontsize=7)
    else:
        pos_err = np.hypot(df.ex, df.ey)
        ax_perr.plot(df.t, pos_err, "b-", linewidth=1.0)
        ax_perr.set_ylabel("Distance [m]")
        ax_perr.set_title("Position Error Magnitude")
    ax_perr.set_xlabel("Time [s]")
    ax_perr.grid(True, alpha=0.2)

    # Shade disturbance on all time-series axes
    if has_dist and df.disturbance.any():
        dist_on = df.t[df.disturbance == 1]
        t_start, t_end = dist_on.iloc[0], dist_on.iloc[-1]
        for ax in [ax_pos, ax_hdg, ax_tau, ax_thr, ax_alloc, ax_int, ax_perr]:
            ax.axvspan(t_start, t_end, alpha=0.10, color="red")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {Path(out_path).resolve()}")

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    plot_dp_log(csv_path=csv_arg)
