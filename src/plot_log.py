import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys

def find_latest_log():
    pattern = str(Path(__file__).parent / "runtime" / "dp_log_*.csv")
    logs = sorted(glob.glob(pattern))
    if logs:
        return logs[-1]
    fallback = str(Path(__file__).parent / "runtime" / "dp_log.csv")
    if Path(fallback).exists():
        return fallback
    return None

def plot_dp_log(csv_path=None, out_path=None):
    if csv_path is None:
        csv_path = find_latest_log()
        if csv_path is None:
            print("No log file found.")
            return

    if out_path is None:
        out_path = Path(csv_path).stem + "_plot.png"

    df = pd.read_csv(csv_path)
    has_dist = "disturbance" in df.columns

    fig, axs = plt.subplots(4, 2, figsize=(16, 12), gridspec_kw={"width_ratios": [2, 1]})

    # Left column: time-series plots
    # Position tracking
    axs[0, 0].plot(df.t, df.x, label="x")
    axs[0, 0].plot(df.t, df.xr, "--", label="x_ref")
    axs[0, 0].plot(df.t, df.y, label="y")
    axs[0, 0].plot(df.t, df.yr, "--", label="y_ref")
    axs[0, 0].set_ylabel("Position [m]")
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set_title("Position Tracking")

    # Heading tracking
    psi_unwrap = np.unwrap(df.psi)
    psir_unwrap = np.unwrap(df.psir)
    axs[1, 0].plot(df.t, np.degrees(psi_unwrap), label="psi")
    axs[1, 0].plot(df.t, np.degrees(psir_unwrap), "--", label="psi_ref")
    axs[1, 0].set_ylabel("Heading [deg]")
    axs[1, 0].legend(loc="upper left")
    axs[1, 0].set_title("Heading Tracking")

    # Control effort
    axs[2, 0].plot(df.t, df.tau_x / 1000, label="tau_x")
    axs[2, 0].plot(df.t, df.tau_y / 1000, label="tau_y")
    axs[2, 0].plot(df.t, df.tau_psi / 1000, label="tau_psi")
    axs[2, 0].set_ylabel("Control [kN / kNm]")
    axs[2, 0].legend(loc="upper left")
    axs[2, 0].set_title("Commanded Generalized Forces")

    # Thruster forces
    axs[3, 0].plot(df.t, df.Fx1 / 1000, label="Fx1")
    axs[3, 0].plot(df.t, df.Fy1 / 1000, label="Fy1")
    axs[3, 0].plot(df.t, df.Fx2 / 1000, label="Fx2")
    axs[3, 0].plot(df.t, df.Fy2 / 1000, label="Fy2")
    axs[3, 0].set_ylabel("Thruster Force [kN]")
    axs[3, 0].set_xlabel("Time [s]")
    axs[3, 0].legend(loc="upper left")
    axs[3, 0].set_title("Individual Thruster Forces")

    # Shade disturbance window on all left-column plots
    if has_dist and df.disturbance.any():
        dist_on = df.t[df.disturbance == 1]
        t_start, t_end = dist_on.iloc[0], dist_on.iloc[-1]
        for row in range(4):
            axs[row, 0].axvspan(t_start, t_end, alpha=0.15, color="red", label="Disturbance" if row == 0 else None)
        axs[0, 0].legend(loc="upper left")

    # Right column: trajectory plot (span all 4 rows)
    for row in range(4):
        axs[row, 1].set_visible(False)
    ax_traj = fig.add_subplot(1, 2, 2)

    ax_traj.plot(df.x, df.y, "b-", linewidth=1.2, label="Actual path", alpha=0.8)
    ax_traj.plot(df.xr, df.yr, "g--", linewidth=1.0, label="Reference path", alpha=0.7)
    ax_traj.plot(df.x.iloc[0], df.y.iloc[0], "bs", markersize=10, label="Start")
    ax_traj.plot(df.x.iloc[-1], df.y.iloc[-1], "r^", markersize=10, label="End")

    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.set_title("2D Trajectory")
    ax_traj.set_aspect("equal")
    ax_traj.legend(loc="best")
    ax_traj.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(out_path)
    plt.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    plot_dp_log(csv_path=csv_arg)
