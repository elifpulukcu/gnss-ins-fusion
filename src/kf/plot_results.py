
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "kf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.kf.filter import ErrorStateKF
from src.imu.initial_state import load_groundtruth
from src.utils.coord_frames import R_ned_ecef, ecef_to_llh

def load_ins_only(run: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "output" / "imu" / f"ins_only_run{run}.csv"
    return pd.read_csv(path)




def load_gnss(run: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "output" / "gnss" / f"SPP_solutions_run{run}.csv"
    gnss = pd.read_csv(path)
    gnss = gnss.sort_values("GPSTime").reset_index(drop=True)

    # Compute GNSS velocity from SPP position differences
    # t = gnss["GPSTime"].values
    # pos = gnss[["X", "Y", "Z"]].values
    
    # # vel = np.zeros_like(pos)

    # # Central differences for internal points
    # dt = t[2:] - t[:-2]
    # vel[1:-1] = (pos[2:] - pos[:-2]) / dt[:, None]

    # # Forward/backward difference for endpoints
    # vel[0] = (pos[1] - pos[0]) / (t[1] - t[0])
    # vel[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])

    # gnss["VX"] = vel[:, 0]
    # gnss["VY"] = vel[:, 1]
    # gnss["VZ"] = vel[:, 2]
    
    return gnss.sort_values("GPSTime").reset_index(drop=True)


def compare_with_groundtruth(run: int, kf_df: pd.DataFrame) -> dict:
    run_name = f"run{run}"
    gt_path = DATA_DIR / run_name / f"{run_name}_groundtruth.txt"
    gt = load_groundtruth(gt_path)

    gt_times = gt["GPSTime"].values
    pos_gt = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].values
    vel_gt = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].values

    # Load GNSS SPP and pure INS
    gnss = load_gnss(run)
    ins = load_ins_only(run)

    def interp_position(df, time_col, cols):
        out = np.zeros((len(gt_times), 3))
        for j, col in enumerate(cols):
            out[:, j] = np.interp(gt_times, df[time_col], df[col])
        return out

    pos_kf = interp_position(
        kf_df,
        "GPSTime",
        ["X-ECEF", "Y-ECEF", "Z-ECEF"],
    )

    pos_ins = interp_position(
        ins,
        "GPSTime",
        ["X-ECEF", "Y-ECEF", "Z-ECEF"],
    )

    pos_gnss = interp_position(
        gnss,
        "GPSTime",
        ["X", "Y", "Z"],
    )

    # Convert everything to local NED for plotting
    ref = pos_gt[0]
    ref_lat, ref_lon, _ = ecef_to_llh(*ref)
    R = R_ned_ecef(ref_lat, ref_lon)

    pos_gt_ned = (R @ (pos_gt - ref).T).T
    pos_kf_ned = (R @ (pos_kf - ref).T).T
    pos_ins_ned = (R @ (pos_ins - ref).T).T
    pos_gnss_ned = (R @ (pos_gnss - ref).T).T

    def interp_velocity(df, time_col, cols):
        out = np.zeros((len(gt_times), 3))
        for j, col in enumerate(cols):
            out[:, j] = np.interp(gt_times, df[time_col], df[col])
        return out

    vel_kf = interp_velocity(
        kf_df,
        "GPSTime",
        ["VX-ECEF", "VY-ECEF", "VZ-ECEF"],
    )

    pos_err = pos_kf - pos_gt
    vel_err = vel_kf - vel_gt
    gnss_err = pos_gnss - pos_gt
    ins_err = pos_ins - pos_gt
    gnss_err_ned = pos_gnss_ned - pos_gt_ned
    kf_err_ned = pos_kf_ned - pos_gt_ned
    ins_err_ned = pos_ins_ned - pos_gt_ned

    return {
        "times": gt_times,
        "pos_gt_ned": pos_gt_ned,
        "pos_kf_ned": pos_kf_ned,
        "pos_ins_ned": pos_ins_ned,
        "pos_gnss_ned": pos_gnss_ned,
        "pos_err": pos_err,
        "vel_err": vel_err,
        "pos_err_mag": np.linalg.norm(pos_err, axis=1),
        "vel_err_mag": np.linalg.norm(vel_err, axis=1),
        "gnss_err_ned": gnss_err_ned,
        "kf_err_ned": kf_err_ned,
        "ins_err_ned": ins_err_ned,
        "gnss_err_mag": np.linalg.norm(gnss_err, axis=1),
        "ins_err_mag": np.linalg.norm(ins_err, axis=1)
    }



def plot_errors(run: int, cmp: dict):
    t = cmp["times"] - cmp["times"][0]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, cmp["pos_err_mag"], label="|position error|")
    axes[0].set_ylabel("Position error [m]")
    axes[0].set_title(f"Run {run}: KF position error")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, cmp["vel_err_mag"], label="|velocity error|")
    axes[1].set_ylabel("Velocity error [m/s]")
    axes[1].set_xlabel("Time since start [s]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if "outage_start_s" in cmp and cmp["outage_start_s"] is not None:
        axes[0].axvspan(
            cmp["outage_start_s"],
            cmp["outage_start_s"] + cmp["outage_duration_s"],
            alpha=0.2,
            label="GNSS outage",
        )
        axes[1].axvspan(
            cmp["outage_start_s"],
            cmp["outage_start_s"] + cmp["outage_duration_s"],
            alpha=0.2,
        )
    fig.tight_layout()
    out_path = FIG_DIR / f"kf_errors_run{run}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_trajectory(run: int, cmp: dict):
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(
        cmp["pos_gt_ned"][:, 1],
        cmp["pos_gt_ned"][:, 0],
        label="Ground truth",
        linewidth=2.0,
    )

    ax.plot(
        cmp["pos_gnss_ned"][:, 1],
        cmp["pos_gnss_ned"][:, 0],
        label="GNSS SPP",
        linewidth=1.0,
        alpha=0.8,
    )

    # ax.plot(
    #     cmp["pos_ins_ned"][:, 1],
    #     cmp["pos_ins_ned"][:, 0],
    #     label="Pure INS",
    #     linewidth=1.0,
    #     alpha=0.8,
    # )

    ax.plot(
        cmp["pos_kf_ned"][:, 1],
        cmp["pos_kf_ned"][:, 0],
        label="GNSS/INS KF",
        linewidth=1.5,
    )

    ax.scatter([0], [0], label="Start", s=45)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(f"Run {run}: Ground Truth vs GNSS vs INS vs KF")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_path = FIG_DIR / f"trajectory_comparison_run{run}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")
    
    
def print_error_stats(run: int, cmp: dict):
    def stats(name, err_ned):
        horiz = np.linalg.norm(err_ned[:, 0:2], axis=1)
        total = np.linalg.norm(err_ned, axis=1)

        print(f"\n{name}")
        print(f"  mean NED error [m]: {err_ned.mean(axis=0)}")
        print(f"  horizontal RMSE [m]: {np.sqrt(np.mean(horiz**2)):.3f}")
        print(f"  3D RMSE [m]: {np.sqrt(np.mean(total**2)):.3f}")
        print(f"  max 3D error [m]: {total.max():.3f}")
        print(f"  final 3D error [m]: {total[-1]:.3f}")

    print(f"\n=== Run {run} error statistics ===")
    stats("GNSS SPP", cmp["gnss_err_ned"])
    stats("Pure INS", cmp["ins_err_ned"])
    stats("GNSS/INS KF", cmp["kf_err_ned"])
    
    
def plot_error_comparison(run: int, cmp: dict):
    t = cmp["times"] - cmp["times"][0]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t, cmp["gnss_err_mag"], label="GNSS SPP")
    ax.plot(t, cmp["pos_err_mag"], label="GNSS/INS KF")
    ax.plot(t, cmp["ins_err_mag"], label="Pure INS", alpha=0.7)

    if "outage_start_s" in cmp and cmp["outage_start_s"] is not None:
        ax.axvspan(
            cmp["outage_start_s"],
            cmp["outage_start_s"] + cmp["outage_duration_s"],
            alpha=0.2,
            label="GNSS outage",
        )

    ax.set_xlabel("Time since start [s]")
    ax.set_ylabel("3D position error [m]")
    ax.set_title(f"Run {run}: Position error comparison")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = FIG_DIR / f"position_error_comparison_run{run}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")