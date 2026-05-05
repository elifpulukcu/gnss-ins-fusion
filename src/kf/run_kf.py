import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.imu.mechanization import (
    load_initial_state,
    mechanize_step,
    GYRO_COLS,
    ACCEL_COLS,
)
from src.kf.initial import build_P0
from src.kf.filter import ErrorStateKF
from src.imu.initial_state import load_groundtruth
from src.utils.coord_frames import R_ned_ecef, ecef_to_llh


DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "kf"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_ins_only(run: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "output" / "imu" / f"ins_only_run{run}.csv"
    return pd.read_csv(path)




def load_gnss(run: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "output" / "gnss" / f"SPP_solutions_run{run}.csv"
    gnss = pd.read_csv(path)
    return gnss.sort_values("GPSTime").reset_index(drop=True)


def run_kf(run: int) -> pd.DataFrame:
    run_name = f"run{run}"

    # Initial nominal INS state and covariance
    state = load_initial_state(run_name)
    P0 = build_P0(run_name)

    # Start with position-only GNSS update
    kf = ErrorStateKF(P0)

    # Load IMU and GNSS
    imu_path = DATA_DIR / run_name / f"{run_name}_imu.txt"
    imu = pd.read_csv(imu_path, sep=r"\s+", engine="python")
    imu = imu[imu["Time"] >= state.t].reset_index(drop=True)

    gnss = load_gnss(run)
    gnss_times = gnss["GPSTime"].values
    gnss_pos = gnss[["X", "Y", "Z"]].values

    gnss_idx = 0
    results = []

    imu_times = imu["Time"].values
    gyros = imu[GYRO_COLS].values
    accels = imu[ACCEL_COLS].values

    for i in range(1, len(imu)):
        dt = float(imu_times[i] - imu_times[i - 1])

        # 1. KF prediction using current state and current accelerometer sample
        kf.predict(state, accels[i], dt)

        # 2. INS mechanization
        state = mechanize_step(state, gyros[i], accels[i], dt)

        # 3. GNSS update if available close to current IMU time
        while gnss_idx < len(gnss_times) and gnss_times[gnss_idx] < state.t - 0.005:
            gnss_idx += 1

        if gnss_idx < len(gnss_times):
            if abs(gnss_times[gnss_idx] - state.t) <= 0.005:
                kf.update(state, gnss_pos[gnss_idx])
                gnss_used = True
                gnss_idx += 1
            else:
                gnss_used = False
        else:
            gnss_used = False

        results.append({
            "GPSTime": state.t,
            "X-ECEF": state.pos_ecef[0],
            "Y-ECEF": state.pos_ecef[1],
            "Z-ECEF": state.pos_ecef[2],
            "VX-ECEF": state.vel_ecef[0],
            "VY-ECEF": state.vel_ecef[1],
            "VZ-ECEF": state.vel_ecef[2],
            "bias_ax": state.bias_a[0],
            "bias_ay": state.bias_a[1],
            "bias_az": state.bias_a[2],
            "bias_gx": state.bias_g[0],
            "bias_gy": state.bias_g[1],
            "bias_gz": state.bias_g[2],
            "gnss_used": gnss_used,
        })

    out = pd.DataFrame(results)
    out_path = OUT_DIR / f"kf_solution_run{run}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    return out



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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=2)
    args = parser.parse_args()

    kf_df = run_kf(args.run)
    cmp = compare_with_groundtruth(args.run, kf_df)

    plot_trajectory(args.run, cmp)
    plot_errors(args.run, cmp)

    print("Final position error:", cmp["pos_err_mag"][-1], "m")
    print("Max position error:", cmp["pos_err_mag"].max(), "m")
    print("Max velocity error:", cmp["vel_err_mag"].max(), "m/s")


if __name__ == "__main__":
    main()