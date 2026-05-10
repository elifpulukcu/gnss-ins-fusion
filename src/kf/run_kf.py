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
from src.kf.plot_results import (load_ins_only, load_gnss, 
                                 compare_with_groundtruth, plot_errors,
                                 plot_trajectory, OUT_DIR, print_error_stats,
                                 plot_error_comparison)
from src.imu.initial_state import load_groundtruth
from src.utils.coord_frames import R_ned_ecef, ecef_to_llh


DATA_DIR = PROJECT_ROOT / "data"



  


def run_kf(run: int, outage_start_s: float | None = None, outage_duration_s: float = 0.0, r_scale: float = 1.0) -> pd.DataFrame:
    run_name = f"run{run}"

    # Initial nominal INS state and covariance
    state = load_initial_state(run_name)
    P0 = build_P0(run_name)
    
    lat0, lon0, _ = ecef_to_llh(*state.pos_ecef)
    R_ne = R_ned_ecef(lat0, lon0)   # ECEF -> NED

    pos_scale = r_scale

    R_pos_ned = np.diag([
        (10.0 * pos_scale)**2,
        (10.0 * pos_scale)**2,
        (25.0 * pos_scale)**2,
    ])
    R_pos_ecef = R_ne.T @ R_pos_ned @ R_ne

    R_vel_ned = np.diag([
        (0.5* pos_scale)**2,
        (0.5* pos_scale)**2,
        (0.8* pos_scale)**2,
    ])

    R_vel_ecef = R_ne.T @ R_vel_ned @ R_ne

    kf = ErrorStateKF(P0, R_pos=R_pos_ecef, R_vel=R_vel_ecef)

    # Load IMU and GNSS
    imu_path = DATA_DIR / run_name / f"{run_name}_imu.txt"
    imu = pd.read_csv(imu_path, sep=r"\s+", engine="python")
    imu = imu[imu["Time"] >= state.t].reset_index(drop=True)

    gnss = load_gnss(run)
    gnss_times = gnss["GPSTime"].values
    gnss_pos = gnss[["X", "Y", "Z"]].values
    gnss_vel = gnss[["VX", "VY", "VZ"]].values

    gnss_idx = 0
    results = []

    imu_times = imu["Time"].values
    gyros = imu[GYRO_COLS].values
    accels = imu[ACCEL_COLS].values
    
    outage_start_abs = None
    outage_end_abs = None

    if outage_start_s is not None:
        outage_start_abs = state.t + outage_start_s
        outage_end_abs = outage_start_abs + outage_duration_s

    for i in range(1, len(imu)):
        dt = float(imu_times[i] - imu_times[i - 1])

        # 1. KF prediction using current state and current accelerometer sample
        kf.predict(state, accels[i], dt)

        # 2. INS mechanization
        state = mechanize_step(state, gyros[i], accels[i], dt)


        in_outage = (
            outage_start_abs is not None
            and outage_start_abs <= state.t <= outage_end_abs
        )

        while gnss_idx < len(gnss_times) and gnss_times[gnss_idx] < state.t - 0.005:
            gnss_idx += 1

        if (not in_outage) and gnss_idx < len(gnss_times):
            if abs(gnss_times[gnss_idx] - state.t) <= 0.005:
                kf.update(state, gnss_pos[gnss_idx], gnss_vel[gnss_idx])
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
            "gnss_outage": in_outage
        })

    out = pd.DataFrame(results)
    out_path = OUT_DIR / f"kf_solution_run{run}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    return out





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=2)
    parser.add_argument("--outage-start", type=float, default=None,
                    help="Outage start time in seconds after run start")
    parser.add_argument("--outage-duration", type=float, default=0.0,
                    help="GNSS outage duration in seconds")
    parser.add_argument("--r-scale", type=float, default=1.0)
    args = parser.parse_args()

    kf_df = run_kf(
        args.run,
        outage_start_s=args.outage_start,
        outage_duration_s=args.outage_duration,
        r_scale = args.r_scale
    )
    print("Running KF solution for run:", args.run, "and scale:", args.r_scale)
    cmp = compare_with_groundtruth(args.run, kf_df)
    cmp["outage_start_s"] = args.outage_start
    cmp["outage_duration_s"] = args.outage_duration

    plot_trajectory(args.run, cmp, args.r_scale)
    plot_errors(args.run, cmp, args.r_scale)

    print_error_stats(args.run, cmp)
    plot_error_comparison(args.run, cmp, args.r_scale)


if __name__ == "__main__":
    main()