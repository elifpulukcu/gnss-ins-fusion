"""
Check whether the large pure-INS drift is caused by our initial roll/pitch.

The normal pipeline starts the INS with:
    - position from our SPP solution
    - velocity = 0, because the robot is static at the beginning
    - yaw from the provided Heading column
    - roll/pitch from static accelerometer leveling
    - IMU biases from the initial static segment

After seeing large pure-INS drift, we wanted to make sure this was not simply
caused by a wrong roll/pitch convention. This script reruns the pure INS twice:

    1. operational
       Uses our normal initialization: roll/pitch from the static accelerometer.

    2. diagnostic_ref_rp
       Uses reference roll/pitch from the groundtruth file, but keeps everything
       else exactly the same.

The second mode is only a diagnostic check. It is not part of the final
operational pipeline. If reference roll/pitch had improved all runs strongly,
that would suggest a problem in our accelerometer-leveling convention. Since it
does not improve all runs consistently, the large pure-INS drift is more likely
caused by normal open-loop IMU error growth.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "imu"))

from coord_frames import R_ned_ecef, ecef_to_llh
from initial_state import load_groundtruth
from mechanization import (
    ACCEL_COLS,
    GYRO_COLS,
    C_b_e_from_initial,
    load_initial_state,
    mechanize_step,
)

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "imu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAMES = ["run2", "run3", "run4"]


def replace_initial_roll_pitch_with_reference(state, run_name):
    """Replace only the initial roll/pitch with the reference values.

    Everything else is intentionally kept the same:
        - same t0
        - same initial position
        - same initial velocity
        - same yaw source
        - same IMU bias estimates
        - same IMU data
        - same mechanization

    This lets us isolate the effect of the initial roll/pitch only.
    """
    gt_path = DATA_DIR / run_name / f"{run_name}_groundtruth.txt"
    gt = load_groundtruth(gt_path)

    gt = gt[gt["GPSTime"] >= state.t].reset_index(drop=True)
    first = gt.iloc[0]

    roll_ref = np.deg2rad(float(first["Roll"]))
    pitch_ref = np.deg2rad(float(first["Pitch"]))
    yaw_ref = np.deg2rad(float(first["Heading"]))

    lat, lon, _ = ecef_to_llh(*state.pos_ecef)

    state.C_b_e = C_b_e_from_initial(
        lat,
        lon,
        heading_rad=yaw_ref,
        pitch_rad=pitch_ref,
        roll_rad=roll_ref,
    )

    return state


def run_pure_ins(run_name, mode):
    """Run one pure-INS pass for one run and one initialization mode."""
    state = load_initial_state(run_name)

    if mode == "diagnostic_ref_rp":
        state = replace_initial_roll_pitch_with_reference(state, run_name)
    elif mode == "operational":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    imu_path = DATA_DIR / run_name / f"{run_name}_imu.txt"
    imu = pd.read_csv(imu_path, sep=r"\s+", engine="python")
    imu = imu[imu["Time"] >= state.t].reset_index(drop=True)

    n = len(imu)
    times = np.zeros(n)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))

    times[0] = state.t
    pos[0] = state.pos_ecef
    vel[0] = state.vel_ecef

    imu_times = imu["Time"].values
    gyros = imu[GYRO_COLS].values
    accels = imu[ACCEL_COLS].values

    for i in range(1, n):
        dt = float(imu_times[i] - imu_times[i - 1])

        state = mechanize_step(
            state,
            gyros[i],
            accels[i],
            dt,
        )

        times[i] = state.t
        pos[i] = state.pos_ecef
        vel[i] = state.vel_ecef

    return times, pos, vel


def compute_error_metrics(run_name, times, pos, vel):
    """Compare the pure-INS result with the reference trajectory."""
    gt_path = DATA_DIR / run_name / f"{run_name}_groundtruth.txt"
    gt = load_groundtruth(gt_path)

    gt_times = gt["GPSTime"].values

    # INS is at IMU rate, so interpolate it to the reference timestamps.
    pos_interp = np.zeros((len(gt_times), 3))
    vel_interp = np.zeros((len(gt_times), 3))

    for j in range(3):
        pos_interp[:, j] = np.interp(gt_times, times, pos[:, j])
        vel_interp[:, j] = np.interp(gt_times, times, vel[:, j])

    pos_ref = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].values
    vel_ref = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].values

    pos_err = pos_interp - pos_ref
    vel_err = vel_interp - vel_ref

    pos_err_mag = np.linalg.norm(pos_err, axis=1)
    vel_err_mag = np.linalg.norm(vel_err, axis=1)

    # Final error in local NED is useful when we want to inspect direction,
    # even though the report table only uses the magnitude.
    ref0 = pos_ref[0]
    ref_lat, ref_lon, _ = ecef_to_llh(*ref0)
    R = R_ned_ecef(ref_lat, ref_lon)
    final_err_ned = R @ (pos_interp[-1] - pos_ref[-1])

    return {
        "duration_s": float(gt_times[-1] - gt_times[0]),
        "final_pos_err_m": float(pos_err_mag[-1]),
        "max_pos_err_m": float(pos_err_mag.max()),
        "mean_pos_err_m": float(pos_err_mag.mean()),
        "final_vel_err_m_s": float(vel_err_mag[-1]),
        "max_vel_err_m_s": float(vel_err_mag.max()),
        "final_err_N_m": float(final_err_ned[0]),
        "final_err_E_m": float(final_err_ned[1]),
        "final_err_D_m": float(final_err_ned[2]),
    }


def main():
    rows = []

    for run_name in RUN_NAMES:
        for mode in ["operational", "diagnostic_ref_rp"]:
            times, pos, vel = run_pure_ins(run_name, mode)
            metrics = compute_error_metrics(run_name, times, pos, vel)

            rows.append({
                "run": run_name,
                "mode": mode,
                **metrics,
            })

    df = pd.DataFrame(rows)

    csv_path = OUT_DIR / "initial_attitude_sensitivity.csv"
    tex_path = OUT_DIR / "initial_attitude_sensitivity_table.tex"

    # Full output for debugging and possible extra analysis.
    df.to_csv(csv_path, index=False)

    # Compact table for Overleaf.
    compact = df[[
        "run",
        "mode",
        "duration_s",
        "final_pos_err_m",
        "max_pos_err_m",
        "mean_pos_err_m",
        "max_vel_err_m_s",
    ]].copy()

    compact.to_latex(
        tex_path,
        index=False,
        float_format="%.2f",
        caption=(
            "Sensitivity of pure INS drift to initial roll/pitch initialization. "
            "The diagnostic mode uses reference roll/pitch only for validation "
            "and is not part of the operational pipeline."
        ),
        label="tab:initial_attitude_sensitivity",
    )

    print(df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()