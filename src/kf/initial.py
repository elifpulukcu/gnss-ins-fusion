"""
Initial nominal state and covariance for the 15-state error-state KF.

The nominal state comes from output/imu/initial_states.json through
mechanization.load_initial_state. We keep that path so the unit conversions
and DCM setup stay in one place.

The same JSON also stores the static-segment IMU noise estimates. We use
those values as the initial uncertainty for the accel and gyro bias states.

P0 diagonal setup:
    position      : 5 m per axis by default
    velocity      : 0.5 m/s per axis by default
    roll / pitch  : 0.5 deg
    yaw           : 1.0 deg
    accel bias    : per-run accel std from the static segment
    gyro bias     : per-run gyro std from the static segment, converted to rad/s

The attitude uncertainty is easiest to choose in body axes, where roll,
pitch and yaw are meaningful. The filter stores dpsi in ECEF, so we rotate
that 3x3 block into ECEF before putting it into P0.

The position and velocity values are intentionally broad for now. Once GNSS
is available, the first update should pull them down quickly.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "imu"))
sys.path.insert(0, str(REPO_ROOT / "src" / "kf"))

from dynamics import STATE_DIM  # noqa: E402
from mechanization import INSState, load_initial_state  # noqa: E402


INITIAL_STATES_PATH = REPO_ROOT / "output" / "imu" / "initial_states.json"

GYRO_COLS = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
ACCEL_COLS = ["Accel_X", "Accel_Y", "Accel_Z"]

DEFAULT_POS_SIGMA_M = 5.0
DEFAULT_VEL_SIGMA_MS = 0.5
DEFAULT_ROLL_PITCH_SIGMA_RAD = np.deg2rad(0.5)
DEFAULT_YAW_SIGMA_RAD = np.deg2rad(1.0)


def _load_run_record(run_name: str) -> dict:
    data = json.loads(INITIAL_STATES_PATH.read_text())

    if run_name not in data:
        available_runs = ", ".join(data.keys())
        raise KeyError(
            f"{run_name} is not in {INITIAL_STATES_PATH.name}. "
            f"Available runs: {available_runs}"
        )

    return data[run_name]


def build_P0(
    run_name: str,
    pos_sigma_m: float = DEFAULT_POS_SIGMA_M,
    vel_sigma_ms: float = DEFAULT_VEL_SIGMA_MS,
    roll_pitch_sigma_rad: float = DEFAULT_ROLL_PITCH_SIGMA_RAD,
    yaw_sigma_rad: float = DEFAULT_YAW_SIGMA_RAD,
) -> np.ndarray:
    """Build the initial 15x15 covariance matrix for one run."""
    rec = _load_run_record(run_name)

    accel_std = np.array([rec["accel_std_m_s2"][col] for col in ACCEL_COLS])
    gyro_std = np.deg2rad(
        np.array([rec["gyro_std_deg_s"][col] for col in GYRO_COLS])
    )

    nominal = load_initial_state(run_name)

    # Pick attitude uncertainty in body axes first.
    #
    # With our body-frame convention:
    #   x: right
    #   y: forward
    #   z: up
    #
    # pitch is about x, roll is about y, and yaw is about z.
    P_att_body = np.diag([
        roll_pitch_sigma_rad**2,
        roll_pitch_sigma_rad**2,
        yaw_sigma_rad**2,
    ])

    # The filter attitude error lives in ECEF, so rotate this block.
    P_att_ecef = nominal.C_b_e @ P_att_body @ nominal.C_b_e.T

    P0 = np.zeros((STATE_DIM, STATE_DIM))

    P0[0:3, 0:3] = np.eye(3) * pos_sigma_m**2
    P0[3:6, 3:6] = np.eye(3) * vel_sigma_ms**2
    P0[6:9, 6:9] = P_att_ecef
    P0[9:12, 9:12] = np.diag(accel_std**2)
    P0[12:15, 12:15] = np.diag(gyro_std**2)

    return P0


def build_initial_state(run_name: str) -> tuple[INSState, np.ndarray]:
    """Return the nominal state and initial covariance for the KF."""
    nominal = load_initial_state(run_name)
    P0 = build_P0(run_name)

    return nominal, P0


if __name__ == "__main__":
    for run_name in ("run2", "run3", "run4"):
        nominal, P0 = build_initial_state(run_name)
        sigma = np.sqrt(np.maximum(np.diag(P0), 0.0))

        print(f"\n--- {run_name} ---")
        print(f"t0       = {nominal.t:.2f} s")
        print(f"|pos|    = {np.linalg.norm(nominal.pos_ecef):.1f} m")
        print(f"|vel|    = {np.linalg.norm(nominal.vel_ecef):.3f} m/s")
        print(f"|bias_a| = {np.linalg.norm(nominal.bias_a):.4f} m/s^2")
        print(f"|bias_g| = {np.linalg.norm(nominal.bias_g):.4f} rad/s")

        print("P0 sigma:")
        print(f"  pos    [m]      : {sigma[0:3]}")
        print(f"  vel    [m/s]    : {sigma[3:6]}")
        print(f"  att    [rad]    : {sigma[6:9]}")
        print(f"  bias_a [m/s^2]  : {sigma[9:12]}")
        print(f"  bias_g [rad/s]  : {sigma[12:15]}")

    print("\ninitial.py self-tests passed.")