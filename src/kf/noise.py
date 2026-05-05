"""
Discrete process noise Q for the 15-state error-state KF.

For each IMU step, we fill four diagonal blocks:

    velocity error   : accel white noise
    attitude error   : gyro white noise
    accel bias error : accel bias random walk
    gyro bias error  : gyro bias random walk

The white-noise terms come from the Allan results in:

    output/imu/imu_calibration.json

For each axis, we use the sigma_at_1s value from the calibration file.

The bias random-walk terms are a first tuning point. We use the stored
bias_instability values for now because we have not fitted the random-walk
slopes from the Allan curve yet. Once GNSS updates are connected, these
terms should be checked with innovation plots and consistency tests.

See IEEE Std 952-1997, Annex C, for the Allan-noise parameter mapping.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "kf"))

from dynamics import (  # noqa: E402
    BA_SLICE,
    BG_SLICE,
    PSI_SLICE,
    STATE_DIM,
    V_SLICE,
)


CALIB_PATH = REPO_ROOT / "output" / "imu" / "imu_calibration.json"

GYRO_AXES = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
ACCEL_AXES = ["Accel_X", "Accel_Y", "Accel_Z"]

DEG_TO_RAD2 = (np.pi / 180.0) ** 2


def _load_allan_params(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data["allan_noise_params"]


def _white_noise_psd(allan: dict, axes: list[str]) -> np.ndarray:
    """Return per-axis white-noise PSD from sigma_at_1s."""
    return np.array([allan[axis]["sigma_at_1s"]**2 for axis in axes])


def _bias_rw_psd(allan: dict, axes: list[str]) -> np.ndarray:
    """Return the per-axis bias random-walk PSD used by the filter.

    This currently uses bias_instability as a practical starting value.
    It is intentionally a bit conservative so the filter can move the bias
    states during early GNSS updates.
    """
    return np.array([allan[axis]["bias_instability"]**2 for axis in axes])


def build_Q(dt: float, calibration_path: Path = CALIB_PATH) -> np.ndarray:
    """Build the discrete process-noise covariance for one IMU step.

    Args:
        dt: IMU sample interval in seconds.
        calibration_path: Path to imu_calibration.json.

    Returns:
        A 15x15 discrete-time Q matrix.

    Notes:
        The position block is left at zero. Position uncertainty grows
        through the position-velocity coupling in F and Phi.
    """
    allan = _load_allan_params(calibration_path)

    accel_white_psd = _white_noise_psd(allan, ACCEL_AXES)
    gyro_white_psd = _white_noise_psd(allan, GYRO_AXES) * DEG_TO_RAD2

    accel_bias_rw_psd = _bias_rw_psd(allan, ACCEL_AXES)
    gyro_bias_rw_psd = _bias_rw_psd(allan, GYRO_AXES) * DEG_TO_RAD2

    Q = np.zeros((STATE_DIM, STATE_DIM))

    Q[V_SLICE, V_SLICE] = np.diag(accel_white_psd) * dt
    Q[PSI_SLICE, PSI_SLICE] = np.diag(gyro_white_psd) * dt
    Q[BA_SLICE, BA_SLICE] = np.diag(accel_bias_rw_psd) * dt
    Q[BG_SLICE, BG_SLICE] = np.diag(gyro_bias_rw_psd) * dt

    return Q


if __name__ == "__main__":
    Q = build_Q(0.01)

    assert Q.shape == (STATE_DIM, STATE_DIM)
    assert np.allclose(Q[:3, :], 0.0), "position rows should stay zero"

    assert np.all(np.diag(Q[V_SLICE, V_SLICE]) > 0.0)
    assert np.all(np.diag(Q[PSI_SLICE, PSI_SLICE]) > 0.0)
    assert np.all(np.diag(Q[BA_SLICE, BA_SLICE]) > 0.0)
    assert np.all(np.diag(Q[BG_SLICE, BG_SLICE]) > 0.0)

    print("Q sigma per IMU step:")
    print(f"  velocity   [m/s]    : {np.sqrt(np.diag(Q[V_SLICE, V_SLICE]))}")
    print(f"  attitude   [rad]    : {np.sqrt(np.diag(Q[PSI_SLICE, PSI_SLICE]))}")
    print(f"  accel bias [m/s^2]  : {np.sqrt(np.diag(Q[BA_SLICE, BA_SLICE]))}")
    print(f"  gyro bias  [rad/s]  : {np.sqrt(np.diag(Q[BG_SLICE, BG_SLICE]))}")

    print("\nnoise.py self-tests passed.")