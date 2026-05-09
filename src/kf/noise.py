"""
Process noise matrix for the 15-state error-state Kalman filter.

This file only builds the discrete Q matrix used during IMU propagation.
The PSD values below are the ones provided in the course email, not values
estimated by our own Allan analysis. We still keep the Allan analysis in the
report as a sanity check, but for the final filter tuning we follow the
provided reference values.

Our error-state ordering is defined in dynamics.py as:

    position error      : 0:3
    velocity error      : 3:6
    attitude error      : 6:9
    accelerometer bias  : 9:12
    gyroscope bias      : 12:15

The email snippet used a different ordering, with attitude first. Because of
that, the numbers are the same as in the email, but the blocks are placed
according to our own state vector.

For each IMU time step dt, we add noise to:
    - velocity error, from accelerometer white noise
    - attitude error, from gyroscope white noise
    - accelerometer bias, from accelerometer bias random walk
    - gyroscope bias, from gyroscope bias random walk

The position block is left as zero on purpose. Position uncertainty still
grows through the state-transition matrix, because position is coupled to
velocity in F/Phi.
"""

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

# PSD values from the course email.
# Units are kept exactly as provided.
ACCEL_NOISE_PSD = 3.462133832010000e-07   # m^2/s^3
ACCEL_BIAS_PSD  = 2.163833645006250e-08   # m^2/s^5
GYRO_NOISE_PSD  = 3.046174197867087e-08   # rad^2/s^3
GYRO_BIAS_PSD   = 2.350443053909789e-09   # rad^2/s^5


def build_Q(dt: float) -> np.ndarray:
    """Build the discrete process-noise covariance for one IMU step."""
    Q = np.zeros((STATE_DIM, STATE_DIM))

    # Accelerometer white noise mainly enters the velocity-error states.
    Q[V_SLICE, V_SLICE] = np.eye(3) * ACCEL_NOISE_PSD * dt

    # Gyroscope white noise mainly enters the attitude-error states.
    Q[PSI_SLICE, PSI_SLICE] = np.eye(3) * GYRO_NOISE_PSD * dt

    # Biases are modelled as random walks, so their uncertainty grows with dt.
    Q[BA_SLICE, BA_SLICE] = np.eye(3) * ACCEL_BIAS_PSD * dt
    Q[BG_SLICE, BG_SLICE] = np.eye(3) * GYRO_BIAS_PSD * dt

    return Q


if __name__ == "__main__":
    # Small sanity check for the nominal 100 Hz IMU rate.
    Q = build_Q(0.01)

    assert Q.shape == (STATE_DIM, STATE_DIM)

    # We do not inject noise directly into position. It grows via velocity.
    assert np.allclose(Q[:3, :], 0.0), "position rows should stay zero"

    assert np.all(np.diag(Q[V_SLICE, V_SLICE]) > 0.0)
    assert np.all(np.diag(Q[PSI_SLICE, PSI_SLICE]) > 0.0)
    assert np.all(np.diag(Q[BA_SLICE, BA_SLICE]) > 0.0)
    assert np.all(np.diag(Q[BG_SLICE, BG_SLICE]) > 0.0)

    print("Q sigma per IMU step (dt=0.01 s):")
    print(f"  velocity error      [m/s]   : {np.sqrt(np.diag(Q[V_SLICE, V_SLICE]))}")
    print(f"  attitude error      [rad]   : {np.sqrt(np.diag(Q[PSI_SLICE, PSI_SLICE]))}")
    print(f"  accel bias error    [m/s^2] : {np.sqrt(np.diag(Q[BA_SLICE, BA_SLICE]))}")
    print(f"  gyro bias error     [rad/s] : {np.sqrt(np.diag(Q[BG_SLICE, BG_SLICE]))}")

    print("\nnoise.py self-check passed.")