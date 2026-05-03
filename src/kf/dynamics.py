"""
Continuous-time F matrix for the 15-state ECEF error-state Kalman filter.

State order:
    dx = [
        dp_e,      # position error, ECEF
        dv_e,      # velocity error, ECEF
        dpsi_e,    # attitude error, ECEF
        dba,       # accelerometer bias error
        dbg,       # gyro bias error
    ]

All attitude, velocity and position error terms are expressed in ECEF.

The bias terms are modeled as random walks. Their driving noise is handled
in Q, so the bias rows in F stay zero.

This follows the standard ECEF error dynamics in Groves, eq. 14.55.
We are not including the gravity-gradient term here. For the short runs and
consumer-grade IMU biases we are targeting, that coupling is well below the
noise level and does not move the filter in practice.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from coord_frames import OMEGA_IE_E, skew  # noqa: E402


STATE_DIM = 15

P_SLICE = slice(0, 3)
V_SLICE = slice(3, 6)
PSI_SLICE = slice(6, 9)
BA_SLICE = slice(9, 12)
BG_SLICE = slice(12, 15)


def build_F(C_b_e: np.ndarray, f_b: np.ndarray) -> np.ndarray:
    """Build the continuous-time error dynamics matrix.

    Args:
        C_b_e: Body-to-ECEF direction cosine matrix, shape (3, 3).
        f_b: Bias-corrected specific force in the body frame, shape (3,).

    Returns:
        Continuous-time F matrix with shape (15, 15).
    """
    f_e = C_b_e @ f_b
    omega_ie_x = skew(OMEGA_IE_E)

    F = np.zeros((STATE_DIM, STATE_DIM))

    # Position error is driven directly by velocity error.
    F[P_SLICE, V_SLICE] = np.eye(3)

    # Velocity error dynamics.
    #
    # Coriolis term:
    #     -2 * [omega_ie_e x] * dv_e
    #
    # Attitude error projects specific force into velocity error:
    #     -[f_e x] * dpsi_e
    #
    # Accelerometer bias error enters through the current attitude:
    #     -C_b_e * dba
    F[V_SLICE, V_SLICE] = -2.0 * omega_ie_x
    F[V_SLICE, PSI_SLICE] = -skew(f_e)
    F[V_SLICE, BA_SLICE] = -C_b_e

    # Attitude error dynamics.
    #
    # Earth rotation affects the ECEF attitude error, and gyro bias error
    # enters through the body-to-ECEF attitude matrix.
    F[PSI_SLICE, PSI_SLICE] = -omega_ie_x
    F[PSI_SLICE, BG_SLICE] = -C_b_e

    return F


def discretize(F: np.ndarray, dt: float) -> np.ndarray:
    """Convert continuous-time F to a discrete transition matrix.

    This uses the first-order approximation:

        Phi = I + F * dt

    At the nominal IMU rate, 100 Hz in our current setup, this is enough for
    the propagation step. If we start using larger dt values, switch this to
    scipy.linalg.expm(F * dt).
    """
    return np.eye(STATE_DIM) + F * dt


if __name__ == "__main__":
    g = 9.81
    f_b = np.array([0.0, 0.0, g])

    F = build_F(np.eye(3), f_b)

    assert F.shape == (STATE_DIM, STATE_DIM)
    assert np.allclose(F[P_SLICE, V_SLICE], np.eye(3))
    assert np.allclose(F[V_SLICE, BA_SLICE], -np.eye(3))
    assert np.allclose(F[PSI_SLICE, BG_SLICE], -np.eye(3))
    assert np.allclose(F[V_SLICE, PSI_SLICE], -skew(f_b))
    assert np.allclose(F[P_SLICE, P_SLICE], 0.0)
    assert np.allclose(F[BA_SLICE, :], 0.0)
    assert np.allclose(F[BG_SLICE, :], 0.0)

    Phi = discretize(F, 0.01)

    assert Phi.shape == (STATE_DIM, STATE_DIM)
    assert np.allclose(Phi[P_SLICE, V_SLICE], 0.01 * np.eye(3))
    assert np.allclose(np.diag(Phi)[BA_SLICE], 1.0)
    assert np.allclose(np.diag(Phi)[BG_SLICE], 1.0)

    print("dynamics.py self-tests passed.")