import pandas as pd
import numpy as np

CLIGHT = 299792458.0
FREQ_L1 = 1575.42e6
LAMBDA_L1 = CLIGHT / FREQ_L1

def get_sat_pos_vel(epoch, tau, svpos, dt_s=1.0):
    """
    Approximate satellite velocity by finite differencing SP3 positions.
    """
    epoch_minus = epoch - pd.Timedelta(seconds=dt_s)
    epoch_plus = epoch + pd.Timedelta(seconds=dt_s)

    sat_now = svpos.getSvPos(epoch, tau)
    sat_minus = svpos.getSvPos(epoch_minus, tau)
    sat_plus = svpos.getSvPos(epoch_plus, tau)

    common = sat_now.index.intersection(sat_minus.index).intersection(sat_plus.index)

    sat_now = sat_now.loc[common]
    sat_minus = sat_minus.loc[common]
    sat_plus = sat_plus.loc[common]

    pos = sat_now.iloc[:, :3]
    vel = (sat_plus.iloc[:, :3].values - sat_minus.iloc[:, :3].values) / (2 * dt_s)

    vel = pd.DataFrame(
        vel,
        index=common,
        columns=["VX", "VY", "VZ"],
    )

    return pos, vel


def doppler_velocity_solution(receiver_pos_ecef, satpos, satvel, doppler_hz):
    """
    Estimate receiver ECEF velocity from Doppler.

    Args:
        receiver_pos_ecef: array-like, shape (3,)
            Receiver position from SPP.
        satpos: DataFrame, shape (n, 3)
            Satellite ECEF positions.
        satvel: DataFrame, shape (n, 3)
            Satellite ECEF velocities.
        doppler_hz: Series or array, shape (n,)
            Doppler observations in Hz.

    Returns:
        pd.Series with VX, VY, VZ, cdt_dot
    """
    receiver_pos_ecef = np.asarray(receiver_pos_ecef)

    sat_xyz = satpos.values
    sat_v = satvel.values

    los = sat_xyz - receiver_pos_ecef
    ranges = np.linalg.norm(los, axis=1)
    e = los / ranges[:, None]

    # Doppler to pseudorange rate [m/s]
    rho_dot_obs = -np.asarray(doppler_hz) * LAMBDA_L1

    # Measurement equation:
    # rho_dot_obs - e dot v_sat = -e dot v_rec + cdt_dot
    L = rho_dot_obs - np.sum(e * sat_v, axis=1)

    A = np.zeros((len(satpos), 4))
    A[:, 0:3] = -e
    A[:, 3] = 1.0

    if np.linalg.matrix_rank(A) < 4:
        return None

    x, *_ = np.linalg.lstsq(A, L, rcond=None)

    return pd.Series(
        x,
        index=["VX", "VY", "VZ", "cdt_dot"],
    )