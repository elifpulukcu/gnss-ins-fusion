"""
INS mechanization in the ECEF frame.

State at each IMU step, 100 Hz for this dataset:
    pos_ecef : (3,)   metres
    vel_ecef : (3,)   m/s
    C_b_e    : (3,3)  DCM body -> ECEF, used as v_e = C_b_e @ v_b
    bias_g   : (3,)   gyro bias [rad/s]
    bias_a   : (3,)   accel bias [m/s^2]

Why ECEF?
This was suggested in the first meeting. It keeps the integration a bit
cleaner because Earth rotation and gravity can be handled directly, without
extra transport-rate bookkeeping.

Body frame convention from the ground truth header:
    x = right, y = forward, z = up

Heading is true compass heading, clockwise from North.
Pitch positive means nose up.
Roll positive means right wing down.

Since body-z points up, ENU is the local nav frame that matches this setup.

Euler decomposition for this body convention:
    C_b_n_ENU = R_z(-H) @ R_x(P) @ R_y(R)

The minus on heading is only because compass heading is clockwise, while the
usual math rotation about +z is counterclockwise.

Then:
    C_b_e = C_n_e @ C_b_n

This file has:
    1. INSState dataclass
    2. initial-state loader from output/imu/initial_states.json
    3. C_b_e construction and quick sanity checks
    4. one-step ECEF mechanization

This runs the convention checks and a short static drift check for run2/3/4.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from coord_frames import (  # noqa: E402
    OMEGA_IE_E,
    R_ecef_enu,
    ecef_to_llh,
    gravity_wgs84,
    skew,
)

INITIAL_STATES_PATH = REPO_ROOT / "output" / "imu" / "initial_states.json"
DATA_DIR = REPO_ROOT / "data"

GYRO_COLS = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
ACCEL_COLS = ["Accel_X", "Accel_Y", "Accel_Z"]


@dataclass
class INSState:
    """Navigation state at a single IMU timestamp.

    Most of the state is expressed in ECEF. The only exceptions are the
    sensor biases, which stay in the body frame because that is where the IMU
    measurements arrive.
    """

    t: float                       # GPS time of this state [s]
    pos_ecef: np.ndarray           # (3,) [m]
    vel_ecef: np.ndarray           # (3,) [m/s]
    C_b_e: np.ndarray              # (3, 3) body -> ECEF DCM
    bias_g: np.ndarray             # (3,) gyro bias [rad/s]
    bias_a: np.ndarray             # (3,) accel bias [m/s^2]
    run_name: str = field(default="")

    def copy(self) -> "INSState":
        return INSState(
            t=self.t,
            pos_ecef=self.pos_ecef.copy(),
            vel_ecef=self.vel_ecef.copy(),
            C_b_e=self.C_b_e.copy(),
            bias_g=self.bias_g.copy(),
            bias_a=self.bias_a.copy(),
            run_name=self.run_name,
        )


# Basic right-hand-rule rotation matrices.

def _R_x(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _R_y(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _R_z(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def C_b_n_enu_from_attitude(heading_rad: float,
                            pitch_rad: float,
                            roll_rad: float) -> np.ndarray:
    """Build the body -> ENU DCM for this dataset's body convention.

    Two details are easy to mix up here:
    - the heading sign is flipped because compass heading is clockwise
    - the rotation order is Z, then X, then Y for our x-right/y-forward/z-up
      body frame
    """
    return _R_z(-heading_rad) @ _R_x(pitch_rad) @ _R_y(roll_rad)


def _rodrigues(phi: np.ndarray) -> np.ndarray:
    """Convert a rotation vector into a rotation matrix.

    We use this for attitude propagation instead of a first-order update.
    It keeps the DCM orthonormal much better during long integrations.
    """
    theta = float(np.linalg.norm(phi))
    K = skew(phi)

    if theta < 1e-12:
        # Small-angle version of Rodrigues.
        return np.eye(3) + K + 0.5 * K @ K

    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / theta ** 2) * K @ K)


def C_b_e_from_initial(lat: float,
                       lon: float,
                       heading_rad: float,
                       pitch_rad: float,
                       roll_rad: float) -> np.ndarray:
    """Build the initial body -> ECEF DCM.

    The rotation chain is:
        body -> ENU -> ECEF
    """
    C_b_n = C_b_n_enu_from_attitude(heading_rad, pitch_rad, roll_rad)
    C_n_e = R_ecef_enu(lat, lon)
    return C_n_e @ C_b_n


def load_initial_state(run_name: str,
                       json_path: Path = INITIAL_STATES_PATH) -> INSState:
    """Load the initial INS state for one run.

    The JSON stores gyro bias in deg/s, so we convert it to rad/s here and
    keep everything SI after this point.
    """
    data = json.loads(Path(json_path).read_text())
    if run_name not in data:
        raise KeyError(f"{run_name} not in {json_path.name} "
                       f"(available: {list(data.keys())})")

    rec = data[run_name]

    pos = np.array(rec["pos_ecef"])
    vel = np.array(rec["vel_ecef"])
    att = rec["attitude_rad"]

    lat, lon, _h = ecef_to_llh(*pos)

    C_b_e = C_b_e_from_initial(
        lat,
        lon,
        heading_rad=att["yaw"],
        pitch_rad=att["pitch"],
        roll_rad=att["roll"],
    )

    bias_g = np.deg2rad(np.array([
        rec["gyro_bias_deg_s"][c] for c in GYRO_COLS
    ]))
    bias_a = np.array([
        rec["accel_bias_m_s2"][c] for c in ACCEL_COLS
    ])

    return INSState(
        t=rec["t0"],
        pos_ecef=pos,
        vel_ecef=vel,
        C_b_e=C_b_e,
        bias_g=bias_g,
        bias_a=bias_a,
        run_name=run_name,
    )


def compensate_imu(state: INSState,
                   gyro_raw_deg_s: np.ndarray,
                   accel_raw_m_s2: np.ndarray) -> tuple:
    """Apply the current bias estimates to one IMU sample.

    Gyro arrives in deg/s in the file. We convert it to rad/s here so the
    rest of the mechanization does not need to care about file units.
    """
    omega_ib_b = np.deg2rad(np.asarray(gyro_raw_deg_s)) - state.bias_g
    f_b = np.asarray(accel_raw_m_s2) - state.bias_a

    return omega_ib_b, f_b


def mechanize_step(state: INSState,
                   gyro_raw_deg_s: np.ndarray,
                   accel_raw_m_s2: np.ndarray,
                   dt: float) -> INSState:
    """Run one ECEF mechanization step.

    Step order:
    1. compensate gyro and accel
    2. propagate attitude
    3. propagate velocity
    4. propagate position

    Bias states are not updated here. The KF can overwrite them between IMU
    steps when it has a new estimate.
    """
    omega_ib_b, f_b = compensate_imu(state, gyro_raw_deg_s, accel_raw_m_s2)

    # Remove Earth rotation from the body gyro rate.
    omega_ie_b = state.C_b_e.T @ OMEGA_IE_E
    omega_eb_b = omega_ib_b - omega_ie_b

    C_b_e_new = state.C_b_e @ _rodrigues(omega_eb_b * dt)

    # Gravity is local-down in ENU, then rotated into ECEF.
    lat, lon, h = ecef_to_llh(*state.pos_ecef)
    g_local = gravity_wgs84(lat, h)
    g_e = R_ecef_enu(lat, lon) @ np.array([0.0, 0.0, -g_local])

    # Midpoint attitude is a cheap way to reduce the velocity integration error.
    C_avg = 0.5 * (state.C_b_e + C_b_e_new)
    f_e = C_avg @ f_b

    # ECEF is rotating, so include the Coriolis term.
    coriolis_e = 2.0 * np.cross(OMEGA_IE_E, state.vel_ecef)

    a_e = f_e + g_e - coriolis_e
    vel_new = state.vel_ecef + a_e * dt

    # Trapezoidal position update.
    pos_new = state.pos_ecef + 0.5 * (state.vel_ecef + vel_new) * dt

    return INSState(
        t=state.t + dt,
        pos_ecef=pos_new,
        vel_ecef=vel_new,
        C_b_e=C_b_e_new,
        bias_g=state.bias_g,
        bias_a=state.bias_a,
        run_name=state.run_name,
    )


# ----------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------

def _predicted_static_specific_force_body(state: INSState,
                                          g_local: float) -> np.ndarray:
    """Predict what a perfect stationary IMU should measure.

    In a static case, the accelerometer measures specific force, so it should
    see roughly the opposite of gravity expressed in the body frame.

    Small but important detail: gravity should be local vertical, not simply
    radial from the centre of the Earth.
    """
    lat, lon, _ = ecef_to_llh(*state.pos_ecef)

    g_n = np.array([0.0, 0.0, -g_local])
    g_e = R_ecef_enu(lat, lon) @ g_n

    return state.C_b_e.T @ (-g_e)


def _run_static_drift_test(run_name: str = "run2",
                           duration_s: float = 30.0) -> bool:
    """Run a short static window through mechanization.

    This is not meant to prove the full INS is perfect. It is just a quick
    check that the signs, frames, and units are not obviously broken.
    """
    import pandas as pd

    init = json.loads(INITIAL_STATES_PATH.read_text())[run_name]
    t0 = init["t0"]
    t_end = t0 + duration_s

    imu_path = DATA_DIR / run_name / f"{run_name}_imu.txt"
    imu = pd.read_csv(imu_path, sep=r"\s+", engine="python")

    imu = imu[(imu["Time"] >= t0) & (imu["Time"] < t_end)].reset_index(drop=True)
    if len(imu) < 2:
        print(f"  {run_name}: not enough IMU samples in [t0, t0+{duration_s}s]")
        return False

    state = load_initial_state(run_name)
    pos0 = state.pos_ecef.copy()

    times = imu["Time"].values
    gyros = imu[GYRO_COLS].values
    accels = imu[ACCEL_COLS].values

    for i in range(1, len(imu)):
        dt = float(times[i] - times[i - 1])
        state = mechanize_step(state, gyros[i], accels[i], dt)

    pos_drift = float(np.linalg.norm(state.pos_ecef - pos0))
    vel_norm = float(np.linalg.norm(state.vel_ecef))
    elapsed = float(times[-1] - times[0])

    dcm_err = float(np.linalg.norm(state.C_b_e @ state.C_b_e.T - np.eye(3)))

    # ENU breakdown of drift and velocity, useful to see whether the residual
    # error is mostly horizontal (leveling/heading) or vertical (gravity/scale).
    from coord_frames import R_enu_ecef, ecef_to_llh
    lat0, lon0, _ = ecef_to_llh(*pos0)
    R_enu = R_enu_ecef(lat0, lon0)
    drift_enu = R_enu @ (state.pos_ecef - pos0)
    vel_enu = R_enu @ state.vel_ecef

    pos_ok = pos_drift < 20.0
    vel_ok = vel_norm < 5.0
    dcm_ok = dcm_err < 1e-9
    ok = pos_ok and vel_ok and dcm_ok

    print(f"  {run_name}: {len(imu)} samples over {elapsed:.2f} s")
    print(f"    |pos drift| = {pos_drift:>7.3f} m   (threshold 20 m) "
          f"[{'OK' if pos_ok else 'FAIL'}]")
    print(f"    |vel|       = {vel_norm:>7.3f} m/s (threshold  5 m/s) "
          f"[{'OK' if vel_ok else 'FAIL'}]")
    print(f"    DCM ortho err = {dcm_err:.2e}        (threshold 1e-9) "
          f"[{'OK' if dcm_ok else 'FAIL'}]")
    print(f"    drift ENU:  E={drift_enu[0]:+7.3f}  "
          f"N={drift_enu[1]:+7.3f}  U={drift_enu[2]:+7.3f}  m")
    print(f"    vel ENU:    E={vel_enu[0]:+7.3f}  "
          f"N={vel_enu[1]:+7.3f}  U={vel_enu[2]:+7.3f}  m/s")

    return ok


if __name__ == "__main__":
    init_data = json.loads(INITIAL_STATES_PATH.read_text())

    print("=" * 72)
    print("1. C_b_e convention check")
    print("=" * 72)
    print(f"{'run':>5}  {'predicted body f [m/s^2]':>34}  "
          f"{'measured static mean [m/s^2]':>34}  "
          f"{'|diff| [m/s^2]':>14}")
    print("-" * 100)

    all_pass = True

    for run_name in ["run2", "run3", "run4"]:
        state = load_initial_state(run_name)
        rec = init_data[run_name]

        g_local = rec["gravity_at_start_m_s2"]
        f_pred = _predicted_static_specific_force_body(state, g_local)
        f_meas = np.array([
            rec["accel_static_mean_m_s2"][c] for c in ACCEL_COLS
        ])

        diff = np.linalg.norm(f_pred - f_meas)

        # This should be around the accelerometer bias level.
        ok = diff < 0.05
        marker = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok

        print(f"{run_name:>5}  "
              f"({f_pred[0]:+.4f}, {f_pred[1]:+.4f}, {f_pred[2]:+.4f})  "
              f"({f_meas[0]:+.4f}, {f_meas[1]:+.4f}, {f_meas[2]:+.4f})  "
              f"{diff:>10.5f} [{marker}]")

    print()
    print("=" * 72)
    print("2. Static drift check for the first 30 seconds")
    print("=" * 72)

    drift_pass = True

    for run_name in ["run2", "run3", "run4"]:
        ok = _run_static_drift_test(run_name, duration_s=30.0)
        drift_pass = drift_pass and ok
        print()

    print("=" * 72)
    print(f"Overall: {'all checks passed' if (all_pass and drift_pass) else 'FAILED'}")

    if not (all_pass and drift_pass):
        sys.exit(1)