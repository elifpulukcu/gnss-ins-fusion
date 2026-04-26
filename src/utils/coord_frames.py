"""
Coordinate-frame helper functions used across the project.

The transformations are implemented explicitly so the frame conventions
and equations remain clear.

Frames:
    LLH  : latitude, longitude, height in (rad, rad, m)
    ECEF : Earth-centred Earth-fixed coordinates (x, y, z) in m
    NED  : local North-East-Down frame at a reference point
    ENU  : local East-North-Up frame, mainly used for plotting
    body : IMU/body frame

Conventions:
    - Euler angles use intrinsic ZYX order: yaw, pitch, roll.
    - R_a_b maps vectors from frame b to frame a:
          v_a = R_a_b @ v_b

Gravity is computed using WGS84 normal gravity with a free-air altitude
correction.
"""

import numpy as np


# WGS84 ellipsoid and Earth rotation constants
WGS84_A = 6378137.0                         # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563               # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)         # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)        # first eccentricity squared
OMEGA_E = 7.2921151467e-5                   # Earth rotation rate [rad/s]
OMEGA_IE_E = np.array([0.0, 0.0, OMEGA_E])  # ECEF-frame rotation vector


def llh_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """Convert LLH (lat, lon, h) to ECEF (x, y, z).

    Input units:
        lat, lon: radians
        h       : metres

    Output:
        ECEF position in metres.
    """
    sinlat = np.sin(lat)
    coslat = np.cos(lat)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sinlat * sinlat)

    x = (N + h) * coslat * np.cos(lon)
    y = (N + h) * coslat * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + h) * sinlat

    return np.array([x, y, z])


def ecef_to_llh(x: float, y: float, z: float) -> np.ndarray:
    """Convert ECEF (x, y, z) to LLH (lat, lon, h).

    Uses a Bowring-style fixed-point iteration.

    Input:
        x, y, z: ECEF position in metres

    Output:
        lat, lon: radians
        h       : metres
    """
    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)

    # Initial latitude estimate
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))

    for _ in range(5):
        sinlat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sinlat * sinlat)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h)))

    sinlat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sinlat * sinlat)
    h = p / np.cos(lat) - N

    return np.array([lat, lon, h])


def R_ned_ecef(lat: float, lon: float) -> np.ndarray:
    """Return the rotation matrix from ECEF to NED.

    Apply as:
        v_ned = R_ned_ecef(lat, lon) @ v_ecef
    """
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)

    return np.array([
        [-sl * co, -sl * so,  cl],
        [    -so,      co,   0.0],
        [-cl * co, -cl * so, -sl],
    ])


def R_ecef_ned(lat: float, lon: float) -> np.ndarray:
    """Return the rotation matrix from NED to ECEF."""
    return R_ned_ecef(lat, lon).T


def R_enu_ecef(lat: float, lon: float) -> np.ndarray:
    """Return the rotation matrix from ECEF to ENU.

    Apply as:
        v_enu = R_enu_ecef(lat, lon) @ v_ecef
    """
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)

    return np.array([
        [    -so,      co,   0.0],
        [-sl * co, -sl * so,  cl],
        [ cl * co,  cl * so,  sl],
    ])


def R_ecef_enu(lat: float, lon: float) -> np.ndarray:
    """Return the rotation matrix from ENU to ECEF."""
    return R_enu_ecef(lat, lon).T


def gravity_wgs84(lat: float, h: float = 0.0) -> float:
    """Compute WGS84 normal gravity at latitude and height.

    Uses the Somigliana surface formula with a free-air altitude correction.

    Input:
        lat: latitude in radians
        h  : height in metres

    Output:
        gravity magnitude in m/s^2
    """
    s2 = np.sin(lat) ** 2

    g_surface = (
        9.7803253359
        * (1.0 + 0.001931853 * s2)
        / np.sqrt(1.0 - WGS84_E2 * s2)
    )

    return g_surface - 3.086e-6 * h


def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX intrinsic Euler angles to a direction cosine matrix.

    The returned matrix is R_n_b, mapping body-frame vectors to nav-frame
    vectors:

        v_n = R_n_b @ v_b
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [    -sp,                 cp * sr,                 cp * cr],
    ])


def dcm_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert a DCM to ZYX Euler angles.

    Output:
        roll, pitch, yaw in radians.
    """
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3D vector.

    For vectors v and u:

        skew(v) @ u == np.cross(v, u)
    """
    x, y, z = v

    return np.array([
        [ 0.0,  -z,    y],
        [   z, 0.0,   -x],
        [  -y,    x, 0.0],
    ])


if __name__ == "__main__":
    # 1) LLH -> ECEF -> LLH round-trip near DTU
    lat = np.deg2rad(55.785)
    lon = np.deg2rad(12.521)
    h = 50.0

    xyz = llh_to_ecef(lat, lon, h)
    lat2, lon2, h2 = ecef_to_llh(*xyz)

    assert abs(lat - lat2) < 1e-10, f"lat mismatch: {lat} vs {lat2}"
    assert abs(lon - lon2) < 1e-10, f"lon mismatch: {lon} vs {lon2}"
    assert abs(h - h2) < 1e-3, f"height mismatch: {h} vs {h2}"

    print(f"LLH <-> ECEF round-trip OK, |dh| = {abs(h - h2):.3e} m")

    # 2) Rotation matrices should be orthonormal with determinant +1
    R = R_ned_ecef(lat, lon)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12), "R_ned_ecef not orthonormal"
    assert abs(np.linalg.det(R) - 1.0) < 1e-12, "det(R_ned_ecef) != 1"

    R2 = R_enu_ecef(lat, lon)
    assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-12), "R_enu_ecef not orthonormal"
    assert abs(np.linalg.det(R2) - 1.0) < 1e-12, "det(R_enu_ecef) != 1"

    print("Rotation matrices OK")

    # 3) NED and ENU should match after axis reordering
    v_ecef = np.array([1.0, 2.0, 3.0])

    v_ned = R_ned_ecef(lat, lon) @ v_ecef
    v_enu = R_enu_ecef(lat, lon) @ v_ecef

    assert abs(v_ned[0] - v_enu[1]) < 1e-12, "N mismatch between NED and ENU"
    assert abs(v_ned[1] - v_enu[0]) < 1e-12, "E mismatch between NED and ENU"
    assert abs(v_ned[2] + v_enu[2]) < 1e-12, "Down should be -Up"

    print("NED/ENU consistency OK")

    # 4) Gravity sanity checks
    g_eq = gravity_wgs84(0.0, 0.0)
    g_45 = gravity_wgs84(np.pi / 4, 0.0)
    g_pole = gravity_wgs84(np.pi / 2, 0.0)

    print(f"g at equator = {g_eq:.5f} m/s^2")
    print(f"g at 45 deg  = {g_45:.5f} m/s^2")
    print(f"g at pole    = {g_pole:.5f} m/s^2")

    assert 9.778 < g_eq < 9.781
    assert 9.831 < g_pole < 9.834

    g_dtu = gravity_wgs84(lat, h)
    print(f"g at DTU     = {g_dtu:.5f} m/s^2")

    # 5) Euler -> DCM -> Euler round-trip
    rpy = np.deg2rad([5.0, -10.0, 30.0])

    R = euler_to_dcm(*rpy)
    rpy2 = dcm_to_euler(R)

    assert np.allclose(rpy, rpy2, atol=1e-12), (
        f"Euler round-trip failed: {rpy} vs {rpy2}"
    )
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12

    print("Euler <-> DCM round-trip OK")

    # 6) Skew matrix check
    v = np.array([1.0, -2.0, 3.0])
    u = np.array([4.0, 5.0, 6.0])

    S = skew(v)

    assert np.allclose(S, -S.T), "skew matrix is not antisymmetric"
    assert np.allclose(S @ u, np.cross(v, u)), "skew(v) @ u != cross(v, u)"

    print("Skew matrix OK")

    print("\nAll self-tests passed.")