"""
Validate coord_frames.py against pymap3d at one reference point.

This script compares the manual coordinate transformations with pymap3d
using a synthetic test case near DTU.

It checks:
    - LLH -> ECEF
    - ECEF -> LLH
    - ECEF offset -> NED
    - ECEF offset -> ENU
    - NED/ENU axis consistency
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pymap3d as pm

from coord_frames import (
    R_enu_ecef,
    R_ned_ecef,
    ecef_to_llh,
    llh_to_ecef,
)


# Reference point near DTU campus
LAT_DEG, LON_DEG = 55.785, 12.521
H = 50.0

LAT = np.deg2rad(LAT_DEG)
LON = np.deg2rad(LON_DEG)


def report(label, ours, theirs, atol):
    diff = np.linalg.norm(np.asarray(ours) - np.asarray(theirs))
    flag = "PASS" if diff < atol else "FAIL"
    print(f"  [{flag}] {label}: |diff| = {diff:.3e}  (tol {atol:.0e})")
    return diff < atol


print(
    "Reference: lat = {:.4f} deg, lon = {:.4f} deg, h = {:.1f} m\n"
    .format(LAT_DEG, LON_DEG, H)
)

passed = []


# 1. LLH -> ECEF
print("LLH -> ECEF")

ours = llh_to_ecef(LAT, LON, H)
theirs = np.array(pm.geodetic2ecef(LAT_DEG, LON_DEG, H, deg=True))

print(f"  ours   : {ours}")
print(f"  pymap3d: {theirs}")

passed.append(report("xyz", ours, theirs, atol=1e-6))
print()


# 2. ECEF -> LLH
print("ECEF -> LLH")

xyz = theirs

ours_llh = ecef_to_llh(*xyz)
theirs_llh = np.array(pm.ecef2geodetic(*xyz, deg=False))

print(
    f"  ours   : lat={np.rad2deg(ours_llh[0]):.10f}  "
    f"lon={np.rad2deg(ours_llh[1]):.10f}  h={ours_llh[2]:.6f}"
)
print(
    f"  pymap3d: lat={np.rad2deg(theirs_llh[0]):.10f}  "
    f"lon={np.rad2deg(theirs_llh[1]):.10f}  h={theirs_llh[2]:.6f}"
)

passed.append(report("lat", ours_llh[0], theirs_llh[0], atol=1e-12))
passed.append(report("lon", ours_llh[1], theirs_llh[1], atol=1e-12))
passed.append(report("h", ours_llh[2], theirs_llh[2], atol=1e-6))
print()


# 3. ECEF offset -> NED
#
# The target point is placed approximately 100 m north, 200 m east,
# and 30 m above the reference point.
TARGET_LAT_DEG = LAT_DEG + 100.0 / 111_000.0
TARGET_LON_DEG = LON_DEG + 200.0 / (111_000.0 * np.cos(LAT))
TARGET_H = H + 30.0

target_ecef = np.array(
    pm.geodetic2ecef(TARGET_LAT_DEG, TARGET_LON_DEG, TARGET_H, deg=True)
)

delta = target_ecef - xyz

print("ECEF offset -> NED")

ours_ned = R_ned_ecef(LAT, LON) @ delta
theirs_ned = np.array(pm.ecef2ned(*target_ecef, LAT, LON, H, deg=False))

print(
    f"  ours   : N={ours_ned[0]:+.4f}  "
    f"E={ours_ned[1]:+.4f}  D={ours_ned[2]:+.4f}"
)
print(
    f"  pymap3d: N={theirs_ned[0]:+.4f}  "
    f"E={theirs_ned[1]:+.4f}  D={theirs_ned[2]:+.4f}"
)

passed.append(report("ned", ours_ned, theirs_ned, atol=1e-6))
print()


# 4. ECEF offset -> ENU
print("ECEF offset -> ENU")

ours_enu = R_enu_ecef(LAT, LON) @ delta
theirs_enu = np.array(pm.ecef2enu(*target_ecef, LAT, LON, H, deg=False))

print(
    f"  ours   : E={ours_enu[0]:+.4f}  "
    f"N={ours_enu[1]:+.4f}  U={ours_enu[2]:+.4f}"
)
print(
    f"  pymap3d: E={theirs_enu[0]:+.4f}  "
    f"N={theirs_enu[1]:+.4f}  U={theirs_enu[2]:+.4f}"
)

passed.append(report("enu", ours_enu, theirs_enu, atol=1e-6))
print()


# 5. NED/ENU axis consistency
print("NED vs ENU consistency")

passed.append(report("N axis", ours_ned[0], ours_enu[1], atol=1e-12))
passed.append(report("E axis", ours_ned[1], ours_enu[0], atol=1e-12))
passed.append(report("D = -U", ours_ned[2], -ours_enu[2], atol=1e-12))
print()


print("=" * 50)

if all(passed):
    print("ALL CHECKS PASSED. coord_frames.py agrees with pymap3d.")
else:
    n_fail = sum(1 for p in passed if not p)
    print(f"{n_fail} check(s) FAILED. Look above.")
    sys.exit(1)