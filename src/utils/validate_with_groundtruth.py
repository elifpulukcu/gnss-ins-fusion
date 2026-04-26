"""
Validate coord_frames.py against pymap3d using real run data.

This script reads ECEF positions from the run2/run3/run4 groundtruth files,
converts them with both implementations, and compares the results.

It checks:
    - ECEF -> LLH
    - ECEF -> NED

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm

from coord_frames import R_ned_ecef, ecef_to_llh


REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_FILES = [
    REPO_ROOT / "data" / "run2" / "run2_groundtruth.txt",
    REPO_ROOT / "data" / "run3" / "run3_groundtruth.txt",
    REPO_ROOT / "data" / "run4" / "run4_groundtruth.txt",
]

OUT_DIR = REPO_ROOT / "output" / "utils"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_groundtruth(path: Path) -> pd.DataFrame:
    """Load a groundtruth file and return the numeric data rows."""
    with open(path) as f:
        lines = f.readlines()

    header_idx = next(
        i for i, line in enumerate(lines)
        if line.lstrip().startswith("GPSTime")
    )
    units_idx = header_idx + 1

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=lambda i: i < header_idx or i == units_idx,
        engine="python",
    )

    # Drop non-numeric rows such as the trailing warning block.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)


def validate_run(path: Path):
    """Compare manual and pymap3d conversions for one run."""
    print(f"\n--- {path.parent.name} ---")

    df = load_groundtruth(path)
    n = len(df)

    print(f"Loaded {n} epochs")

    xyz = df[["X-ECEF", "Y-ECEF", "Z-ECEF"]].values

    # ECEF -> LLH on a subsampled trajectory
    step = max(1, n // 500)
    xyz_sub = xyz[::step]

    llh_ours = np.array([ecef_to_llh(*p) for p in xyz_sub])
    llh_pm = np.array([pm.ecef2geodetic(*p, deg=False) for p in xyz_sub])

    dlat = np.abs(llh_ours[:, 0] - llh_pm[:, 0])
    dlon = np.abs(llh_ours[:, 1] - llh_pm[:, 1])
    dh = np.abs(llh_ours[:, 2] - llh_pm[:, 2])

    print(f"ECEF -> LLH ({len(xyz_sub)} sampled epochs)")
    print(f"  max |dlat| = {dlat.max():.3e} rad ({np.rad2deg(dlat.max()):.3e} deg)")
    print(f"  max |dlon| = {dlon.max():.3e} rad ({np.rad2deg(dlon.max()):.3e} deg)")
    print(f"  max |dh|   = {dh.max():.3e} m")

    # ECEF -> NED using the first epoch as reference
    ref = xyz[0]
    ref_lat, ref_lon, ref_h = ecef_to_llh(*ref)

    R = R_ned_ecef(ref_lat, ref_lon)
    ned_ours = (R @ (xyz - ref).T).T

    ned_pm = np.array([
        pm.ecef2ned(*p, ref_lat, ref_lon, ref_h, deg=False)
        for p in xyz
    ])

    diff_ned = np.linalg.norm(ned_ours - ned_pm, axis=1)

    print(f"ECEF -> NED ({n} epochs, ref = epoch 0)")
    print(f"  max  |diff| = {diff_ned.max():.3e} m")
    print(f"  mean |diff| = {diff_ned.mean():.3e} m")
    print(
        "  trajectory extent: "
        f"N={np.ptp(ned_ours[:, 0]):.1f} m, "
        f"E={np.ptp(ned_ours[:, 1]):.1f} m, "
        f"D={np.ptp(ned_ours[:, 2]):.1f} m"
    )

    return ned_ours, ned_pm


if __name__ == "__main__":
    print("Validating coord_frames.py against pymap3d on real groundtruth data")

    results = []

    for path in RUN_FILES:
        if not path.exists():
            print(f"\nSkipping missing file: {path}")
            continue

        ned_ours, ned_pm = validate_run(path)
        results.append((path.parent.name, ned_ours, ned_pm))

    if not results:
        raise FileNotFoundError("No groundtruth files were found.")

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(5 * len(results), 5),
        squeeze=False,
    )

    for ax, (name, ned_ours, ned_pm) in zip(axes[0], results):
        ax.plot(
            ned_ours[:, 1],
            ned_ours[:, 0],
            lw=1.0,
            label="coord_frames.py",
        )
        ax.plot(
            ned_pm[:, 1],
            ned_pm[:, 0],
            "--",
            lw=1.0,
            label="pymap3d",
            alpha=0.7,
        )
        ax.scatter(
            [0],
            [0],
            c="red",
            s=30,
            zorder=5,
            label="reference",
        )

        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("NED trajectories: coord_frames.py vs pymap3d")
    fig.tight_layout()

    out_path = OUT_DIR / "trajectory_ned_validation.png"
    fig.savefig(out_path, dpi=120)

    print(f"\nSaved {out_path}")