"""
Pure INS test.

This script runs each dataset using only the IMU mechanization. No Kalman
filter and no GNSS corrections are used here. We just integrate the IMU from
the initial state and compare the result with groundtruth.

Why we have this script:
    1. To check that the mechanization behaves reasonably on real motion data.
    2. To show how quickly pure INS drifts without corrections.
    3. To use the drift plots as motivation for the KF + GNSS update step.

Note on run4:
    run4 is planned as the held-out run for KF tuning later. Here we are not
    tuning anything, only running the same pure INS pipeline and reporting the
    error. So this does not affect the holdout setup.
"""

import sys
import warnings
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "imu"))

from coord_frames import R_ned_ecef, ecef_to_llh                    # noqa: E402
from initial_state import load_groundtruth                          # noqa: E402
from mechanization import (                                          # noqa: E402
    ACCEL_COLS,
    GYRO_COLS,
    load_initial_state,
    mechanize_step,
)

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "imu"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAMES = ["run2", "run3", "run4"]

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"


def run_pure_ins(run_name: str) -> dict:
    """Run pure INS for one full dataset.

    We start from the initial state saved in initial_states.json and then call
    mechanize_step for every IMU sample after that time.
    """
    print(f"\n--- {run_name} ---")

    state = load_initial_state(run_name)

    imu_path = DATA_DIR / run_name / f"{run_name}_imu.txt"
    imu = pd.read_csv(imu_path, sep=r"\s+", engine="python")

    # Only keep the IMU samples after the selected initial time.
    imu = imu[imu["Time"] >= state.t].reset_index(drop=True)

    n = len(imu)
    duration = float(imu["Time"].iloc[-1] - imu["Time"].iloc[0])
    print(f"Mechanizing {n} IMU samples over {duration:.1f} s")

    times = np.zeros(n)
    pos_ins = np.zeros((n, 3))
    vel_ins = np.zeros((n, 3))

    times[0] = state.t
    pos_ins[0] = state.pos_ecef
    vel_ins[0] = state.vel_ecef

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
        pos_ins[i] = state.pos_ecef
        vel_ins[i] = state.vel_ecef

    return {
        "run": run_name,
        "times": times,
        "pos_ins": pos_ins,
        "vel_ins": vel_ins,
    }


def compare_with_groundtruth(ins: dict) -> dict:
    """Compare the INS result with groundtruth.

    The INS output is at IMU rate, while groundtruth usually has a different
    timestamp grid. We interpolate the INS result onto the groundtruth times
    before computing errors.
    """
    run_name = ins["run"]
    gt_path = DATA_DIR / run_name / f"{run_name}_groundtruth.txt"
    gt = load_groundtruth(gt_path)

    # Use the first groundtruth point as the local NED reference.
    ref = np.array([
        gt["X-ECEF"].iloc[0],
        gt["Y-ECEF"].iloc[0],
        gt["Z-ECEF"].iloc[0],
    ])

    ref_lat, ref_lon, _ = ecef_to_llh(*ref)
    R = R_ned_ecef(ref_lat, ref_lon)

    gt_times = gt["GPSTime"].values

    pos_ins_at_gt = np.zeros((len(gt_times), 3))
    vel_ins_at_gt = np.zeros((len(gt_times), 3))

    # Interpolate each ECEF component separately.
    for j in range(3):
        pos_ins_at_gt[:, j] = np.interp(
            gt_times,
            ins["times"],
            ins["pos_ins"][:, j],
        )
        vel_ins_at_gt[:, j] = np.interp(
            gt_times,
            ins["times"],
            ins["vel_ins"][:, j],
        )

    pos_gt = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].values
    vel_gt = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].values

    pos_err_e = pos_ins_at_gt - pos_gt
    vel_err_e = vel_ins_at_gt - vel_gt

    # Convert positions to local NED only for easier plotting.
    pos_ins_ned = (R @ (pos_ins_at_gt - ref).T).T
    pos_gt_ned = (R @ (pos_gt - ref).T).T

    return {
        "gt_times": gt_times,
        "pos_ins_ecef": pos_ins_at_gt,
        "pos_gt_ecef": pos_gt,
        "pos_ins_ned": pos_ins_ned,
        "pos_gt_ned": pos_gt_ned,
        "pos_err_e": pos_err_e,
        "vel_err_e": vel_err_e,
        "pos_err_mag": np.linalg.norm(pos_err_e, axis=1),
        "vel_err_mag": np.linalg.norm(vel_err_e, axis=1),
    }


def plot_trajectory(run_name: str, cmp: dict) -> Path:
    """Plot groundtruth and pure INS trajectory in the local NED frame."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        cmp["pos_gt_ned"][:, 1],
        cmp["pos_gt_ned"][:, 0],
        color="C0",
        lw=1.5,
        label="groundtruth",
    )
    ax.plot(
        cmp["pos_ins_ned"][:, 1],
        cmp["pos_ins_ned"][:, 0],
        color="C1",
        lw=0.8,
        alpha=0.85,
        label="pure INS",
    )
    ax.scatter(
        [0],
        [0],
        color="black",
        s=40,
        zorder=5,
        label="start",
    )

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(f"{run_name}: pure INS vs groundtruth in NED")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out = FIG_DIR / f"ins_only_{run_name}_trajectory.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    return out


def _ecef_to_latlon_deg(pos_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an (N, 3) ECEF array to (lat_deg, lon_deg) using coord_frames."""
    lat, lon, _ = ecef_to_llh(pos_ecef[:, 0], pos_ecef[:, 1], pos_ecef[:, 2])
    return np.rad2deg(lat), np.rad2deg(lon)


def _trajectory_gdf(pos_ecef: np.ndarray) -> gpd.GeoDataFrame:
    """Build a Web-Mercator LineString GeoDataFrame from an ECEF trajectory."""
    lat, lon = _ecef_to_latlon_deg(pos_ecef)
    line = LineString(np.column_stack([lon, lat]))
    return gpd.GeoDataFrame(geometry=[line], crs=WGS84).to_crs(WEB_MERCATOR)


def plot_trajectory_map(run_name: str, cmp: dict) -> Path:
    """Overlay groundtruth and pure INS trajectory on an OSM basemap."""
    gdf_gt = _trajectory_gdf(cmp["pos_gt_ecef"])
    gdf_ins = _trajectory_gdf(cmp["pos_ins_ecef"])

    fig, ax = plt.subplots(figsize=(10, 8))

    gdf_gt.plot(ax=ax, color="#2196F3", linewidth=2.0, label="groundtruth", zorder=4)
    gdf_ins.plot(ax=ax, color="#F44336", linewidth=1.0, alpha=0.9, label="pure INS", zorder=3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")

    ax.set_title(
        f"Pure INS vs Ground Truth - {run_name}",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_axis_off()
    ax.legend(
        loc="lower right",
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.tight_layout()

    out = FIG_DIR / f"ins_only_{run_name}_trajectory_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out


def plot_errors(run_name: str, cmp: dict) -> Path:
    """Plot position and velocity error over time."""
    t_rel = cmp["gt_times"] - cmp["gt_times"][0]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(
        t_rel,
        cmp["pos_err_mag"],
        color="black",
        lw=1.2,
        label="|pos err|",
    )
    axes[0].plot(
        t_rel,
        cmp["pos_err_e"][:, 0],
        color="C0",
        lw=0.6,
        alpha=0.7,
        label="X",
    )
    axes[0].plot(
        t_rel,
        cmp["pos_err_e"][:, 1],
        color="C1",
        lw=0.6,
        alpha=0.7,
        label="Y",
    )
    axes[0].plot(
        t_rel,
        cmp["pos_err_e"][:, 2],
        color="C2",
        lw=0.6,
        alpha=0.7,
        label="Z",
    )
    axes[0].set_ylabel("position error (ECEF) [m]")
    axes[0].set_title(f"{run_name}: pure INS errors against groundtruth")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        t_rel,
        cmp["vel_err_mag"],
        color="black",
        lw=1.2,
        label="|vel err|",
    )
    axes[1].plot(
        t_rel,
        cmp["vel_err_e"][:, 0],
        color="C0",
        lw=0.6,
        alpha=0.7,
        label="VX",
    )
    axes[1].plot(
        t_rel,
        cmp["vel_err_e"][:, 1],
        color="C1",
        lw=0.6,
        alpha=0.7,
        label="VY",
    )
    axes[1].plot(
        t_rel,
        cmp["vel_err_e"][:, 2],
        color="C2",
        lw=0.6,
        alpha=0.7,
        label="VZ",
    )
    axes[1].set_ylabel("velocity error (ECEF) [m/s]")
    axes[1].set_xlabel("time since start [s]")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    out = FIG_DIR / f"ins_only_{run_name}_errors.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    return out


def save_csv(run_name: str, ins: dict) -> Path:
    """Save the raw pure INS trajectory."""
    df = pd.DataFrame({
        "GPSTime": ins["times"],
        "X-ECEF": ins["pos_ins"][:, 0],
        "Y-ECEF": ins["pos_ins"][:, 1],
        "Z-ECEF": ins["pos_ins"][:, 2],
        "VX-ECEF": ins["vel_ins"][:, 0],
        "VY-ECEF": ins["vel_ins"][:, 1],
        "VZ-ECEF": ins["vel_ins"][:, 2],
    })

    out = OUT_DIR / f"ins_only_{run_name}.csv"
    df.to_csv(out, index=False, float_format="%.6f")

    return out


def main():
    summary = []

    for run in RUN_NAMES:
        ins = run_pure_ins(run)
        cmp = compare_with_groundtruth(ins)

        plot_trajectory(run, cmp)
        plot_trajectory_map(run, cmp)
        plot_errors(run, cmp)
        save_csv(run, ins)

        elapsed = float(cmp["gt_times"][-1] - cmp["gt_times"][0])
        final_pos = float(cmp["pos_err_mag"][-1])
        max_pos = float(cmp["pos_err_mag"].max())
        max_vel = float(cmp["vel_err_mag"].max())

        print(f"  duration:        {elapsed:>8.1f} s")
        print(f"  final |pos err|: {final_pos:>8.1f} m")
        print(f"  max   |pos err|: {max_pos:>8.1f} m")
        print(f"  max   |vel err|: {max_vel:>8.2f} m/s")

        summary.append({
            "run": run,
            "duration_s": elapsed,
            "final_pos_err_m": final_pos,
            "max_pos_err_m": max_pos,
            "max_vel_err_m_s": max_vel,
        })

    print()
    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"{'run':>6}  {'dur [s]':>9}  {'final |perr| [m]':>20}  "
          f"{'max |perr| [m]':>17}  {'max |verr| [m/s]':>18}")

    for s in summary:
        print(f"{s['run']:>6}  {s['duration_s']:>9.1f}  "
              f"{s['final_pos_err_m']:>20.1f}  "
              f"{s['max_pos_err_m']:>17.1f}  "
              f"{s['max_vel_err_m_s']:>18.2f}")


if __name__ == "__main__":
    main()