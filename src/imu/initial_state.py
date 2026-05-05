"""
Build the initial state for each run.

For run2/3/4 this script saves:
  - t0, position and velocity from the first groundtruth epoch
  - roll and pitch estimated from the mean accelerometer reading while static
  - heading from the provided Heading column
  - gyro and accelerometer bias estimates from the initial static segment
  - local gravity at the start point

The long static IMU file is useful as a reference, but each run has its own
static period at the start. Using that period gives a bias estimate that is
closer to the actual sensor state during the run.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from coord_frames import ecef_to_llh, gravity_wgs84

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "imu"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_CALIB_PATH = OUT_DIR / "imu_calibration.json"
RUN_NAMES = ["run2", "run3", "run4"]

GYRO_COLS = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
ACCEL_COLS = ["Accel_X", "Accel_Y", "Accel_Z"]

STATIC_VEL_THRESHOLD = 0.05
WARMUP_SAMPLES = 50
MIN_STATIC_DURATION_S = 30.0


def load_groundtruth(path: Path) -> pd.DataFrame:
    """Load numeric rows from a runX_groundtruth.txt file."""
    with open(path) as f:
        lines = f.readlines()

    header_idx = next(
        i for i, line in enumerate(lines) if line.lstrip().startswith("GPSTime")
    )
    units_idx = header_idx + 1

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=lambda i: i < header_idx or i == units_idx,
        engine="python",
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)


def load_imu(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", engine="python")


def detect_static_segment(gt: pd.DataFrame) -> tuple[int, float]:
    """Find the initial static period from the groundtruth speed."""
    vel = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].values
    speed = np.linalg.norm(vel, axis=1)

    end_idx = WARMUP_SAMPLES
    while end_idx < len(speed) and speed[end_idx] < STATIC_VEL_THRESHOLD:
        end_idx += 1

    duration_s = float(gt["GPSTime"].iloc[end_idx - 1] - gt["GPSTime"].iloc[0])
    return end_idx, duration_s


def expected_accel_body(roll_rad: float, pitch_rad: float, g_mag: float) -> np.ndarray:
    """Expected static accelerometer reading for the dataset body frame.

    Body frame: x=right, y=forward, z=up.
    """
    sr, cr = np.sin(roll_rad), np.cos(roll_rad)
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)

    return g_mag * np.array([
        -sr * cp,
        sp,
        cp * cr,
    ])


def attitude_from_static_accel(accel_mean: np.ndarray) -> tuple[float, float]:
    """Estimate roll and pitch from the mean static accelerometer reading.

    This uses gravity, so yaw cannot be estimated here.
    """
    ax, ay, az = accel_mean

    pitch_rad = float(np.arctan2(ay, np.hypot(ax, az)))
    roll_rad = float(np.arctan2(-ax, az))

    return roll_rad, pitch_rad


def imu_in_window(imu_df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    """Return IMU samples in [t_start, t_end)."""
    mask = (imu_df["Time"] >= t_start) & (imu_df["Time"] < t_end)
    return imu_df.loc[mask].reset_index(drop=True)


def process_run(run_name: str, global_calib: Optional[dict]) -> dict:
    print(f"\n--- {run_name} ---")

    gt = load_groundtruth(DATA_DIR / run_name / f"{run_name}_groundtruth.txt")
    imu = load_imu(DATA_DIR / run_name / f"{run_name}_imu.txt")

    first = gt.iloc[0]

    t0 = float(first["GPSTime"])
    pos0 = np.array([first["X-ECEF"], first["Y-ECEF"], first["Z-ECEF"]])
    vel0 = np.array([first["VX-ECEF"], first["VY-ECEF"], first["VZ-ECEF"]])

    # Heading comes from the provided Heading column. Roll and pitch are
    # estimated from the static accelerometer mean below.
    yaw0_deg = float(first["Heading"])
    yaw0 = float(np.deg2rad(yaw0_deg))

    # Groundtruth roll/pitch are kept only for comparison.
    roll_gt_deg = float(first["Roll"])
    pitch_gt_deg = float(first["Pitch"])

    end_idx, dur = detect_static_segment(gt)
    t_static_end = float(gt["GPSTime"].iloc[end_idx - 1])

    print(
        f"Static segment: {end_idx} groundtruth epochs, {dur:.1f} s "
        f"(GPSTime {t0:.2f} -> {t_static_end:.2f})"
    )

    if dur < MIN_STATIC_DURATION_S:
        print(f"  Warning: segment shorter than {MIN_STATIC_DURATION_S} s")

    imu_static = imu_in_window(imu, t0, t_static_end)
    n_imu = len(imu_static)
    print(f"IMU samples in window: {n_imu}")

    lat0, _, h0 = ecef_to_llh(*pos0)
    g_mag = float(gravity_wgs84(lat0, h0))

    gyro_mean = imu_static[GYRO_COLS].mean().values
    gyro_std = imu_static[GYRO_COLS].std().values

    accel_mean = imu_static[ACCEL_COLS].mean().values
    accel_std = imu_static[ACCEL_COLS].std().values

    roll0, pitch0 = attitude_from_static_accel(accel_mean)
    roll0_deg = float(np.rad2deg(roll0))
    pitch0_deg = float(np.rad2deg(pitch0))

    accel_expected = expected_accel_body(roll0, pitch0, g_mag)
    accel_bias = accel_mean - accel_expected

    g_measured = float(np.linalg.norm(accel_mean))
    scale_factor = g_measured / g_mag
    scale_ppm = (scale_factor - 1.0) * 1e6

    print(f"g at run location:  {g_mag:.5f} m/s^2")
    print(
        f"Initial attitude:   roll={roll0_deg:+.3f}, "
        f"pitch={pitch0_deg:+.3f}, yaw={yaw0_deg:+.3f} deg"
    )
    print(
        f"  vs groundtruth:   roll={roll_gt_deg:+.3f}, "
        f"pitch={pitch_gt_deg:+.3f} "
        f"(delta roll={roll0_deg - roll_gt_deg:+.3f}, "
        f"delta pitch={pitch0_deg - pitch_gt_deg:+.3f} deg)"
    )
    print(
        f"Gyro bias  [deg/s]: {gyro_mean[0]:+.4f}, "
        f"{gyro_mean[1]:+.4f}, {gyro_mean[2]:+.4f}"
    )
    print(
        f"Gyro std   [deg/s]: {gyro_std[0]:+.4f}, "
        f"{gyro_std[1]:+.4f}, {gyro_std[2]:+.4f}"
    )
    print(
        f"Accel bias [m/s^2]: {accel_bias[0]:+.4f}, "
        f"{accel_bias[1]:+.4f}, {accel_bias[2]:+.4f}"
    )
    print(
        f"Accel std  [m/s^2]: {accel_std[0]:+.4f}, "
        f"{accel_std[1]:+.4f}, {accel_std[2]:+.4f}"
    )
    print(
        f"|accel mean| = {g_measured:.5f} m/s^2  vs  "
        f"g_local = {g_mag:.5f} -> scale = {scale_factor:.6f} "
        f"({scale_ppm:+.1f} ppm)"
    )

    if global_calib is not None:
        global_gyro = np.array([global_calib["bias"][c] for c in GYRO_COLS])
        delta_gyro = gyro_mean - global_gyro
        print(
            f"Gyro bias delta vs static_imu.txt: "
            f"{delta_gyro[0]:+.4f}, {delta_gyro[1]:+.4f}, {delta_gyro[2]:+.4f}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t_rel = imu_static["Time"].values - t0

    for col in GYRO_COLS:
        axes[0].plot(t_rel, imu_static[col], label=col, lw=0.5)
    axes[0].set_ylabel("Gyro [deg/s]")
    axes[0].set_title(
        f"{run_name}: initial static segment ({dur:.1f} s, {n_imu} IMU samples)"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    for col in ACCEL_COLS:
        axes[1].plot(t_rel, imu_static[col], label=col, lw=0.5)
    axes[1].set_ylabel("Accel [m/s^2]")
    axes[1].set_xlabel("Time since t0 [s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_fig = FIG_DIR / f"initial_static_{run_name}.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)
    print(f"Saved {out_fig}")

    return {
        "t0": t0,
        "pos_ecef": pos0.tolist(),
        "vel_ecef": vel0.tolist(),
        "attitude_deg": {
            "roll": roll0_deg,
            "pitch": pitch0_deg,
            "yaw": yaw0_deg,
        },
        "attitude_rad": {
            "roll": float(roll0),
            "pitch": float(pitch0),
            "yaw": float(yaw0),
        },
        "attitude_source": {
            "roll": "static_accel_leveling",
            "pitch": "static_accel_leveling",
            "yaw": "provided_heading_column",
        },
        "attitude_groundtruth_deg": {
            "roll": roll_gt_deg,
            "pitch": pitch_gt_deg,
            "yaw": yaw0_deg,
        },
        "static_segment": {
            "duration_s": dur,
            "n_groundtruth_epochs": int(end_idx),
            "t_start": t0,
            "t_end": t_static_end,
            "n_imu_samples": int(n_imu),
        },
        "gravity_at_start_m_s2": g_mag,
        "gyro_bias_deg_s": dict(zip(GYRO_COLS, gyro_mean.tolist())),
        "gyro_std_deg_s": dict(zip(GYRO_COLS, gyro_std.tolist())),
        "accel_static_mean_m_s2": dict(zip(ACCEL_COLS, accel_mean.tolist())),
        "accel_expected_gravity_m_s2": dict(zip(ACCEL_COLS, accel_expected.tolist())),
        "accel_bias_m_s2": dict(zip(ACCEL_COLS, accel_bias.tolist())),
        "accel_std_m_s2": dict(zip(ACCEL_COLS, accel_std.tolist())),
        "accel_scale_estimate": {
            "g_measured": g_measured,
            "g_expected_wgs84": g_mag,
            "scale_factor": scale_factor,
            "ppm_offset": scale_ppm,
        },
    }


def plot_bias_comparison(results: dict, global_calib: Optional[dict]) -> None:
    """Plot per-run bias estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3)
    w = 0.2
    runs = list(results.keys())

    if global_calib is not None:
        global_gyro = np.array([global_calib["bias"][c] for c in GYRO_COLS])
        axes[0].bar(x - 1.5 * w, global_gyro, width=w, label="static_imu")

        for i, run in enumerate(runs):
            vals = [results[run]["gyro_bias_deg_s"][c] for c in GYRO_COLS]
            axes[0].bar(x - 0.5 * w + i * w, vals, width=w, label=run)
    else:
        for i, run in enumerate(runs):
            vals = [results[run]["gyro_bias_deg_s"][c] for c in GYRO_COLS]
            axes[0].bar(x - w + i * w, vals, width=w, label=run)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(GYRO_COLS)
    axes[0].set_ylabel("bias [deg/s]")
    axes[0].set_title("Gyro bias estimates")
    axes[0].axhline(0, lw=0.5)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for i, run in enumerate(runs):
        vals = [results[run]["accel_bias_m_s2"][c] for c in ACCEL_COLS]
        axes[1].bar(x - w + i * w, vals, width=w, label=run)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ACCEL_COLS)
    axes[1].set_ylabel("bias [m/s^2]")
    axes[1].set_title("Accelerometer bias estimates")
    axes[1].axhline(0, lw=0.5)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = FIG_DIR / "per_run_bias_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"\nSaved {out}")


def main() -> None:
    if GLOBAL_CALIB_PATH.exists():
        global_calib = json.loads(GLOBAL_CALIB_PATH.read_text())
        print(f"Reference calibration: {GLOBAL_CALIB_PATH.relative_to(REPO_ROOT)}")
    else:
        global_calib = None
        print(
            f"No reference calibration at {GLOBAL_CALIB_PATH}, "
            f"per-run results will not be compared."
        )

    results = {name: process_run(name, global_calib) for name in RUN_NAMES}

    out_json = OUT_DIR / "initial_states.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_json}")

    plot_bias_comparison(results, global_calib)


if __name__ == "__main__":
    main()