"""
Per-run initial state for INS mechanization.

For each of run2/3/4 we pull together everything the mechanization and KF
need to start running:
  - t0, pos_ecef, vel_ecef from the first row of the groundtruth file
  - initial attitude (roll, pitch, yaw) from the same row
  - per-run gyro bias and gravity-compensated accel bias, from the IMU
    samples that fall in the initial static segment
  - the local gravity magnitude at that point (so mechanization doesn't
    have to look it up again later)

Why we redo bias here instead of just reusing static_imu.txt:
the 45 min static recording was taken on a different day, possibly with
a different setup, so the biases there are at best a starting point. Each
run has its own static segment at the start; using those gives us a fresh
bias estimate that matches the actual IMU state during the run.

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

# "Stationary" if the groundtruth speed is below this. 5 cm/s comfortably
# clears GNSS noise without missing a slow start.
STATIC_VEL_THRESHOLD = 0.05
# Skip the first half-second so any settling jitter doesn't trigger
# the first-moving detector.
WARMUP_SAMPLES = 50
# We expect at least ~30 s of static at the start. If the segment is
# shorter than this we print a warning so we know to look.
MIN_STATIC_DURATION_S = 30.0


def load_groundtruth(path: Path) -> pd.DataFrame:
    """Same loader logic as src/utils/validate_with_groundtruth.py."""
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
    """Find how long the rover sits still at the start of the run.

    Returns (end_idx_exclusive, duration_s). The start is implicitly 0;
    end_idx is the first epoch where the speed exceeds the threshold,
    after a small warmup window.
    """
    vel = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].values
    speed = np.linalg.norm(vel, axis=1)
    end_idx = WARMUP_SAMPLES
    while end_idx < len(speed) and speed[end_idx] < STATIC_VEL_THRESHOLD:
        end_idx += 1
    duration_s = float(gt["GPSTime"].iloc[end_idx - 1] - gt["GPSTime"].iloc[0])
    return end_idx, duration_s


def expected_accel_body(roll_rad: float, pitch_rad: float, g_mag: float) -> np.ndarray:
    """Specific force a stationary, zero-bias IMU would read.

    Body-frame convention from the groundtruth headers (line 14 of every
    runX_groundtruth.txt): x = right, y = forward, z = up.
    With that convention:
      pitch = rotation around the right axis (x) -> tilts the forward axis (y)
      roll  = rotation around the forward axis (y) -> tilts the right axis (x)
      yaw   = rotation around up (z) -> drops out for gravity
    Gravity-only specific force in body:
      a_x = -sin(roll) * cos(pitch) * g
      a_y =  sin(pitch)             * g
      a_z =  cos(pitch) * cos(roll) * g
    """
    sr, cr = np.sin(roll_rad), np.cos(roll_rad)
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)
    return g_mag * np.array([-sr * cp, sp, cp * cr])


def imu_in_window(imu_df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    """IMU samples whose timestamps fall in [t_start, t_end)."""
    mask = (imu_df["Time"] >= t_start) & (imu_df["Time"] < t_end)
    return imu_df.loc[mask].reset_index(drop=True)


def process_run(run_name: str, global_calib: Optional[dict]) -> dict:
    print(f"\n--- {run_name} ---")
    gt = load_groundtruth(DATA_DIR / run_name / f"{run_name}_groundtruth.txt")
    imu = load_imu(DATA_DIR / run_name / f"{run_name}_imu.txt")

    # First epoch of the groundtruth = our t0
    first = gt.iloc[0]
    t0 = float(first["GPSTime"])
    pos0 = np.array([first["X-ECEF"], first["Y-ECEF"], first["Z-ECEF"]])
    vel0 = np.array([first["VX-ECEF"], first["VY-ECEF"], first["VZ-ECEF"]])
    roll0_deg = float(first["Roll"])
    pitch0_deg = float(first["Pitch"])
    yaw0_deg = float(first["Heading"])
    roll0, pitch0, yaw0 = np.deg2rad([roll0_deg, pitch0_deg, yaw0_deg])

    # Static segment from groundtruth speed
    end_idx, dur = detect_static_segment(gt)
    t_static_end = float(gt["GPSTime"].iloc[end_idx - 1])
    print(f"Static segment: {end_idx} groundtruth epochs, {dur:.1f} s "
          f"(GPSTime {t0:.2f} -> {t_static_end:.2f})")
    if dur < MIN_STATIC_DURATION_S:
        print(f"  Warning: segment shorter than {MIN_STATIC_DURATION_S} s")

    # IMU samples that fall in that window
    imu_static = imu_in_window(imu, t0, t_static_end)
    n_imu = len(imu_static)
    print(f"IMU samples in window: {n_imu}")

    # Local gravity magnitude (lat from initial position)
    lat0, _, h0 = ecef_to_llh(*pos0)
    g_mag = float(gravity_wgs84(lat0, h0))

    # Gyro bias = mean (Earth rotation rate ~0.004 deg/s, well below noise).
    # Std on the same window is the per-run noise floor and feeds straight
    # into the bias diagonal of P0 in the KF.
    gyro_mean = imu_static[GYRO_COLS].mean().values
    gyro_std = imu_static[GYRO_COLS].std().values

    # Accel bias = static mean - gravity-only specific force in body.
    # Std again gives the noise floor for P0.
    accel_mean = imu_static[ACCEL_COLS].mean().values
    accel_std = imu_static[ACCEL_COLS].std().values
    accel_expected = expected_accel_body(roll0, pitch0, g_mag)
    accel_bias = accel_mean - accel_expected

    # Bulk accelerometer scale-factor estimate.
    # |a_static| should equal local gravity if the sensor is perfectly scaled
    # and bias-free. Any residual ratio is a mix of true scale error and
    # axis-projected bias - we can't separate the two without controlled motion.
    # If the three runs give consistent ratios, we read it as scale; if they
    # scatter, the variation is mostly bias instability.
    g_measured = float(np.linalg.norm(accel_mean))
    scale_factor = g_measured / g_mag
    scale_ppm = (scale_factor - 1.0) * 1e6

    # Pretty-print
    print(f"g at run location:  {g_mag:.5f} m/s^2")
    print(f"Initial attitude:   roll={roll0_deg:+.3f}, pitch={pitch0_deg:+.3f}, yaw={yaw0_deg:+.3f} deg")
    print(f"Gyro bias  [deg/s]: {gyro_mean[0]:+.4f}, {gyro_mean[1]:+.4f}, {gyro_mean[2]:+.4f}")
    print(f"Gyro std   [deg/s]: {gyro_std[0]:+.4f}, {gyro_std[1]:+.4f}, {gyro_std[2]:+.4f}")
    print(f"Accel bias [m/s^2]: {accel_bias[0]:+.4f}, {accel_bias[1]:+.4f}, {accel_bias[2]:+.4f}")
    print(f"Accel std  [m/s^2]: {accel_std[0]:+.4f}, {accel_std[1]:+.4f}, {accel_std[2]:+.4f}")
    print(f"|accel mean| = {g_measured:.5f} m/s^2  vs  g_local = {g_mag:.5f}  "
          f"-> scale = {scale_factor:.6f}  ({scale_ppm:+.1f} ppm)")

    if global_calib is not None:
        global_gyro = np.array([global_calib["bias"][c] for c in GYRO_COLS])
        delta_gyro = gyro_mean - global_gyro
        print(f"Gyro bias delta vs static_imu.txt: "
              f"{delta_gyro[0]:+.4f}, {delta_gyro[1]:+.4f}, {delta_gyro[2]:+.4f}")

    # Plot the static-segment IMU raw signals for this run
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t_rel = imu_static["Time"].values - t0
    for col in GYRO_COLS:
        axes[0].plot(t_rel, imu_static[col], label=col, lw=0.5)
    axes[0].set_ylabel("Gyro [deg/s]")
    axes[0].set_title(f"{run_name}: initial static segment ({dur:.1f} s, {n_imu} IMU samples)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    for col in ACCEL_COLS:
        axes[1].plot(t_rel, imu_static[col], label=col, lw=0.5)
    axes[1].set_ylabel("Accel [m/s^2]")
    axes[1].set_xlabel("Time since t0 [s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"initial_static_{run_name}.png", dpi=120)
    plt.close(fig)
    print(f"Saved {FIG_DIR / f'initial_static_{run_name}.png'}")

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
    """One figure: gyro bias side-by-side with static_imu reference, and
    accel bias per run (gravity compensated, so this is the real sensor
    bias number mechanization will subtract)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3)
    w = 0.2
    runs = list(results.keys())

    # Gyro panel: static_imu reference + each run
    if global_calib is not None:
        global_gyro = np.array([global_calib["bias"][c] for c in GYRO_COLS])
        axes[0].bar(x - 1.5 * w, global_gyro, width=w, label="static_imu", color="grey")
        for i, r in enumerate(runs):
            vals = [results[r]["gyro_bias_deg_s"][c] for c in GYRO_COLS]
            axes[0].bar(x - 0.5 * w + i * w, vals, width=w, label=r)
    else:
        for i, r in enumerate(runs):
            vals = [results[r]["gyro_bias_deg_s"][c] for c in GYRO_COLS]
            axes[0].bar(x - w + i * w, vals, width=w, label=r)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(GYRO_COLS)
    axes[0].set_ylabel("bias [deg/s]")
    axes[0].set_title("Gyro bias: per-run vs static_imu reference")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Accel panel: only per-run (the static_imu "bias" includes gravity
    # because we never extracted attitude there, so it's not directly
    # comparable to the gravity-compensated per-run numbers).
    for i, r in enumerate(runs):
        vals = [results[r]["accel_bias_m_s2"][c] for c in ACCEL_COLS]
        axes[1].bar(x - w + i * w, vals, width=w, label=r)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ACCEL_COLS)
    axes[1].set_ylabel("bias [m/s^2]")
    axes[1].set_title("Accel bias per run (gravity compensated)")
    axes[1].axhline(0, color="black", lw=0.5)
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
        print(f"No reference calibration at {GLOBAL_CALIB_PATH}, "
              f"per-run results will not be compared.")

    results = {name: process_run(name, global_calib) for name in RUN_NAMES}

    out_json = OUT_DIR / "initial_states.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_json}")

    plot_bias_comparison(results, global_calib)


if __name__ == "__main__":
    main()
