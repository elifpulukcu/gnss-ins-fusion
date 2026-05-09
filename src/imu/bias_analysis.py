"""
Static IMU analysis on the 45 min stationary recording.

What we get out of this:
  1. per-axis bias + noise std (mean and std of each channel while stationary)
  2. roll / pitch inferred from gravity (sanity check + initial alignment)
  3. Allan deviation curves -> ARW, VRW and bias instability numbers
  4. one big JSON (output/imu/imu_calibration.json) + a few PNGs

The JSON is what Kerick should load when filling in the Q matrix and the
initial bias states. The PNGs are mostly for the report and for us to
sanity-check that nothing weird is going on with the data.

"""

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "static_imu.txt"
OUT_DIR = REPO_ROOT / "output" / "imu"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GYRO_COLS = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
ACCEL_COLS = ["Accel_X", "Accel_Y", "Accel_Z"]
GRAVITY = 9.80665  # WGS84 reference value, m/s^2


# %%
def load_static_imu(path: Path) -> pd.DataFrame:
    # the file is whitespace separated with one header row, no other quirks
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df["t_rel"] = df["Time"] - df["Time"].iloc[0]
    return df


df = load_static_imu(DATA_PATH)
duration_s = df["t_rel"].iloc[-1]
n = len(df)
fs = (n - 1) / duration_s  # nominally 100 Hz, compute from data to be safe
print(f"Loaded {n} samples, duration={duration_s:.2f} s, fs={fs:.2f} Hz")
print(df.head())


# %%
# When the IMU is stationary the mean of each axis is its bias, and the std is
# the white-noise floor on top of that bias. min/max are just to spot outliers.
stats = df[GYRO_COLS + ACCEL_COLS].agg(["mean", "std", "min", "max"]).T
stats.columns = ["mean", "std", "min", "max"]
print("\nPer-axis statistics:")
print(stats.to_string(float_format=lambda x: f"{x:+.6f}"))


# %%
# Sanity check 1: |a| should be close to 9.81 if the accelerometer is sane.
g_meas = np.linalg.norm(stats.loc[ACCEL_COLS, "mean"].values)
print(f"\n|accel mean| = {g_meas:.4f}  (expected ~{GRAVITY:.4f} m/s^2)")
print(f"X/Y bias (tilt + sensor bias mixed): "
      f"ax={stats.loc['Accel_X', 'mean']:+.4f}, "
      f"ay={stats.loc['Accel_Y', 'mean']:+.4f}")

# This sensor outputs Az ~ +9.81 when stationary, so the convention is
# specific force with body-z pointing up. With body convention
# x=right, y=forward, z=up, pitch is rotation about x and roll is
# rotation about y, so the accelerometer-leveling formulas are:
#   pitch = atan2(a_y, sqrt(a_x^2 + a_z^2))
#   roll  = atan2(-a_x, a_z)
# Heads up: if a future IMU swaps signs (Az ~ -9.81) flip these.
ax, ay, az = stats.loc[ACCEL_COLS, "mean"].values
pitch_rad = np.arctan2(ay, np.hypot(ax, az))
roll_rad = np.arctan2(-ax, az)
print(f"Implied roll  = {np.rad2deg(roll_rad):+.3f} deg")
print(f"Implied pitch = {np.rad2deg(pitch_rad):+.3f} deg")
# These are non-zero because the IMU was not perfectly level. That tilt
# also leaks into the X/Y accelerometer means above, which is why we
# can't read the X/Y "sensor bias" directly off the static mean.


# %%
# The header doesn't say if gyro is in deg/s or rad/s. Quick heuristic:
# a stationary IMU sees earth rotation (about 0.004 deg/s = 7.3e-5 rad/s)
# plus its own bias and noise. If the magnitudes we see are ~0.1, that's
# nowhere near earth rotation in rad/s, so it's almost certainly deg/s.
gyro_mag = np.linalg.norm(stats.loc[GYRO_COLS, "mean"].values)
gyro_unit = "deg/s (likely)" if gyro_mag > 0.01 else "rad/s (likely)"
print(f"\n|gyro mean| = {gyro_mag:.6f}  -> unit guess: {gyro_unit}")
# TODO: confirm with whoever generated the dataset. For now we treat
# everything as deg/s and convert on the consumer side.


# %%
# Plot 1: the raw 6 channels over the whole 45 min static window.
# Useful to eyeball whether anything jumps or drifts dramatically.
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for col in GYRO_COLS:
    axes[0].plot(df["t_rel"], df[col], label=col, lw=0.5)
axes[0].set_ylabel("Gyro [unit per spec]")
axes[0].legend(loc="upper right")
axes[0].set_title(f"Static IMU, duration {duration_s:.0f} s, fs={fs:.1f} Hz")
axes[0].grid(alpha=0.3)

for col in ACCEL_COLS:
    axes[1].plot(df["t_rel"], df[col], label=col, lw=0.5)
axes[1].set_ylabel("Accel [m/s^2]")
axes[1].set_xlabel("Time [s]")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "static_imu_timeseries.png", dpi=120)
print(f"\nSaved {FIG_DIR / 'static_imu_timeseries.png'}")


# %%
# Plot 2: histogram per axis. We're looking to see roughly Gaussian
# distributions. If they're bimodal or skewed something is off.
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes[0], GYRO_COLS):
    ax.hist(df[col], bins=80, density=True, alpha=0.7)
    ax.axvline(df[col].mean(), color="red", lw=1, label=f"mean={df[col].mean():+.4f}")
    ax.set_title(col)
    ax.legend()
    ax.grid(alpha=0.3)
for ax, col in zip(axes[1], ACCEL_COLS):
    ax.hist(df[col], bins=80, density=True, alpha=0.7)
    ax.axvline(df[col].mean(), color="red", lw=1, label=f"mean={df[col].mean():+.4f}")
    ax.set_title(col)
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle("Static IMU per-axis histograms")
fig.tight_layout()
fig.savefig(FIG_DIR / "static_imu_histogram.png", dpi=120)
print(f"Saved {FIG_DIR / 'static_imu_histogram.png'}")


# %%
# Plot 3: rolling mean per axis. Each channel gets its own panel so the
# slow bias variations are visible regardless of the channel's absolute
# value. Useful to see whether the bias is creeping over the 45-minute
# window. Allan variance below tells the same story properly, but this
# plot is easier to read at a glance.
window = int(30 * fs)

fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)

for ax, col in zip(axes[0], GYRO_COLS):
    rolling = df[col].rolling(window).mean()
    ax.plot(df["t_rel"], rolling, lw=0.8)
    ax.axhline(df[col].mean(), color="red", lw=0.5, linestyle="--",
               label=f"overall mean = {df[col].mean():+.4f}")
    ax.set_title(col)
    ax.set_ylabel("deg/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

for ax, col in zip(axes[1], ACCEL_COLS):
    rolling = df[col].rolling(window).mean()
    ax.plot(df["t_rel"], rolling, lw=0.8)
    ax.axhline(df[col].mean(), color="red", lw=0.5, linestyle="--",
               label=f"overall mean = {df[col].mean():+.4f}")
    ax.set_title(col)
    ax.set_ylabel("m/s$^2$")
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

fig.suptitle(f"30-second rolling mean per axis (window = {window} samples)")
fig.tight_layout()
fig.savefig(FIG_DIR / "static_imu_rolling_mean.png", dpi=120)
print(f"Saved {FIG_DIR / 'static_imu_rolling_mean.png'}")


# %%
# Allan variance is the standard way of pulling IMU noise specs from data.
# Reading it off the log-log plot:
#   short tau, slope -1/2  ->  white noise          (gives us ARW / VRW)
#   bottom of the curve     ->  bias instability
#   long tau, slope +1/2    ->  rate random walk
#
# We use the overlapping form because we have plenty of samples and it
# gives cleaner curves. Standard formula in terms of the integrated
# signal theta = cumsum(x) * dt:
#     sigma^2(tau) = 1 / (2 * tau^2 * (N - 2m))
#                    * sum_k (theta[k+2m] - 2*theta[k+m] + theta[k])^2
def allan_deviation(x: np.ndarray, dt: float, n_taus: int = 60):
    """Overlapping Allan deviation. Input: raw signal + dt. Output: (taus, sigmas)."""
    n = len(x)
    # log-spaced cluster sizes m = tau / dt. Cap at N/3 so we still have a
    # decent number of samples per cluster (anything more is statistically junk).
    m_max = n // 3
    m_arr = np.unique(np.round(np.logspace(0, np.log10(m_max), n_taus)).astype(int))
    m_arr = m_arr[m_arr >= 1]
    theta = np.cumsum(x) * dt
    taus, sigmas = [], []
    for m in m_arr:
        if 2 * m >= n:
            break
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        var = (d ** 2).sum() / (2 * (m * dt) ** 2 * (n - 2 * m))
        taus.append(m * dt)
        sigmas.append(np.sqrt(var))
    return np.array(taus), np.array(sigmas)


dt = 1.0 / fs
allan_results = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for col in GYRO_COLS:
    taus, sig = allan_deviation(df[col].values, dt)
    axes[0].loglog(taus, sig, label=col)
    allan_results[col] = {"tau_s": taus.tolist(), "sigma": sig.tolist()}
axes[0].set_xlabel("tau [s]")
axes[0].set_ylabel("Allan deviation sigma_A(tau)  [gyro unit]")
axes[0].set_title("Gyro Allan deviation")
axes[0].grid(which="both", alpha=0.3)
axes[0].legend()

for col in ACCEL_COLS:
    taus, sig = allan_deviation(df[col].values, dt)
    axes[1].loglog(taus, sig, label=col)
    allan_results[col] = {"tau_s": taus.tolist(), "sigma": sig.tolist()}
axes[1].set_xlabel("tau [s]")
axes[1].set_ylabel("Allan deviation sigma_A(tau)  [m/s^2]")
axes[1].set_title("Accelerometer Allan deviation")
axes[1].grid(which="both", alpha=0.3)
axes[1].legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "static_imu_allan.png", dpi=120)
print(f"\nSaved {FIG_DIR / 'static_imu_allan.png'}")


# %%
# Pull the standard noise numbers off each Allan curve.
#   ARW / VRW: sigma_A read at tau = 1 s. Reported as deg/sqrt(hr) for gyros
#              and m/s/sqrt(hr) for accelerometers.
#   bias instability: minimum of the curve, scaled by ~1.5 (IEEE 952-1997 conv.)
def extract_noise_params(taus, sigmas, sensor_kind: str):
    idx_1s = int(np.argmin(np.abs(taus - 1.0)))
    sig_1s = float(sigmas[idx_1s])
    bi = float(sigmas.min()) * (2 * np.log(2) / np.pi) ** -0.5  # ~ x 1.50
    tau_bi = float(taus[int(np.argmin(sigmas))])
    out = {"sigma_at_1s": sig_1s, "bias_instability": bi, "tau_at_min_s": tau_bi}
    if sensor_kind == "gyro":
        out["arw_deg_per_sqrt_hr"] = sig_1s * 60.0  # deg/s/sqrt(Hz) -> deg/sqrt(hr)
        out["bias_instability_deg_per_hr"] = bi * 3600.0
    else:
        out["vrw_m_per_s_per_sqrt_hr"] = sig_1s * 60.0
        out["bias_instability_mg"] = bi / 9.80665 * 1000.0
    return out


noise_params = {}
print("\nAllan-derived noise parameters:")
for col in GYRO_COLS:
    res = allan_results[col]
    noise_params[col] = extract_noise_params(np.array(res["tau_s"]),
                                             np.array(res["sigma"]), "gyro")
    p = noise_params[col]
    print(f"  {col}: sigma@1s={p['sigma_at_1s']:.5f}  "
          f"ARW={p['arw_deg_per_sqrt_hr']:.4f} deg/sqrt(hr)  "
          f"BI={p['bias_instability_deg_per_hr']:.3f} deg/hr "
          f"@ tau={p['tau_at_min_s']:.1f}s")
for col in ACCEL_COLS:
    res = allan_results[col]
    noise_params[col] = extract_noise_params(np.array(res["tau_s"]),
                                             np.array(res["sigma"]), "accel")
    p = noise_params[col]
    print(f"  {col}: sigma@1s={p['sigma_at_1s']:.5f}  "
          f"VRW={p['vrw_m_per_s_per_sqrt_hr']:.4f} m/s/sqrt(hr)  "
          f"BI={p['bias_instability_mg']:.3f} mg "
          f"@ tau={p['tau_at_min_s']:.1f}s")


# %%
# Dump everything into one JSON. This is the file Kerick should load when
# building Q and the initial bias state covariance. Anyone else who needs
# the static-attitude angles or the gravity check can read from here too.
calibration = {
    "source_file": str(DATA_PATH.relative_to(REPO_ROOT)),
    "n_samples": int(n),
    "duration_s": float(duration_s),
    "sample_rate_hz": float(fs),
    "gyro_unit_guess": gyro_unit,
    "gravity_magnitude_measured": float(g_meas),
    "static_attitude_from_gravity": {
        "roll_deg": float(np.rad2deg(roll_rad)),
        "pitch_deg": float(np.rad2deg(pitch_rad)),
    },
    "bias": {col: float(stats.loc[col, "mean"]) for col in GYRO_COLS + ACCEL_COLS},
    "noise_std": {col: float(stats.loc[col, "std"]) for col in GYRO_COLS + ACCEL_COLS},
    "min": {col: float(stats.loc[col, "min"]) for col in GYRO_COLS + ACCEL_COLS},
    "max": {col: float(stats.loc[col, "max"]) for col in GYRO_COLS + ACCEL_COLS},
    "allan_noise_params": noise_params,
}

calib_path = OUT_DIR / "imu_calibration.json"
calib_path.write_text(json.dumps(calibration, indent=2))
print(f"\nSaved {calib_path}")
print(json.dumps(calibration, indent=2))

# %%
# Simple report table from our IMU bias analysis

summary_rows = []

for col in GYRO_COLS:
    summary_rows.append({
        "Quantity": col,
        "Mean": stats.loc[col, "mean"],
        "Std": stats.loc[col, "std"],
        "Allan metric": (
            f"ARW={noise_params[col]['arw_deg_per_sqrt_hr']:.3f} deg/sqrt(hr)"
        ),
        "Interpretation": "Approximate gyro bias/noise level"
    })

for col in ACCEL_COLS:
    summary_rows.append({
        "Quantity": col,
        "Mean": stats.loc[col, "mean"],
        "Std": stats.loc[col, "std"],
        "Allan metric": (
            f"VRW={noise_params[col]['vrw_m_per_s_per_sqrt_hr']:.3f} m/s/sqrt(hr)"
        ),
        "Interpretation": "Gravity projection + accelerometer bias/noise"
    })

summary_table = pd.DataFrame(summary_rows)

print("\nIMU CHARACTERIZATION SUMMARY")
print(summary_table)

summary_table.to_csv(
    OUT_DIR / "imu_characterization_summary.csv",
    index=False
)

summary_table.to_latex(
    OUT_DIR / "imu_characterization_summary.tex",
    index=False,
    float_format="%.5g",
    caption="Static IMU characterization obtained from the stationary recording.",
    label="tab:imu_characterization"
)

print(f"\nSaved {OUT_DIR / 'imu_characterization_summary.tex'}")