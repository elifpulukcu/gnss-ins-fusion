"""
Predict-only check for the IMU-side KF pieces.

This script runs the mechanization and covariance propagation for a short
window of IMU data. There is no GNSS update here. The goal is only to check
that F, Q and the initial covariance behave sensibly together.

What we expect to see:
    position sigma  : grows as velocity and bias uncertainty integrate
    velocity sigma  : grows steadily
    attitude sigma  : grows from gyro noise and coupling terms
    bias sigmas     : grow slowly through their random-walk noise

This is a quick guard against unit mistakes in F or Q. If something is off,
the curves usually either blow up immediately or stay unrealistically flat.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "imu"))
sys.path.insert(0, str(REPO_ROOT / "src" / "kf"))

from dynamics import build_F, discretize  # noqa: E402
from initial import build_initial_state  # noqa: E402
from mechanization import ACCEL_COLS, GYRO_COLS, mechanize_step  # noqa: E402
from noise import build_Q  # noqa: E402


DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "kf"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run(run_name: str = "run2", duration_s: float = 60.0) -> dict:
    print(f"\n--- predict-only check: {run_name} ---")

    state, P = build_initial_state(run_name)
    sigma0 = np.sqrt(np.maximum(np.diag(P), 0.0))

    print(
        f"t0 = {state.t:.2f}, initial sigma heads: "
        f"pos={sigma0[0]:.3f}, "
        f"vel={sigma0[3]:.3f}, "
        f"att={sigma0[6]:.4f}"
    )

    imu_path = DATA_DIR / run_name / f"{run_name}_imu.txt"
    imu = pd.read_csv(imu_path, sep=r"\s+", engine="python")

    imu = imu[
        (imu["Time"] >= state.t)
        & (imu["Time"] < state.t + duration_s)
    ].reset_index(drop=True)

    n_samples = len(imu)
    if n_samples < 2:
        raise RuntimeError(
            f"Only {n_samples} IMU samples found in the requested window "
            f"[t0, t0 + {duration_s}s]."
        )

    times = np.zeros(n_samples)
    sigmas = np.zeros((n_samples, 15))

    times[0] = state.t
    sigmas[0] = sigma0

    imu_times = imu["Time"].values
    gyros = imu[GYRO_COLS].values
    accels = imu[ACCEL_COLS].values

    Q_step = None
    Q_cached_dt: float | None = None

    for i in range(1, n_samples):
        dt = float(imu_times[i] - imu_times[i - 1])

        # Use the current nominal attitude and current bias-corrected
        # specific force for the local linearization.
        f_b = accels[i] - state.bias_a
        F = build_F(state.C_b_e, f_b)
        Phi = discretize(F, dt)

        # Most files have a fixed IMU dt. Reuse Q unless the timestamp step
        # changes, which keeps this loop cheap and still safe for gaps.
        if Q_step is None or abs(dt - (Q_cached_dt or 0.0)) > 1e-9:
            Q_step = build_Q(dt)
            Q_cached_dt = dt

        P = Phi @ P @ Phi.T + Q_step

        # Keep P symmetric after the matrix multiply. This avoids tiny
        # numerical asymmetry showing up in later checks.
        P = 0.5 * (P + P.T)

        state = mechanize_step(state, gyros[i], accels[i], dt)

        times[i] = state.t
        sigmas[i] = np.sqrt(np.maximum(np.diag(P), 0.0))

    elapsed = float(times[-1] - times[0])
    print(f"Propagated {n_samples - 1} IMU steps over {elapsed:.2f} s")

    final_sigma = sigmas[-1]

    print("Final sigma:")
    print(f"  pos    [m]      : {final_sigma[0:3]}")
    print(f"  vel    [m/s]    : {final_sigma[3:6]}")
    print(f"  att    [rad]    : {final_sigma[6:9]}")
    print(f"  bias_a [m/s^2]  : {final_sigma[9:12]}")
    print(f"  bias_g [rad/s]  : {final_sigma[12:15]}")

    blocks = [
        ("position [m]", sigmas[:, 0:3], ["X", "Y", "Z"]),
        ("velocity [m/s]", sigmas[:, 3:6], ["VX", "VY", "VZ"]),
        ("attitude [rad]", sigmas[:, 6:9], ["psi_x", "psi_y", "psi_z"]),
        ("accel bias [m/s^2]", sigmas[:, 9:12], ["bx", "by", "bz"]),
        ("gyro bias [rad/s]", sigmas[:, 12:15], ["bgx", "bgy", "bgz"]),
    ]

    t_rel = times - times[0]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes_flat = axes.flatten()

    for ax, (title, sigma_block, labels) in zip(axes_flat, blocks):
        for axis_idx, label in enumerate(labels):
            ax.plot(t_rel, sigma_block[:, axis_idx], lw=0.8, label=label)

        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("sigma")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    axes_flat[-1].set_visible(False)

    fig.suptitle(f"{run_name}: predict-only KF sigma growth ({elapsed:.0f} s)")
    fig.tight_layout()

    fig_path = FIG_DIR / f"predict_sigma_{run_name}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    print(f"Saved {fig_path}")

    return {
        "run": run_name,
        "duration_s": elapsed,
        "sigma_final": final_sigma,
        "fig_path": fig_path,
    }


if __name__ == "__main__":
    for run_name in ("run2", "run3", "run4"):
        run(run_name, duration_s=60.0)

    print("\nvalidate_predict.py finished.")