"""
Make comparison tables for the IMU calibration and reference bias values.

This script is only for analysis/report figures. It does not change the
mechanization or the Kalman filter.

We use it to compare three things:

1. White-noise PSD
   Our Allan-derived short-time noise estimate is compared with the PSD values
   provided in the course email.

2. Bias random-walk PSD
   Our Allan-derived bias-instability estimate is compared with the provided
   bias random-walk PSD values. These two are not exactly the same physical
   quantity, so this table should be interpreted as an order-of-magnitude
   sanity check rather than a strict validation.

3. Initial bias values
   The bias values we use for mechanization initialization are compared with
   the bias states from the post-processed reference GNSS/INS solution.
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

CALIB_PATH = REPO_ROOT / "output" / "imu" / "imu_calibration.json"
INIT_STATES_PATH = REPO_ROOT / "output" / "imu" / "initial_states.json"

REF_BIAS_PATHS = {
    "run2": REPO_ROOT / "data" / "run2" / "run2_imu_biases.txt",
    "run3": REPO_ROOT / "data" / "run3" / "run3_imu_biases.txt",
    "run4": REPO_ROOT / "data" / "run4" / "run4_imu_biases.txt",
}

FIG_DIR = REPO_ROOT / "output" / "imu" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEG2RAD = np.pi / 180.0


# ---------------------------------------------------------------------------
# Reference PSD values from the course email
# ---------------------------------------------------------------------------
# These are the values used in the final Q matrix. Our Allan analysis is kept
# as an independent check, but the filter tuning follows the provided PSDs.
REF_ACCEL_NOISE_PSD = 3.462133832010000e-07
REF_ACCEL_BIAS_PSD = 2.163833645006250e-08
REF_GYRO_NOISE_PSD = 3.046174197867087e-08
REF_GYRO_BIAS_PSD = 2.350443053909789e-09


# ---------------------------------------------------------------------------
# Values from our own IMU characterization
# ---------------------------------------------------------------------------
calib = json.loads(CALIB_PATH.read_text())
allan = calib["allan_noise_params"]
static_bias = calib["bias"]

GYRO_AXES = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
ACCEL_AXES = ["Accel_X", "Accel_Y", "Accel_Z"]

GYRO_COLS = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
ACCEL_COLS = ["Accel_X", "Accel_Y", "Accel_Z"]

SENSOR_NAMES = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]


# Allan values are stored in the units used in the IMU files.
# Gyro values are converted from deg-based units to rad-based units before
# comparing with the PSD values used by the KF.
gyro_white_our = np.array([
    allan[axis]["sigma_at_1s"] ** 2 * DEG2RAD**2
    for axis in GYRO_AXES
])
accel_white_our = np.array([
    allan[axis]["sigma_at_1s"] ** 2
    for axis in ACCEL_AXES
])

gyro_bias_our = np.array([
    allan[axis]["bias_instability"] ** 2 * DEG2RAD**2
    for axis in GYRO_AXES
])
accel_bias_our = np.array([
    allan[axis]["bias_instability"] ** 2
    for axis in ACCEL_AXES
])

gyro_static = np.array([static_bias[axis] for axis in GYRO_AXES])
accel_static = np.array([static_bias[axis] for axis in ACCEL_AXES])


# Per-run bias values that are actually used to initialize mechanization.
init_states = json.loads(INIT_STATES_PATH.read_text())

mech_accel_bias = {
    run: np.array([
        init_states[run]["accel_bias_m_s2"][axis]
        for axis in ACCEL_COLS
    ])
    for run in ("run2", "run3", "run4")
}

mech_gyro_bias = {
    run: np.array([
        init_states[run]["gyro_bias_deg_s"][axis]
        for axis in GYRO_COLS
    ])
    for run in ("run2", "run3", "run4")
}


# ---------------------------------------------------------------------------
# Reference bias trajectories from the provided post-processed solution
# ---------------------------------------------------------------------------
def load_reference_bias_file(path: Path) -> pd.DataFrame:
    """Load one provided IMU-bias file.

    The files include a text header, so we keep only numeric rows with the
    seven expected fields:

        GPSTime AccBiasX AccBiasY AccBiasZ GyroDriftX GyroDriftY GyroDriftZ
    """
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 7:
                continue

            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric reference-bias rows found in {path}")

    return pd.DataFrame(
        rows,
        columns=[
            "GPSTime",
            "AccBiasX", "AccBiasY", "AccBiasZ",
            "GyroDriftX", "GyroDriftY", "GyroDriftZ",
        ],
    )


ref_bias_runs = {
    run: load_reference_bias_file(path)
    for run, path in REF_BIAS_PATHS.items()
}

ref_accel_init = {
    run: df[["AccBiasX", "AccBiasY", "AccBiasZ"]].iloc[0].values
    for run, df in ref_bias_runs.items()
}

ref_gyro_init = {
    run: df[["GyroDriftX", "GyroDriftY", "GyroDriftZ"]].iloc[0].values
    for run, df in ref_bias_runs.items()
}


# ---------------------------------------------------------------------------
# Simple colour scheme for visual report tables
# ---------------------------------------------------------------------------
GOOD_COLOR = "#d4edda"
WARN_COLOR = "#fff3cd"
BAD_COLOR = "#f8d7da"
HEAD_COLOR = "#343a40"
ALT_COLOR = "#f8f9fa"
WHITE = "#ffffff"


def percent_difference(ours: float, ref: float) -> float:
    """Percent difference using the reference value as denominator."""
    return 100.0 * (ours - ref) / ref


def agreement_color(pct_val: float, thresholds=(20, 50)) -> str:
    """Colour-code agreement based on absolute percent difference."""
    low, high = thresholds

    if abs(pct_val) <= low:
        return GOOD_COLOR
    if abs(pct_val) <= high:
        return WARN_COLOR
    return BAD_COLOR


def agreement_label(pct_val: float, thresholds=(20, 50)) -> str:
    """Text label matching the colour thresholds."""
    low, high = thresholds

    if abs(pct_val) <= low:
        return "Good"
    if abs(pct_val) <= high:
        return "Moderate"
    return "Large"


def draw_table(ax, col_labels, row_data, col_widths, title):
    """Draw a compact table as a Matplotlib figure.

    row_data is a list of:
        (row_label, [cell_texts], [cell_colours])
    """
    ax.axis("off")

    x0, y0 = 0.0, 0.92
    row_h = 0.10

    total_w = sum(col_widths)
    col_x = [
        x0 + sum(col_widths[:i]) / total_w
        for i in range(len(col_labels))
    ]
    col_w_n = [w / total_w for w in col_widths]

    def cell(
        x,
        y,
        w,
        h,
        text,
        bg,
        fc="black",
        fontsize=9,
        bold=False,
        align="center",
    ):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y - h),
                w - 0.004,
                h - 0.004,
                boxstyle="square,pad=0",
                linewidth=0.5,
                edgecolor="#aaaaaa",
                facecolor=bg,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

        ax.text(
            x + w * (0.06 if align == "left" else 0.5),
            y - h / 2,
            text,
            transform=ax.transAxes,
            ha=align,
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=fc,
        )

    for label, w, x in zip(col_labels, col_w_n, col_x):
        cell(
            x,
            y0,
            w,
            row_h,
            label,
            HEAD_COLOR,
            fc="white",
            fontsize=8.5,
            bold=True,
        )

    for i, (row_label, texts, colors) in enumerate(row_data):
        y = y0 - (i + 1) * row_h
        row_bg = ALT_COLOR if i % 2 == 0 else WHITE

        cell(
            col_x[0],
            y,
            col_w_n[0],
            row_h,
            row_label,
            row_bg,
            fontsize=9,
            bold=True,
            align="left",
        )

        for j, (txt, color) in enumerate(zip(texts, colors), start=1):
            cell(col_x[j], y, col_w_n[j], row_h, txt, color, fontsize=9)

    ax.text(
        0.5,
        y0 + 0.05,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#212529",
    )


def add_agreement_legend(fig):
    """Add the small colour legend under each table."""
    lax = fig.add_axes([0.05, 0.01, 0.90, 0.05])
    lax.axis("off")
    lax.legend(
        handles=[
            mpatches.Patch(
                facecolor=GOOD_COLOR,
                edgecolor="#aaaaaa",
                label="Good agreement",
            ),
            mpatches.Patch(
                facecolor=WARN_COLOR,
                edgecolor="#aaaaaa",
                label="Moderate difference",
            ),
            mpatches.Patch(
                facecolor=BAD_COLOR,
                edgecolor="#aaaaaa",
                label="Large difference",
            ),
        ],
        loc="center",
        ncol=3,
        fontsize=9,
        frameon=False,
    )


# ---------------------------------------------------------------------------
# Table 1: white-noise PSD
# ---------------------------------------------------------------------------
def save_white_noise_table() -> None:
    """Compare Allan-derived white-noise magnitude with provided PSD values."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_position([0.03, 0.12, 0.94, 0.80])

    col_labels = [
        "Sensor axis",
        "Allan-derived estimate\nsigma_A(1 s)^2",
        "Provided value\nwhite-noise PSD",
        "Difference",
        "Agreement",
    ]
    col_widths = [1.5, 3.2, 3.2, 1.5, 1.5]

    our_values = list(accel_white_our) + list(gyro_white_our)
    ref_values = [REF_ACCEL_NOISE_PSD] * 3 + [REF_GYRO_NOISE_PSD] * 3
    units = ["m^2/s^3"] * 3 + ["rad^2/s^3"] * 3

    rows = []

    for name, ours, ref, unit in zip(SENSOR_NAMES, our_values, ref_values, units):
        pct = percent_difference(ours, ref)
        color = agreement_color(pct)
        label = agreement_label(pct)

        rows.append((
            name,
            [
                f"{ours:.4e}  [{unit}]",
                f"{ref:.4e}  [{unit}]",
                f"{pct:+.1f}%",
                label,
            ],
            [WHITE, WHITE, color, color],
        ))

    draw_table(
        ax,
        col_labels,
        rows,
        col_widths,
        "Table 1 - White-noise PSD sanity check",
    )
    add_agreement_legend(fig)

    out = FIG_DIR / "table1_white_noise_psd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Table 2: bias random-walk PSD
# ---------------------------------------------------------------------------
def save_bias_random_walk_table() -> None:
    """Compare Allan bias-instability scale with provided bias PSD values.

    This is only an order-of-magnitude check. Bias instability and bias
    random-walk PSD are related to long-term bias behaviour, but they are not
    exactly the same parameter.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_position([0.03, 0.12, 0.94, 0.80])

    col_labels = [
        "Sensor axis",
        "Allan bias scale\nbias instability^2",
        "Provided value\nbias random-walk PSD",
        "Difference",
        "Agreement",
    ]
    col_widths = [1.5, 3.2, 3.2, 1.5, 1.5]

    our_values = list(accel_bias_our) + list(gyro_bias_our)
    ref_values = [REF_ACCEL_BIAS_PSD] * 3 + [REF_GYRO_BIAS_PSD] * 3
    units = ["m^2/s^5"] * 3 + ["rad^2/s^5"] * 3

    rows = []

    for name, ours, ref, unit in zip(SENSOR_NAMES, our_values, ref_values, units):
        pct = percent_difference(ours, ref)
        color = agreement_color(pct, thresholds=(30, 100))
        label = agreement_label(pct, thresholds=(30, 100))

        rows.append((
            name,
            [
                f"{ours:.4e}  [{unit}]",
                f"{ref:.4e}  [{unit}]",
                f"{pct:+.1f}%",
                label,
            ],
            [WHITE, WHITE, color, color],
        ))

    draw_table(
        ax,
        col_labels,
        rows,
        col_widths,
        "Table 2 - Bias random-walk PSD sanity check",
    )
    add_agreement_legend(fig)

    out = FIG_DIR / "table2_bias_rw_psd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Table 3: initial bias comparison
# ---------------------------------------------------------------------------
def save_initial_bias_table() -> None:
    """Compare our initial bias values with the provided reference bias states."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_position([0.03, 0.08, 0.94, 0.82])

    col_labels = [
        "Sensor axis",
        "Our static-segment estimate\nused for initialization",
        "Reference run2\nfirst epoch [diff]",
        "Reference run3\nfirst epoch [diff]",
        "Reference run4\nfirst epoch [diff]",
    ]
    col_widths = [1.5, 3.0, 2.2, 2.2, 2.2]

    units = ["m/s^2"] * 3 + ["deg/s"] * 3
    rows = []

    for i, (name, unit) in enumerate(zip(SENSOR_NAMES, units)):
        is_accel = i < 3
        axis_idx = i % 3

        run_cells = []
        run_colors = []
        our_vals = []

        for run in ("run2", "run3", "run4"):
            ours = (
                mech_accel_bias[run][axis_idx]
                if is_accel
                else mech_gyro_bias[run][axis_idx]
            )
            ref = (
                ref_accel_init[run][axis_idx]
                if is_accel
                else ref_gyro_init[run][axis_idx]
            )

            diff = ours - ref
            our_vals.append(ours)

            # Avoid over-penalising tiny reference values near zero.
            denom = max(abs(ref), 1e-6)
            pct = abs(diff) / denom * 100.0

            color = agreement_color(pct, thresholds=(20, 50))
            run_cells.append(f"{ref:+.5f}  [{diff:+.5f}]")
            run_colors.append(color)

        our_str = (
            f"r2: {our_vals[0]:+.5f}\n"
            f"r3: {our_vals[1]:+.5f}\n"
            f"r4: {our_vals[2]:+.5f}  [{unit}]"
        )

        rows.append((name, [our_str] + run_cells, [WHITE] + run_colors))

    draw_table(
        ax,
        col_labels,
        rows,
        col_widths,
        "Table 3 - Initial bias: static segment vs reference solution",
    )
    add_agreement_legend(fig)

    out = FIG_DIR / "table3_initial_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved -> {out}")


def main() -> None:
    save_white_noise_table()
    save_bias_random_walk_table()
    save_initial_bias_table()


if __name__ == "__main__":
    main()