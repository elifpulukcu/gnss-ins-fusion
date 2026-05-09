"""
Summarize the IMU bias files provided with the dataset.

The files run2_imu_biases.txt, run3_imu_biases.txt, and run4_imu_biases.txt
were provided after the initial dataset release. These are not raw IMU
measurements. They are bias estimates from the post-processed reference
GNSS/INS solution.

We use this script for two things:
    1. Save a full CSV summary with mean/std/min/max for each bias state.
    2. Save a compact LaTeX table with only the mean values for the report.

Important:
    These values are useful as reference bias magnitudes and for comparison
    with our Kalman-filter bias states. They should not be described as
    directly measured sensor biases from the static IMU data.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "imu"
OUT_DIR.mkdir(parents=True, exist_ok=True)


BIAS_COLS = [
    "AccBiasX", "AccBiasY", "AccBiasZ",
    "GyroDriftX", "GyroDriftY", "GyroDriftZ",
]


def load_imu_bias_file(path: Path) -> pd.DataFrame:
    """Load one provided IMU-bias file.

    The files contain a text header followed by numeric rows. Instead of
    relying on a fixed number of header lines, we keep only the rows with the
    seven numeric fields we need:

        GPSTime AccBiasX AccBiasY AccBiasZ GyroDriftX GyroDriftY GyroDriftZ
    """
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 7:
                continue

            try:
                values = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(values)

    if not rows:
        raise ValueError(f"No numeric bias rows found in {path}")

    return pd.DataFrame(
        rows,
        columns=[
            "GPSTime",
            "AccBiasX", "AccBiasY", "AccBiasZ",
            "GyroDriftX", "GyroDriftY", "GyroDriftZ",
        ],
    )


def unit_for(quantity: str) -> str:
    """Return the unit used by the provided bias files."""
    return "m/s^2" if quantity.startswith("AccBias") else "deg/s"


def main() -> None:
    summary_rows = []

    for run in ["run2", "run3", "run4"]:
        path = DATA_DIR / run / f"{run}_imu_biases.txt"
        df = load_imu_bias_file(path)

        for col in BIAS_COLS:
            summary_rows.append({
                "Run": run,
                "Quantity": col,
                "Mean": df[col].mean(),
                "Std": df[col].std(),
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Unit": unit_for(col),
            })

    summary = pd.DataFrame(summary_rows)

    # Full numerical summary, useful for checking the provided bias trajectories.
    csv_path = OUT_DIR / "provided_imu_bias_summary.csv"
    summary.to_csv(csv_path, index=False)

    # Compact report table: mean bias per run.
    compact = summary.pivot(
        index="Quantity",
        columns="Run",
        values="Mean",
    ).reset_index()

    unit_map = {
        "AccBiasX": "m/s$^2$",
        "AccBiasY": "m/s$^2$",
        "AccBiasZ": "m/s$^2$",
        "GyroDriftX": "deg/s",
        "GyroDriftY": "deg/s",
        "GyroDriftZ": "deg/s",
    }
    compact["Unit"] = compact["Quantity"].map(unit_map)

    compact["Quantity"] = compact["Quantity"].replace({
        "AccBiasX": "AccBias\\_X",
        "AccBiasY": "AccBias\\_Y",
        "AccBiasZ": "AccBias\\_Z",
        "GyroDriftX": "GyroDrift\\_X",
        "GyroDriftY": "GyroDrift\\_Y",
        "GyroDriftZ": "GyroDrift\\_Z",
    })

    tex_path = OUT_DIR / "provided_imu_bias_mean_table.tex"
    compact.to_latex(
        tex_path,
        index=False,
        float_format="%.6g",
        escape=False,
        caption=(
            "Mean IMU bias estimates provided from the post-processed "
            "reference GNSS/INS solution."
        ),
        label="tab:provided_imu_biases",
    )

    print("\nProvided reference-bias summary:")
    print(summary)

    print("\nCompact mean table for the report:")
    print(compact)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()