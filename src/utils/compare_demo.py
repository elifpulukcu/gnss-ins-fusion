"""
Small demo: compare 10 sample epochs using coord_frames.py and pymap3d.

The validation scripts already show that both implementations agree,
but here we print and plot a small subset so the results are easier to
inspect visually.

We take 10 epochs from the moving part of run2, convert them with both
methods, and compare the outputs.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm

from coord_frames import R_ned_ecef, ecef_to_llh
from validate_with_groundtruth import load_groundtruth

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = REPO_ROOT / "data" / "run2" / "run2_groundtruth.txt"
OUT = REPO_ROOT / "output" / "utils"
OUT.mkdir(parents=True, exist_ok=True)


# 1. Pick 10 evenly spaced epochs from the moving part of run2.
df = load_groundtruth(RUN_PATH)
xyz_all = df[["X-ECEF", "Y-ECEF", "Z-ECEF"]].values

ref0 = xyz_all[0]
moved = np.linalg.norm(xyz_all - ref0, axis=1) > 0.5
start = int(np.argmax(moved)) if moved.any() else 0

idx = np.linspace(start, len(df) - 1, 10, dtype=int)
sample = df.iloc[idx].reset_index(drop=True)
xyz = sample[["X-ECEF", "Y-ECEF", "Z-ECEF"]].values

print(
    f"First epoch with motion > 0.5 m from start: index {start} "
    f"(t = {df['GPSTime'].iloc[start] - df['GPSTime'].iloc[0]:.1f} s into run)"
)


# 2. Convert with both implementations.
llh_ours = np.array([ecef_to_llh(*p) for p in xyz])
llh_pm = np.array([pm.ecef2geodetic(*p, deg=False) for p in xyz])

# Use the first selected sample as the NED reference.
ref = xyz[0]
ref_lat, ref_lon, ref_h = ecef_to_llh(*ref)

R = R_ned_ecef(ref_lat, ref_lon)
ned_ours = (R @ (xyz - ref).T).T

ned_pm = np.array([
    pm.ecef2ned(*p, ref_lat, ref_lon, ref_h, deg=False)
    for p in xyz
])

diff_llh = np.abs(llh_ours - llh_pm)
diff_ned = np.linalg.norm(ned_ours - ned_pm, axis=1)


# 3. Print a small table to the console.
print(f"Comparing 10 epochs from {RUN_PATH.parent.name}")
print()
print(
    f"{'#':>2}  {'X-ECEF [m]':>14}  {'Y-ECEF [m]':>14}  {'Z-ECEF [m]':>14}   "
    f"{'lat [deg]':>12}  {'lon [deg]':>12}  {'h [m]':>9}   "
    f"{'N [m]':>9}  {'E [m]':>9}  {'D [m]':>9}"
)
print("-" * 145)

for i in range(len(xyz)):
    lat_d = np.rad2deg(llh_ours[i, 0])
    lon_d = np.rad2deg(llh_ours[i, 1])

    print(
        f"{i:>2}  {xyz[i, 0]:>14.3f}  {xyz[i, 1]:>14.3f}  {xyz[i, 2]:>14.3f}   "
        f"{lat_d:>12.8f}  {lon_d:>12.8f}  {llh_ours[i, 2]:>9.4f}   "
        f"{ned_ours[i, 0]:>+9.3f}  {ned_ours[i, 1]:>+9.3f}  {ned_ours[i, 2]:>+9.3f}"
    )

print()
print("Residuals against pymap3d (max over all 10 epochs):")
print(f"  |dlat| max = {diff_llh[:, 0].max():.3e} rad")
print(f"  |dlon| max = {diff_llh[:, 1].max():.3e} rad")
print(f"  |dh|   max = {diff_llh[:, 2].max():.3e} m")
print(f"  |dNED| max = {diff_ned.max():.3e} m")


# 4. Save a Markdown table for the report.
md = []

md.append("# coord_frames.py vs pymap3d - 10 sample epochs from run2")
md.append("")
md.append(f"Source: `{RUN_PATH.relative_to(REPO_ROOT)}`")
md.append(
    f"Reference for NED: epoch 0 "
    f"(lat={np.rad2deg(ref_lat):.6f} deg, lon={np.rad2deg(ref_lon):.6f} deg)."
)
md.append("")
md.append(
    "| # | X-ECEF [m] | Y-ECEF [m] | Z-ECEF [m] "
    "| lat ours [deg] | lat pm [deg] | \\|dlat\\| [rad] "
    "| h ours [m] | h pm [m] | \\|dh\\| [m] "
    "| NED ours (N,E,D) [m] | NED pm (N,E,D) [m] | \\|dNED\\| [m] |"
)
md.append(
    "|---|-----------:|-----------:|-----------:"
    "|---------------:|-------------:|---------------:"
    "|-----------:|---------:|----------:"
    "|---------------------:|-------------------:|-------------:|"
)

for i in range(len(xyz)):
    lat_o = np.rad2deg(llh_ours[i, 0])
    lat_p = np.rad2deg(llh_pm[i, 0])

    n_o, e_o, d_o = ned_ours[i]
    n_p, e_p, d_p = ned_pm[i]

    md.append(
        f"| {i} | {xyz[i, 0]:.3f} | {xyz[i, 1]:.3f} | {xyz[i, 2]:.3f} "
        f"| {lat_o:.10f} | {lat_p:.10f} | {diff_llh[i, 0]:.2e} "
        f"| {llh_ours[i, 2]:.4f} | {llh_pm[i, 2]:.4f} | {diff_llh[i, 2]:.2e} "
        f"| ({n_o:+.3f}, {e_o:+.3f}, {d_o:+.3f}) "
        f"| ({n_p:+.3f}, {e_p:+.3f}, {d_p:+.3f}) "
        f"| {diff_ned[i]:.2e} |"
    )

md.append("")
md.append("**Residual summary** (max over 10 epochs):")
md.append(
    f"- max |dlat| = {diff_llh[:, 0].max():.3e} rad "
    f"(~{np.rad2deg(diff_llh[:, 0].max()):.2e} deg)"
)
md.append(f"- max |dlon| = {diff_llh[:, 1].max():.3e} rad")
md.append(f"- max |dh|   = {diff_llh[:, 2].max():.3e} m")
md.append(f"- max |dNED| = {diff_ned.max():.3e} m")
md.append("")
md.append(
    "The residuals are very small, indicating that the two implementations "
    "agree numerically."
)

(OUT / "comparison_table.md").write_text("\n".join(md))
print(f"\nSaved {OUT / 'comparison_table.md'}")


# 5. Save a figure: NED trajectory and residuals.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(
    ned_ours[:, 1],
    ned_ours[:, 0],
    "-",
    color="C0",
    lw=1.5,
    label="coord_frames.py",
)
ax.scatter(
    ned_pm[:, 1],
    ned_pm[:, 0],
    marker="x",
    color="C1",
    s=80,
    label="pymap3d",
)

for i in range(len(xyz)):
    ax.annotate(
        str(i),
        (ned_ours[i, 1], ned_ours[i, 0]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_title("Sample epochs in the NED frame")
ax.grid(alpha=0.3)
ax.legend()
ax.set_aspect("equal", adjustable="datalim")


ax = axes[1]

# Replace exact zeros with a small value so they are visible on the log scale.
floor = 1e-18

to_plot_lat = np.maximum(diff_llh[:, 0], floor)
to_plot_lon = np.maximum(diff_llh[:, 1], floor)
to_plot_h = np.maximum(diff_llh[:, 2], floor)
to_plot_ned = np.maximum(diff_ned, floor)

x = np.arange(len(xyz))
w = 0.2

ax.bar(x - 1.5 * w, to_plot_lat, width=w, label="|dlat| [rad]", color="C0")
ax.bar(x - 0.5 * w, to_plot_lon, width=w, label="|dlon| [rad]", color="C1")
ax.bar(x + 0.5 * w, to_plot_h, width=w, label="|dh| [m]", color="C2")
ax.bar(x + 1.5 * w, to_plot_ned, width=w, label="|dNED| [m]", color="C3")

ax.axhline(
    1e-15,
    color="grey",
    lw=0.6,
    ls=":",
    label="small numerical error level",
)

ax.set_yscale("log")
ax.set_ylim(floor, 1e-6)
ax.set_xticks(x)
ax.set_xlabel("epoch index")
ax.set_ylabel("|residual| (log scale)")
ax.set_title("Per-epoch residuals against pymap3d reference")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="lower right")

fig.tight_layout()

fig.savefig(OUT / "comparison_residuals.png", dpi=120)
print(f"Saved {OUT / 'comparison_residuals.png'}")