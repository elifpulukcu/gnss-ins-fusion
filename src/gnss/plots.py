"""
GNSS Trajectory Analysis
Visualizes SPP solutions vs ground truth for multiple runs,
each saved as an individual figure overlaid on an OSM basemap.
"""

import io
import warnings
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore")


WGS84        = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"
RUNS         = [2, 3, 4]

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output/gnss")

PLOT_STYLE = {
    "ground_truth": dict(color="#2196F3", markersize=2, label="Ground Truth", zorder=3),
    "spp":          dict(color="#F44336", markersize=4, label="SPP Solution",  zorder=4),
}


def ecef_to_latlon(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECEF coordinates to geodetic lat/lon/height (WGS84).

    Parameters
    ----------
    x, y, z : np.ndarray
        ECEF coordinates in metres.

    Returns
    -------
    lat, lon, h : np.ndarray
        Latitude and longitude in degrees; ellipsoidal height in metres.
    """
    a  = 6_378_137.0       # WGS84 semi-major axis (m)
    e2 = 6.69437999014e-3  # WGS84 first eccentricity squared

    lon = np.arctan2(y, x)
    p   = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))  # Bowring initial estimate

    for _ in range(10):
        N   = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h




def load_spp(run: int) -> gpd.GeoDataFrame:
    """Load SPP solution CSV and return a Web-Mercator GeoDataFrame."""
    path    = OUTPUT_DIR / f"SPP_solutions_run{run}.csv"
    df      = pd.read_csv(path, index_col=0)
    lat, lon, _ = ecef_to_latlon(df["X"].values, df["Y"].values, df["Z"].values)

    return gpd.GeoDataFrame(
        {"GPSTime": df.index},
        geometry=[Point(lo, la) for lo, la in zip(lon, lat)],
        crs=WGS84,
    ).to_crs(WEB_MERCATOR)


def load_ground_truth(run: int) -> gpd.GeoDataFrame:
    """Load ground-truth text file and return a Web-Mercator GeoDataFrame."""
    path = DATA_DIR / f"run{run}" / f"run{run}_groundtruth.txt"

    with path.open() as fh:
        numeric_lines = [l for l in fh if l.strip() and l.strip()[0].isdigit()]

    df = pd.read_csv(
        io.StringIO("".join(numeric_lines)),
        sep=r"\s+",
        header=None,
        names=["GPSTime", "X", "Y", "Z", "Heading", "Pitch", "Roll", "VX", "VY", "VZ", "UTCTime"],
    )

    lat, lon, _ = ecef_to_latlon(df["X"].values, df["Y"].values, df["Z"].values)

    return gpd.GeoDataFrame(
        {"GPSTime": df["GPSTime"].values},
        geometry=[Point(lo, la) for lo, la in zip(lon, lat)],
        crs=WGS84,
    ).to_crs(WEB_MERCATOR)




def plot_run(run: int) -> plt.Figure:
    """
    Create and save a standalone trajectory map for a single run.

    Parameters
    ----------
    run : int
        Run number to load and plot.

    Returns
    -------
    fig : plt.Figure
    """
    print(f"  Loading run {run}…")
    gdf_gt  = load_ground_truth(run)
    gdf_spp = load_spp(run)

    fig, ax = plt.subplots(figsize=(10, 8))

    gdf_gt.plot(ax=ax,  **PLOT_STYLE["ground_truth"])
    gdf_spp.plot(ax=ax, **PLOT_STYLE["spp"])

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")

    ax.set_title(f"SPP Solution vs Ground Truth — Run {run}", fontsize=14, fontweight="bold", pad=12)
    ax.set_axis_off()

    ax.legend(
        loc="lower right",
        fontsize=11,
        markerscale=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    return fig



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        print(f"Generating plot for run {run}…")
        fig      = plot_run(run)
        out_path = OUTPUT_DIR / f"trajectory_map_run{run}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
        plt.show()
        plt.close(fig)  # free memory before next run

    print("Done.")


if __name__ == "__main__":
    main()