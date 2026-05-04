import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


def ecef_to_latlon(x, y, z):
    """Convert ECEF to geodetic lat/lon/h (WGS84)"""
    a = 6378137.0          # semi-major axis
    e2 = 6.69437999014e-3  # eccentricity squared
    
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))  # initial estimate
    
    for _ in range(10):  # iterate to converge
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)
    
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h

# --- Load SPP ---
spp = pd.read_csv("output/gnss/SPP_solutions_run2.csv", index_col=0)
spp_lat, spp_lon, _ = ecef_to_latlon(spp['X'].values, spp['Y'].values, spp['Z'].values)

# --- Load Ground Truth ---
import io

with open("data/run2/run2_groundtruth.txt", 'r') as f:
    lines = [line for line in f if line.strip() and line.strip()[0].isdigit()]

gt = pd.read_csv(io.StringIO(''.join(lines)),
                 sep=r'\s+',
                 header=None,
                 names=['GPSTime','X','Y','Z','Heading','Pitch','Roll','VX','VY','VZ','UTCTime'])

print(gt.head())
gt.columns = ['GPSTime','X','Y','Z','Heading','Pitch','Roll','VX','VY','VZ','UTCTime']
gt_lat, gt_lon, _ = ecef_to_latlon(gt['X'].values, gt['Y'].values, gt['Z'].values)

# --- Create GeoDataFrames ---
gdf_spp = gpd.GeoDataFrame(
    {'GPSTime': spp.index},
    geometry=[Point(lon, lat) for lon, lat in zip(spp_lon, spp_lat)],
    crs="EPSG:4326"
)

gdf_gt = gpd.GeoDataFrame(
    {'GPSTime': gt['GPSTime'].values},
    geometry=[Point(lon, lat) for lon, lat in zip(gt_lon, gt_lat)],
    crs="EPSG:4326"
)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 8))
gdf_gt.plot(ax=ax, color='blue', markersize=2, label='Ground Truth')
gdf_spp.plot(ax=ax, color='red', markersize=4, label='SPP Solution')
plt.legend()
plt.title("SPP vs Ground Truth")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

components = ['X', 'Y', 'Z']
colors_spp = ['red', 'green', 'blue']
colors_gt = ['darkred', 'darkgreen', 'darkblue']

# Compare groundtruth and spp solution for each coordinate and time step

for i, (comp, c_spp, c_gt) in enumerate(zip(components, colors_spp, colors_gt)):
    axes[i].plot(spp.index, spp[comp], color=c_spp, linewidth=1, label='SPP', alpha=0.7)
    axes[i].plot(gt['GPSTime'], gt[comp], color=c_gt, linewidth=1, label='Ground Truth', alpha=0.7)
    axes[i].set_ylabel(f'{comp} (m)')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

axes[2].set_xlabel('GPS Time (SoW)')
fig.suptitle('SPP vs Ground Truth - Position Components over Time')
plt.tight_layout()
plt.show()