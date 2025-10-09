# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:11:58 2025
@author: jhonr
"""

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import os

# --- Path to Training.parquet ---
# Move one level up from SRC/Visualizations to reach Repository/, then enter Data/
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_path = os.path.join(base_dir, 'Data', 'Training.parquet')

# --- Load data ---
sample_df = pd.read_parquet(data_path)

# --- Grid setup ---
lon_grid = np.linspace(0, 360, 720)
lat_grid = np.linspace(-90, 90, 361)
Lon, Lat = np.meshgrid(lon_grid, lat_grid)

points = np.vstack((sample_df['lon'], sample_df['lat'])).T
values = sample_df['g_total_mGal']

# --- Interpolation ---
grid_total = griddata(points, values, (Lon, Lat), method='linear')
mask = np.isnan(grid_total)
if np.any(mask):
    grid_total[mask] = griddata(points, values, (Lon[mask], Lat[mask]), method='nearest')

# --- Plot ---
proj = ccrs.PlateCarree(central_longitude=180)  # Center map on the Pacific
fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={'projection': proj})
ax.set_global()

# Remove coastlines and gridlines
# ax.coastlines(linewidth=0.8, color='black')  # removed
gl = ax.gridlines(draw_labels=True, linewidth=0, color='none')  # no visible lines

# Show only left and bottom labels
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True

# Plot data
im = ax.pcolormesh(
    Lon, Lat, grid_total,
    cmap='viridis',
    shading='auto',
    transform=ccrs.PlateCarree(),
    vmin=0, vmax=200
)

# Colorbar and title
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, aspect=50)
cbar.set_label('mGal')
ax.set_title('2190-2 (EGM2008, Pacific-centered)', fontsize=13, pad=12)

plt.tight_layout()
plt.show()
