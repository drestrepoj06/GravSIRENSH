"""
Plotting the training perturbations data
jhonr
"""

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_path = os.path.join(base_dir, 'Data', 'Samples_2190_5M_r0.parquet')

outputs_dir = os.path.join(base_dir, "Outputs")
figures_dir = os.path.join(outputs_dir, "Figures")
os.makedirs(figures_dir, exist_ok=True)

sample_df = pd.read_parquet(data_path)

lon_grid = np.linspace(0, 360, 720)
lat_grid = np.linspace(-90, 90, 361)
Lon, Lat = np.meshgrid(lon_grid, lat_grid)

points = np.vstack((sample_df['lon'], sample_df['lat'])).T
values = sample_df['dg_total_mGal']

grid_total = griddata(points, values, (Lon, Lat), method='linear')
mask = np.isnan(grid_total)
if np.any(mask):
    grid_total[mask] = griddata(points, values, (Lon[mask], Lat[mask]), method='nearest')

proj = ccrs.PlateCarree(central_longitude=180)
fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={'projection': proj})
ax.set_global()

gl = ax.gridlines(draw_labels=True, linewidth=0, color='none')
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True

im = ax.pcolormesh(
    Lon, Lat, grid_total,
    cmap='viridis',
    shading='auto',
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, aspect=50)
cbar.set_label('mGal')
ax.set_title('2190-2 (EGM2008)', fontsize=13, pad=12)

plt.tight_layout()

figure_path = os.path.join(figures_dir, "Training_perturbations.png")
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {figure_path}")

plt.show()
