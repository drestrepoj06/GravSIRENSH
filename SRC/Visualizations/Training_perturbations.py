"""
Plotting the training perturbations data
jhonr
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs


class GravityDataPlotter:
    def __init__(self, data_path, output_dir=None):
        self.data_path = data_path
        self.sample_df = pd.read_parquet(data_path)
        self.filename = os.path.basename(data_path)

        # Default output directory
        if output_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(data_path), '..', 'Outputs', 'Figures'))
            os.makedirs(base_dir, exist_ok=True)
            self.output_dir = base_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir

        # Extract metadata from filename
        self.lmax, self.n_samples, self.altitude, self.mode = self._parse_filename(self.filename)

    # ------------------------------------------------------------------
    def _parse_filename(self, filename):
        """
        Extracts key info from filenames like:
        Samples_2190_5M_r0_train.parquet
        """
        parts = filename.replace(".parquet", "").split("_")
        lmax = int(parts[1]) if len(parts) > 1 else None
        n_samples = parts[2] if len(parts) > 2 else ""
        altitude = parts[3].replace("r", "") if len(parts) > 3 else "0"
        mode = parts[4] if len(parts) > 4 else ""
        return lmax, n_samples, altitude, mode

    # ------------------------------------------------------------------
    def _generate_title(self, lmax_base=None):
        if lmax_base is not None:
            return f"EGM2008 Δg (Lmax={self.lmax}, base={lmax_base})"
        else:
            return f"EGM2008 Δg (Lmax={self.lmax})"

    # ------------------------------------------------------------------
    def plot_map(self, value_col="dg_total_mGal", lmax_base=None, cmap="viridis"):
        """Interpolated global map of the selected field."""

        lon_grid = np.linspace(0, 360, 720)
        lat_grid = np.linspace(-90, 90, 361)
        Lon, Lat = np.meshgrid(lon_grid, lat_grid)

        points = np.vstack((self.sample_df["lon"], self.sample_df["lat"])).T
        values = self.sample_df[value_col]

        grid_total = griddata(points, values, (Lon, Lat), method="linear")
        mask = np.isnan(grid_total)
        if np.any(mask):
            grid_total[mask] = griddata(points, values, (Lon[mask], Lat[mask]), method="nearest")

        proj = ccrs.PlateCarree(central_longitude=180)
        fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={"projection": proj})
        ax.set_global()

        gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

        im = ax.pcolormesh(
            Lon, Lat, grid_total,
            cmap=cmap,
            shading="auto",
            transform=ccrs.PlateCarree()
        )

        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, aspect=50)
        cbar.set_label("mGal")

        ax.set_title(self._generate_title(lmax_base), fontsize=13, pad=12)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"Map_L{self.lmax}_{self.mode}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved: {output_path}")
        plt.show()

    # ------------------------------------------------------------------
    def plot_density(self, axis="lat"):
        """Plots sampling density along latitude or longitude."""

        if axis == "lat":
            bins = np.linspace(-90, 90, 181)
            counts, edges = np.histogram(self.sample_df["lat"], bins=bins)
            label = "Latitude (°)"
            color = "teal"
        elif axis == "lon":
            bins = np.linspace(0, 360, 361)
            counts, edges = np.histogram(self.sample_df["lon"], bins=bins)
            label = "Longitude (°)"
            color = "darkorange"
        else:
            raise ValueError("axis must be 'lat' or 'lon'")

        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.figure(figsize=(8, 4))
        plt.bar(centers, counts, width=np.diff(edges), color=color, edgecolor="black", alpha=0.7)
        plt.xlabel(label)
        plt.ylabel("Number of samples")
        plt.title(f"Sampling Density along {label}")
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"Density_{axis}_L{self.lmax}_{self.mode}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Density figure saved: {output_path}")
        plt.show()

    # ------------------------------------------------------------------
    def plot_scatter(self, color_by="lat", s=2):
        """Simple scatter plot to visualize distribution."""
        plt.figure(figsize=(10, 5))
        plt.scatter(
            self.sample_df["lon"], self.sample_df["lat"],
            c=self.sample_df[color_by] if color_by in self.sample_df.columns else self.sample_df["lat"],
            s=s, cmap="viridis"
        )
        plt.xlabel("Longitude (°)")
        plt.ylabel("Latitude (°)")
        plt.title(f"Sample Distribution ({self.mode}, Lmax={self.lmax})")
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"Scatter_L{self.lmax}_{self.mode}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Scatter figure saved: {output_path}")
        plt.show()

