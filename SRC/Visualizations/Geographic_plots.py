"""
Plotting geographically the training, test and predictions data
jhonr
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import glob


class GravityDataPlotter:
    def __init__(self, data_path, output_dir=None, predictions_dir=None):
        self.data_path = data_path
        self.sample_df = pd.read_parquet(data_path)
        self.filename = os.path.basename(data_path)

        if output_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(data_path), '..', 'Outputs', 'Figures'))
            os.makedirs(base_dir, exist_ok=True)
            self.output_dir = base_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir

        self.predictions_dir = predictions_dir or os.path.abspath(
            os.path.join(os.path.dirname(data_path), '..', 'Outputs', 'Predictions')
        )

        # Extract metadata
        self.lmax, self.lmax_base, self.n_samples, self.altitude, self.mode = self._parse_filename(self.filename)

        # Check for matching predictions
        self.preds_path = self._find_predictions_file()
        self.has_predictions = False
        if self.preds_path:
            self._load_predictions()

    def _parse_filename(self, filename):
        name = filename.replace(".parquet", "")
        parts = name.split("_")

        lmax_part = parts[1] if len(parts) > 1 else ""
        if "-" in lmax_part:
            lmax_full, lmax_base = lmax_part.split("-")
            lmax_full, lmax_base = int(lmax_full), int(lmax_base)
        else:
            lmax_full, lmax_base = int(lmax_part), None

        n_samples = parts[2] if len(parts) > 2 else ""
        altitude = parts[3].replace("r", "") if len(parts) > 3 else "0"
        mode = parts[4] if len(parts) > 4 else ""
        return lmax_full, lmax_base, n_samples, altitude, mode

    def _generate_title(self):
        if self.lmax_base is not None:
            return f"EGM2008 Δg (Lmax={self.lmax}, base={self.lmax_base})"
        else:
            return f"EGM2008 Δg (Lmax={self.lmax})"

    def _find_predictions_file(self):
        pattern = os.path.join(self.predictions_dir, f"*_preds*.npy")
        matches = glob.glob(pattern)

        if matches:
            latest = max(matches, key=os.path.getmtime)
            print(f"📂 Found predictions file: {os.path.basename(latest)}")
            return latest
        else:
            print("⚠️ No predictions file found for this dataset.")
            return None

    def _load_predictions(self):
        preds = np.load(self.preds_path)
        if len(preds) != len(self.sample_df):
            print("⚠️ Prediction file length does not match test data.")
            return
        self.sample_df["predicted_dg_total_mGal"] = preds
        self.sample_df["error_mGal"] = preds - self.sample_df["dg_total_mGal"]
        self.has_predictions = True
        print(f"✅ Loaded {len(preds):,} predictions.")

    def _make_grid(self, value_col):
        lon_grid = np.linspace(0, 360, 720)
        lat_grid = np.linspace(-90, 90, 361)
        Lon, Lat = np.meshgrid(lon_grid, lat_grid)
        points = np.vstack((self.sample_df["lon"], self.sample_df["lat"])).T
        values = self.sample_df[value_col]
        grid = griddata(points, values, (Lon, Lat), method="linear")
        mask = np.isnan(grid)
        if np.any(mask):
            grid[mask] = griddata(points, values, (Lon[mask], Lat[mask]), method="nearest")
        return Lon, Lat, grid

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

        ax.set_title(self._generate_title(), fontsize=13, pad=12)

        output_filename = (
                f"Map_L{self.lmax}"
                + (f"-{self.lmax_base}" if self.lmax_base is not None else "")
                + f"_{self.mode}.png"
        )
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved: {output_path}")
        plt.close()

    def plot_density(self, axis="lat"):
        """Plots sampling density along latitude or longitude."""
        if self.mode == "preds":
            return  # skip density plots for predictions

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

        suffix = f"L{self.lmax}" + (f"-{self.lmax_base}" if self.lmax_base is not None else "")
        output_path = os.path.join(self.output_dir, f"Density_{axis}_{suffix}_{self.mode}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Density figure saved: {output_path}")
        plt.close()

    def plot_scatter(self, color_by="lat", s=0.5, alpha=0.8, cmap="viridis"):
        """Scatter plot of sample distribution or Δg values."""
        import re

        # ------------------------------------------------------------------
        # Case 1: Regular scatter plot (train/test)
        # ------------------------------------------------------------------
        if not self.has_predictions:
            plt.figure(figsize=(10, 5))
            color_data = (
                self.sample_df[color_by]
                if color_by in self.sample_df.columns
                else self.sample_df["lat"]
            )

            sc = plt.scatter(
                self.sample_df["lon"],
                self.sample_df["lat"],
                c=color_data,
                s=s,
                alpha=alpha,
                cmap=cmap,
            )

            plt.xlabel("Longitude (°)")
            plt.ylabel("Latitude (°)")
            plt.title(f"Sample Distribution ({self.mode}, Lmax={self.lmax}, base={self.lmax_base})")
            plt.colorbar(sc, label=color_by)
            plt.tight_layout()

            suffix = f"L{self.lmax}" + (f"-{self.lmax_base}" if self.lmax_base is not None else "")
            output_path = os.path.join(self.output_dir, f"Scatter_{suffix}_{self.mode}_{color_by}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✅ Scatter figure saved: {output_path}")
            plt.close()
            return

        # ------------------------------------------------------------------
        # Case 2: Comparison scatter map (predicted vs true)
        # ------------------------------------------------------------------
        cols = ["dg_total_mGal", "predicted_dg_total_mGal"]
        titles = ["True Δg", "Predicted Δg"]

        # Try to extract lmax and timestamp from predictions filename
        preds_name = os.path.basename(self.preds_path)
        match_lmax = re.search(r"lmax(\d+)", preds_name)
        match_time = re.search(r"(\d{8}_\d{6})", preds_name)
        model_lmax = match_lmax.group(1) if match_lmax else "unknown"
        timestamp = match_time.group(1) if match_time else "unknown"

        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(12, 5),
                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        if ncols == 1:
            axes = [axes]

        # Common color scale for better comparison
        vmin = min(self.sample_df["dg_total_mGal"].min(), self.sample_df["predicted_dg_total_mGal"].min())
        vmax = max(self.sample_df["dg_total_mGal"].max(), self.sample_df["predicted_dg_total_mGal"].max())

        for ax, col, title in zip(axes, cols, titles):
            sc = ax.scatter(
                self.sample_df["lon"],
                self.sample_df["lat"],
                c=self.sample_df[col],
                s=s,
                alpha=alpha,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                vmin=vmin, vmax=vmax
            )
            ax.set_global()
            gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = True
            gl.left_labels = True
            cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.04, aspect=40)
            cbar.set_label("mGal")
            ax.set_title(title, fontsize=11, pad=10)

        # Output filename uses preds info
        output_filename = (
            f"Scatter_Comparison_modelL{model_lmax}_{timestamp}_"
            f"vs_dataL{self.lmax}_{self.mode}.png"
        )
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Scatter comparison figure saved: {output_path}")
        plt.close()

        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        sc = ax.scatter(
            self.sample_df["lon"],
            self.sample_df["lat"],
            c=self.sample_df["predicted_dg_total_mGal"],
            s=s,
            alpha=alpha,
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )
        ax.set_global()
        gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        cbar = plt.colorbar(sc, orientation="horizontal", pad=0.04, aspect=40)
        cbar.set_label("mGal")
        ax.set_title(
            f"Predicted Δg (Model L{model_lmax}, Data L{self.lmax}, {self.mode})",
            fontsize=11, pad=10
        )

        output_filename_pred = (
            f"Scatter_Predicted_modelL{model_lmax}_{timestamp}_"
            f"dataL{self.lmax}_{self.mode}.png"
        )
        output_path_pred = os.path.join(self.output_dir, output_filename_pred)
        plt.savefig(output_path_pred, dpi=300, bbox_inches="tight")
        print(f"✅ Separate predictions figure saved: {output_path_pred}")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(
            self.sample_df["predicted_dg_total_mGal"],
            bins=100,
            color="steelblue",
            edgecolor="black",
            alpha=0.8
        )
        plt.xlabel("Predicted Δg (mGal)")
        plt.ylabel("Frequency")
        plt.title(
            f"Histogram of Predicted Δg\n(Model L{model_lmax}, Data L{self.lmax}, {self.mode})",
            fontsize=11
        )
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()

        output_filename_hist = (
            f"Histogram_Predicted_modelL{model_lmax}_{timestamp}_"
            f"dataL{self.lmax}_{self.mode}.png"
        )
        output_path_hist = os.path.join(self.output_dir, output_filename_hist)
        plt.savefig(output_path_hist, dpi=300, bbox_inches="tight")
        print(f"✅ Histogram of predictions saved: {output_path_hist}")
        plt.close()

    @classmethod
    def from_latest(cls, data_dir, output_dir=None):
        pattern_train = os.path.join(data_dir, "Samples_*_train.parquet")
        pattern_test = os.path.join(data_dir, "Samples_*_test.parquet")

        train_files = sorted(glob.glob(pattern_train), key=os.path.getmtime, reverse=True)
        test_files = sorted(glob.glob(pattern_test), key=os.path.getmtime, reverse=True)

        if not train_files:
            raise FileNotFoundError(f"No *_train.parquet found in {data_dir}")
        if not test_files:
            raise FileNotFoundError(f"No *_test.parquet found in {data_dir}")

        latest_train = train_files[0]
        latest_test = test_files[0]

        print(f"📂 Found latest train file: {os.path.basename(latest_train)}")
        print(f"📂 Found latest test file:  {os.path.basename(latest_test)}")

        train_plotter = cls(latest_train, output_dir)
        train_plotter.has_predictions = False

        test_plotter = cls(latest_test, output_dir)
        return train_plotter, test_plotter


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Outputs', 'Figures')
    os.makedirs(output_dir, exist_ok=True)

    train_plotter, test_plotter = GravityDataPlotter.from_latest(data_dir, output_dir)

    # Train plots
    #train_plotter.plot_map(value_col="dg_total_mGal")
    #train_plotter.plot_density("lat")
    #train_plotter.plot_density("lon")
    train_plotter.plot_scatter(color_by="dg_total_mGal", s=0.3, alpha=0.6)

    # Test plots (+ predictions if available)
    #test_plotter.plot_map(value_col="dg_total_mGal")  # adds predicted & error maps if found
    #test_plotter.plot_density("lat")
    #test_plotter.plot_density("lon")
    test_plotter.plot_scatter(color_by="dg_total_mGal", s=0.3, alpha=0.6)

    print("\n✅ All plots saved in:", output_dir)


if __name__ == "__main__":
    main()