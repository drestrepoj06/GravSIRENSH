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
import re


class GravityDataPlotter:
    def __init__(self, data_path, output_dir=None, predictions_dir=None, target_type="acceleration"):
        """
        target_type: 'acceleration' or 'potential'
        """
        self.data_path = data_path
        self.sample_df = pd.read_parquet(data_path)
        self.filename = os.path.basename(data_path)
        self.target_type = target_type.lower()

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

        if self.target_type not in ["acceleration", "potential"]:
            if "dg_total_mGal" in self.sample_df.columns:
                self.target_type = "acceleration"
            elif "dV_m2_s2" in self.sample_df.columns:
                self.target_type = "potential"
            else:
                raise ValueError("Cannot determine target type from dataset.")

        # Check for matching predictions
        self.preds_path = self._find_predictions_file()
        self.has_predictions = False
        if self.preds_path:
            self._load_predictions()

    def _fname_suffix(self):
        """Suffix used in filenames: includes Lmax, base (if any), and ΔL."""
        if self.lmax_base is None:
            return f"L{self.lmax}"
        return f"L{self.lmax}-{self.lmax_base}"

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
        symbol = "Δg" if self.target_type == "acceleration" else "ΔV"
        if self.lmax_base is not None:
            return f"EGM2008 {symbol} (Lmax={self.lmax}, base={self.lmax_base})"
        else:
            return f"EGM2008 {symbol} (Lmax={self.lmax})"

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

        if self.target_type == "acceleration":
            true_col = "dg_total_mGal"
            pred_col = "predicted_dg_total_mGal"
            unit = "mGal"
        else:
            true_col = "dV_m2_s2"
            pred_col = "predicted_dV_m2_s2"
            unit = "m²/s²"

        self.sample_df[pred_col] = preds
        self.sample_df[f"error_{unit}"] = preds - self.sample_df[true_col]
        self.has_predictions = True
        print(f"✅ Loaded {len(preds):,} predictions for target '{self.target_type}'.")

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


    def plot_map(self, cmap="viridis"):
        """Interpolated global map of the selected field."""
        if self.target_type == "acceleration":
            value_col = "dg_total_mGal"
            unit = "mGal"
        else:
            value_col = "dV_m2_s2"
            unit = "m²/s²"

        Lon, Lat, grid_total = self._make_grid(value_col)
        proj = ccrs.PlateCarree(central_longitude=180)
        fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={"projection": proj})
        ax.set_global()

        gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

        im = ax.pcolormesh(Lon, Lat, grid_total, cmap=cmap, shading="auto", transform=ccrs.PlateCarree())
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, aspect=50)
        cbar.set_label(unit)
        ax.set_title(self._generate_title(), fontsize=13, pad=12)

        suffix = f"{self._fname_suffix()}_{self.mode}_{self.target_type[:3]}"
        output_filename = f"Map_{suffix}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved: {output_path}")
        plt.close()

    def plot_density(self, axis="lat"):
        """Plots sampling density along latitude or longitude."""
        if self.mode == "preds":
            return  # skip for predictions

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

        suffix = f"{self._fname_suffix()}_{self.mode}"
        output_path = os.path.join(self.output_dir, f"Density_{axis}_{suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Density figure saved: {output_path}")
        plt.close()

    def plot_scatter(self, color_by=None, s=0.5, alpha=0.8, cmap="viridis"):
        """Scatter plot of samples OR predicted vs true + predicted-only map + histogram."""
        if self.target_type == "acceleration":
            true_col = "dg_total_mGal"
            pred_col = "predicted_dg_total_mGal"
            unit = "mGal"
            symbol = "Δg"
        else:
            true_col = "dV_m2_s2"
            pred_col = "predicted_dV_m2_s2"
            unit = "m²/s²"
            symbol = "ΔV"

        if not self.has_predictions:
            plt.figure(figsize=(10, 5))
            color_data = self.sample_df[color_by] if color_by in self.sample_df.columns else self.sample_df[true_col]
            sc = plt.scatter(self.sample_df["lon"], self.sample_df["lat"], c=color_data, s=s, alpha=alpha, cmap=cmap)
            plt.xlabel("Longitude (°)")
            plt.ylabel("Latitude (°)")
            plt.title(f"Sample Distribution ({symbol}, Lmax={self.lmax})")
            plt.colorbar(sc, label=color_by or true_col)
            plt.tight_layout()
            suffix = f"{self._fname_suffix()}_{self.mode}_{self.target_type[:3]}"
            output_path = os.path.join(self.output_dir, f"Scatter_{suffix}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✅ Scatter figure saved: {output_path}")
            plt.close()
            return

        cols = [true_col, pred_col]
        titles = [f"True {symbol}", f"Predicted {symbol}"]

        preds_name = os.path.basename(self.preds_path)
        match_lmax = re.search(r"lmax(\d+)", preds_name)
        match_time = re.search(r"(\d{8}_\d{6})", preds_name)
        model_lmax = match_lmax.group(1) if match_lmax else "unknown"
        timestamp = match_time.group(1) if match_time else "unknown"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        vmin = min(self.sample_df[true_col].min(), self.sample_df[pred_col].min())
        vmax = max(self.sample_df[true_col].max(), self.sample_df[pred_col].max())

        for ax, col, title in zip(axes, cols, titles):
            sc = ax.scatter(self.sample_df["lon"], self.sample_df["lat"],
                            c=self.sample_df[col], s=s, alpha=alpha, cmap=cmap,
                            transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
            ax.set_global()
            gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = True
            gl.left_labels = True
            cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.04, aspect=40)
            cbar.set_label(unit)
            ax.set_title(title, fontsize=11, pad=10)

        comp_suffix = f"{self._fname_suffix()}_modelL{model_lmax}_{timestamp}_{self.target_type}"
        output_path = os.path.join(self.output_dir, f"Scatter_Comparison_{comp_suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Scatter comparison figure saved: {output_path}")
        plt.close()

        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        sc = ax.scatter(self.sample_df["lon"], self.sample_df["lat"],
                        c=self.sample_df[pred_col], s=s, alpha=alpha, cmap=cmap,
                        transform=ccrs.PlateCarree())
        ax.set_global()
        gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        cbar = plt.colorbar(sc, orientation="horizontal", pad=0.04, aspect=40)
        cbar.set_label(unit)
        ax.set_title(f"Predicted {symbol} (Model L{model_lmax}, Data L{self.lmax}, {self.mode})", fontsize=11, pad=10)

        pred_only_suffix = f"{self._fname_suffix()}_modelL{model_lmax}_{timestamp}_{self.target_type}"
        output_path_pred = os.path.join(self.output_dir, f"Scatter_Predicted_{pred_only_suffix}.png")
        plt.savefig(output_path_pred, dpi=300, bbox_inches="tight")
        print(f"✅ Separate predictions figure saved: {output_path_pred}")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(self.sample_df[pred_col], bins=100, color="steelblue", edgecolor="black", alpha=0.8)
        plt.xlabel(f"Predicted {symbol} ({unit})")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Predicted {symbol}\n(Model L{model_lmax}, Data L{self.lmax}, {self.mode})", fontsize=11)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()

        hist_suffix = f"{self._fname_suffix()}_modelL{model_lmax}_{timestamp}_{self.target_type}"
        output_path_hist = os.path.join(self.output_dir, f"Histogram_Predicted_{hist_suffix}.png")
        plt.savefig(output_path_hist, dpi=300, bbox_inches="tight")
        print(f"✅ Histogram of predictions saved: {output_path_hist}")
        plt.close()


    @classmethod
    def from_latest(cls, data_dir, output_dir=None, target_type="acceleration"):
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

        train_plotter = cls(latest_train, output_dir, target_type=target_type)
        train_plotter.has_predictions = False
        test_plotter = cls(latest_test, output_dir, target_type=target_type)
        return train_plotter, test_plotter


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Outputs', 'Figures')
    os.makedirs(output_dir, exist_ok=True)

    # Choose target type here: 'acceleration' or 'potential'
    target_type = "acceleration"

    train_plotter, test_plotter = GravityDataPlotter.from_latest(data_dir, output_dir, target_type=target_type)

    # Example plots
    #train_plotter.plot_map()
    #train_plotter.plot_density("lat")
    #train_plotter.plot_density("lon")
    #train_plotter.plot_scatter(s=0.3, alpha=0.6)

    #test_plotter.plot_map()
    #test_plotter.plot_density("lat")
    #test_plotter.plot_density("lon")
    test_plotter.plot_scatter(s=0.3, alpha=0.6)

    print("\n✅ All plots saved in:", output_dir)


if __name__ == "__main__":
    main()