"""
Plotting geographically the training, test and predictions data
jhonr
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import glob

class GravityDataPlotter:
    def __init__(
            self,
            data_path,
            output_dir=None,
            predictions_dir=None,
            linear_dir=None,
            linear_type="mag",
            target_type="acceleration"
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.predictions_dir = predictions_dir
        self.linear_dir = linear_dir
        self.linear_type = linear_type
        self.target_type = target_type.lower()

        self.sample_df = pd.read_parquet(data_path)
        self.filename = os.path.basename(data_path)

        if "lat" in self.sample_df.columns:
            mask = np.abs(self.sample_df["lat"].values) < 89.9999
            self.sample_df = self.sample_df[mask].reset_index(drop=True)

        if self.output_dir is None:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(data_path), "..", "Outputs", "Figures")
            )
            os.makedirs(base_dir, exist_ok=True)
            self.output_dir = base_dir
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        if self.predictions_dir is None:
            self.predictions_dir = self.output_dir

        self.lmax, self.lmax_base, self.n_samples, self.altitude, self.mode = \
            self._parse_filename(self.filename)

        self.pred_files = self._find_predictions_file()
        self.nn_preds = {}
        self.nn_errors = {}

        self._build_subset_dataframes()

        if self.pred_files:
            self._load_predictions()
            self.has_predictions = True
        else:
            self.has_predictions = False

        self.linear_available = False
        self.linear_preds = {"model": {}, "equiv": {}}

        if self.linear_dir is not None:
            self._load_linear_predictions()

    def _build_subset_dataframes(self):

        df = self.sample_df

        g = df["dg_total_mGal"].values
        mean_g = g.mean()
        std_g = g.std()

        mask_F = np.abs(g - mean_g) > 2 * std_g
        mask_C = ~mask_F

        self.A_idx = np.arange(len(df))
        self.F_idx = np.where(mask_F)[0]
        self.C_idx = np.where(mask_C)[0]

        self.sample_df_subset = {
            "A": df.iloc[self.A_idx].reset_index(drop=True),
            "F": df.iloc[self.F_idx].reset_index(drop=True),
            "C": df.iloc[self.C_idx].reset_index(drop=True),
        }

    def _fname_suffix(self):
        """Suffix used in filenames: includes Lmax, base (if any), and ΔL."""
        if self.lmax_base is None:
            return f"L{self.lmax}"
        return f"L{self.lmax}-{self.lmax_base}"

    def _load_linear_predictions(self):
        """Load linear predictions for A/F/C for both model-lmax and L_equiv."""

        component = "mag" if self.target_type == "acceleration" else "U"
        fname_base = f"linear_g_{component}" if component != "U" else "linear_U"

        self.linear_preds = {"model": {}, "equiv": {}}
        self.linear_available = False

        # Ensure subset DataFrames exist
        if not hasattr(self, "sample_df_subset"):
            raise RuntimeError("sample_df_subset must be defined before calling _load_linear_predictions().")

        for subset in ["A", "F", "C"]:

            path_model = os.path.join(self.linear_dir, f"{fname_base}_{subset}_model.npy")

            if os.path.exists(path_model):
                arr = np.load(path_model)

                df_subset = self.sample_df_subset[subset]

                if len(arr) == len(df_subset):
                    self.linear_preds["model"][subset] = arr
                    self.linear_available = True
                else:
                    print(
                        f"⚠ Length mismatch for {subset} model-lmax: file={len(arr)}, df={len(df_subset)} — skipping.")

            path_equiv = os.path.join(self.linear_dir, f"{fname_base}_{subset}_equiv.npy")

            if os.path.exists(path_equiv):
                arr = np.load(path_equiv)

                df_subset = self.sample_df_subset[subset]

                if len(arr) == len(df_subset):
                    self.linear_preds["equiv"][subset] = arr
                    self.linear_available = True
                else:
                    print(f"⚠ Length mismatch for {subset} L_equiv: file={len(arr)}, df={len(df_subset)} — skipping.")

        if not self.linear_available:
            print("⚠ No linear predictions available.")

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
        symbol = "Δg" if self.target_type == "acceleration" else "ΔU"
        title = f"EGM2008 {symbol} (Lmax={self.lmax})"
        if self.lmax_base is not None:
            title += f", base={self.lmax_base}"
        if self.has_predictions:
            title += " — Predicted vs True"
        return title

    def _find_predictions_file(self):
        patterns = {
            "potential": "test_results_{subset}_U.npy",
            "acceleration": "test_results_{subset}_*_mag.npy",
            "gradients": "test_results_{subset}_g_mag_grad.npy",
        }

        if self.target_type not in patterns:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        subset_results = {}

        for subset in ["A", "F", "C"]:
            pattern = os.path.join(
                self.predictions_dir,
                patterns[self.target_type].format(subset=subset)
            )
            matches = glob.glob(pattern)

            if matches:
                latest = max(matches, key=os.path.getmtime)
                subset_results[subset] = latest
            else:
                print(f"⚠ No predictions found for subset {subset}")

        if not subset_results:
            print(f"⚠ No predictions files found for target '{self.target_type}' in {self.predictions_dir}")
            return None

        self.pred_files = subset_results
        return subset_results

    def _load_predictions(self):
        if not hasattr(self, "pred_files") or not self.pred_files:
            print("⚠ No prediction files loaded — skipping.")
            return

        self.nn_preds = {}
        self.nn_errors = {}

        if self.target_type == "acceleration":
            true_col = "dg_total_mGal"
            pred_col = "predicted_dg_total_mGal"
        elif self.target_type == "potential":
            true_col = "dU_m2_s2"
            pred_col = "predicted_dU_m2_s2"
        elif self.target_type == "gradients":
            true_col = "dg_total_mGal"
            pred_col = "predicted_g_mag_mGal"
        else:
            raise ValueError(f"Unknown target_type '{self.target_type}'")

        if not hasattr(self, "sample_df_subset"):
            raise RuntimeError("sample_df_subset (A/F/C) must be created before calling _load_predictions().")


        for subset, path in self.pred_files.items():
            preds = np.load(path)

            df_subset = self.sample_df_subset[subset]

            if len(preds) != len(df_subset):
                print(f"⚠ Length mismatch for subset {subset}: preds={len(preds)}, df={len(df_subset)} — skipping.")
                continue

            self.nn_preds[subset] = preds

            true_vals = df_subset[true_col].to_numpy()
            errors = preds - true_vals
            self.nn_errors[subset] = errors

            df_subset[pred_col] = preds

            if self.target_type == "acceleration" or self.target_type == "gradients":
                df_subset["error_mGal"] = errors
            elif self.target_type == "potential":
                df_subset["error_m2s2"] = errors

        self.has_predictions = True

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
            value_col = "dU_m2_s2"
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
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, aspect=50)
        cbar.set_label(unit)
        ax.set_title(self._generate_title(), fontsize=13, pad=12)

        suffix = f"{self._fname_suffix()}_{self.mode}_{self.target_type[:3]}"
        output_filename = f"Map_{suffix}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _iter_subsets(self):
        """Helper to iterate over subsets A/F/C."""
        return ["A", "F", "C"]

    def plot_scatter(self, s=0.5, alpha=0.8, cmap="viridis"):
        """
        Subset-aware plotting:
            - A (All)
            - F (High-perturbation)
            - C (Complement)

        Generates maps + comparison + histograms per subset.
        """

        # Determine true/pred columns based on target type
        if self.target_type == "acceleration":
            true_col = "dg_total_mGal"
            pred_col = "predicted_dg_total_mGal"
            unit = "mGal"
            symbol = "Δg"

        elif self.target_type == "gradients":
            true_col = "dg_total_mGal"
            pred_col = "predicted_g_mag_mGal"
            unit = "mGal"
            symbol = "Δg"

        else:
            true_col = "dU_m2_s2"
            pred_col = "predicted_dU_m2_s2"
            unit = "m²/s²"
            symbol = "ΔU"

        for subset in ["A", "F", "C"]:

            df = self.sample_df_subset[subset]
            subset_dir = os.path.join(self.output_dir, f"Maps_{subset}")
            os.makedirs(subset_dir, exist_ok=True)

            plt.figure(figsize=(10, 5))
            color_data = df[true_col]

            sc = plt.scatter(
                df["lon"], df["lat"],
                c=color_data, s=s, alpha=alpha, cmap=cmap
            )

            plt.xlabel("Longitude (°)")
            plt.ylabel("Latitude (°)")
            plt.title(f"Sample Distribution ({subset}) ({symbol}, Lmax={self.lmax})")
            plt.colorbar(sc, label="mGal" if self.target_type == "acceleration"  else "m²/s²")
            plt.tight_layout()

            out = os.path.join(subset_dir, f"Scatter_Samples_{subset}_{self.target_type}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            df_true = df[true_col].to_numpy()
            df_pred = df[pred_col].to_numpy()

            # Side-by-side maps
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5),
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            )

            vmin = min(df_true.min(), df_pred.min())
            vmax = max(df_true.max(), df_pred.max())

            for ax, data, title in zip(
                    axes,
                    [df_true, df_pred],
                    [f"True {symbol}", f"Predicted {symbol}"]):
                sc = ax.scatter(df["lon"], df["lat"],
                                c=data, s=s, alpha=alpha, cmap=cmap,
                                transform=ccrs.PlateCarree(),
                                vmin=vmin, vmax=vmax)

                ax.set_global()
                gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
                gl.top_labels = False
                gl.right_labels = False

                cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08)
                cbar.set_label(unit)
                ax.set_title(title)

            out = os.path.join(subset_dir, f"Scatter_True_vs_Pred_{subset}_{self.target_type}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

            sc = ax.scatter(df["lon"], df["lat"],
                            c=df_pred, s=s, alpha=alpha, cmap=cmap,
                            transform=ccrs.PlateCarree())

            ax.set_global()
            gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
            gl.top_labels = False
            gl.right_labels = False

            cbar = plt.colorbar(sc, orientation="horizontal", pad=0.08)
            cbar.set_label(unit)

            ax.set_title(f"Predicted {symbol} ({subset})")
            out = os.path.join(subset_dir, f"Scatter_Pred_only_{subset}_{self.target_type}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(8, 5))
            bins = min(50, max(10, len(np.unique(df_true)) // 4))

            plt.hist(df_true, bins=bins, alpha=0.6, label="True", color="orange")
            plt.hist(df_pred, bins=bins, alpha=0.6, label="Predicted", color="steelblue")

            plt.xlabel(f"{symbol} ({unit})")
            plt.ylabel("Frequency")
            plt.title(f"Histogram {subset}: True vs Predicted {symbol}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)

            out = os.path.join(subset_dir, f"Histogram_True_vs_Pred_{subset}_{self.target_type}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            if self.linear_available:

                for lin_label, lin_dict in [
                    ("Model_lmax", self.linear_preds["model"]),
                    ("L_equiv", self.linear_preds["equiv"])
                ]:

                    if subset not in lin_dict:
                        continue

                    lin_vals = lin_dict[subset]

                    plt.figure(figsize=(10, 5))
                    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

                    sc = ax.scatter(df["lon"], df["lat"],
                                    c=lin_vals, s=s, alpha=alpha, cmap=cmap,
                                    transform=ccrs.PlateCarree())

                    ax.set_global()
                    gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
                    gl.top_labels = False
                    gl.right_labels = False

                    cbar = plt.colorbar(sc, orientation="horizontal", pad=0.08)
                    cbar.set_label(unit)

                    ax.set_title(f"Linear Prediction ({lin_label}) — {subset}")
                    out = os.path.join(subset_dir, f"Linear_{lin_label}_{subset}_{self.target_type}.png")
                    plt.savefig(out, dpi=300, bbox_inches="tight")
                    plt.close()

                    fig, axes = plt.subplots(
                        1, 2, figsize=(12, 5),
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
                    )

                    vmin_lin = min(df_true.min(), lin_vals.min())
                    vmax_lin = max(df_true.max(), lin_vals.max())

                    for ax, arr, title in zip(
                            axes,
                            [df_true, lin_vals],
                            ["True", f"Linear ({lin_label})"]):
                        sc = ax.scatter(df["lon"], df["lat"],
                                        c=arr, s=s, alpha=alpha, cmap=cmap,
                                        transform=ccrs.PlateCarree(),
                                        vmin=vmin_lin, vmax=vmax_lin)

                        ax.set_global()
                        gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
                        gl.top_labels = False
                        gl.right_labels = False

                        cbar = plt.colorbar(sc, orientation="horizontal", pad=0.08)
                        cbar.set_label(unit)
                        ax.set_title(title)

                    out = os.path.join(subset_dir, f"True_vs_Linear_{lin_label}_{subset}_{self.target_type}.png")
                    plt.savefig(out, dpi=300, bbox_inches="tight")
                    plt.close()


