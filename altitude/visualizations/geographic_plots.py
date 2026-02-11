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
                os.path.join(os.path.dirname(data_path), "..", "outputs", "figures")
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

    def _make_grid(self, value_col, alt_center_m=None, alt_half_width_m=10_000):
        lon_grid = np.linspace(0, 360, 720)
        lat_grid = np.linspace(-90, 90, 361)
        Lon, Lat = np.meshgrid(lon_grid, lat_grid)

        df = self.sample_df
        if alt_center_m is not None:
            if "altitude_m" not in df.columns:
                raise ValueError("sample_df must contain 'altitude_m' to grid by altitude.")
            df = df[np.abs(df["altitude_m"].values - alt_center_m) <= alt_half_width_m].copy()

        if len(df) < 5000:
            raise ValueError(f"Not enough points in altitude slice to grid: n={len(df)}")

        points = np.vstack((df["lon"].values, df["lat"].values)).T
        values = df[value_col].values

        grid = griddata(points, values, (Lon, Lat), method="linear")

        mask = np.isnan(grid)
        if np.any(mask):
            grid[mask] = griddata(points, values, (Lon[mask], Lat[mask]), method="nearest")

        return Lon, Lat, grid

    def plot_map(self, cmap="viridis", alt_center_m=None, alt_half_width_m=10_000):
        """Interpolated global map of the selected field (optionally at an altitude slice)."""
        if self.target_type == "acceleration":
            value_col = "dg_total_mGal"
            unit = "mGal"
        else:
            value_col = "dU_m2_s2"
            unit = "m²/s²"

        Lon, Lat, grid_total = self._make_grid(
            value_col,
            alt_center_m=alt_center_m,
            alt_half_width_m=alt_half_width_m,
        )

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

        title = self._generate_title()
        if alt_center_m is not None:
            title += f" — h≈{alt_center_m / 1000:.0f} km (±{alt_half_width_m / 1000:.0f} km)"
        ax.set_title(title, fontsize=13, pad=12)

        suffix = f"{self._fname_suffix()}_{self.mode}_{self.target_type[:3]}"
        if alt_center_m is not None:
            suffix += f"_h{int(round(alt_center_m))}_dh{int(round(alt_half_width_m))}"

        output_filename = f"Map_{suffix}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_scatter_maps(
            self,
            color_by=None,
            s=0.5,
            alpha=0.8,
            cmap="viridis",
            alt_center_m=None,
            alt_half_width_m=10_000,
            alt_range_m=None,
    ):
        """
        Scatter maps for A/F/C, optionally restricted to an altitude slice or range.

        Use ONE of:
          - alt_center_m + alt_half_width_m  (slice)
          - alt_range_m=(min_m, max_m)       (range)
        """

        if ("altitude_m" not in self.sample_df.columns) and (alt_center_m is not None or alt_range_m is not None):
            raise ValueError("sample_df must include 'altitude_m' to filter by altitude.")

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

        # Build a suffix for filenames/titles
        alt_tag = ""
        if alt_center_m is not None:
            alt_tag = f"_h{int(round(alt_center_m / 1000))}km_dh{int(round(alt_half_width_m / 1000))}km"
        elif alt_range_m is not None:
            alt_tag = f"_h{int(round(alt_range_m[0] / 1000))}-{int(round(alt_range_m[1] / 1000))}km"

        for subset in ["A", "F", "C"]:
            df = self.sample_df_subset[subset]

            # ---- NEW: altitude filter ----
            if alt_center_m is not None:
                df = df[np.abs(df["altitude_m"].values - alt_center_m) <= alt_half_width_m].copy()
            elif alt_range_m is not None:
                lo, hi = alt_range_m
                df = df[(df["altitude_m"].values >= lo) & (df["altitude_m"].values <= hi)].copy()

            if df.empty:
                print(f"⚠ subset {subset}: no points after altitude filter {alt_tag} — skipping.")
                continue

            subset_dir = os.path.join(self.output_dir, f"Maps_{subset}")
            os.makedirs(subset_dir, exist_ok=True)

            # optional: keep color scale comparable within this slice
            df_true = df[true_col].to_numpy()
            df_pred = df[pred_col].to_numpy()

            # Sample distribution
            plt.figure(figsize=(10, 5))
            color_data = df[color_by] if (color_by in df.columns) else df[true_col]
            sc = plt.scatter(df["lon"], df["lat"], c=color_data, s=s, alpha=alpha, cmap=cmap)
            plt.xlabel("Longitude (°)")
            plt.ylabel("Latitude (°)")
            title = f"Sample Distribution ({subset}) ({symbol}, Lmax={self.lmax})"
            if alt_tag:
                title += f" {alt_tag.replace('_', ' ')}"
            plt.title(title)
            plt.colorbar(sc, label=color_by or true_col)
            plt.tight_layout()
            plt.savefig(os.path.join(subset_dir, f"Scatter_Samples_{subset}_{self.target_type}{alt_tag}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            # True vs Pred side-by-side maps
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5),
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            )

            vmin = min(df_true.min(), df_pred.min())
            vmax = max(df_true.max(), df_pred.max())

            for ax, data, title0 in zip(axes, [df_true, df_pred], [f"True {symbol}", f"Predicted {symbol}"]):
                sc = ax.scatter(df["lon"], df["lat"], c=data, s=s, alpha=alpha, cmap=cmap,
                                transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
                ax.set_global()
                gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
                gl.top_labels = False
                gl.right_labels = False
                cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08)
                cbar.set_label(unit)

                ttl = title0
                if alt_tag:
                    ttl += f"\n{alt_tag.replace('_', ' ')}"
                ax.set_title(ttl)

            plt.savefig(os.path.join(subset_dir, f"Scatter_True_vs_Pred_{subset}_{self.target_type}{alt_tag}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            # Pred only
            plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
            sc = ax.scatter(df["lon"], df["lat"], c=df_pred, s=s, alpha=alpha, cmap=cmap,
                            transform=ccrs.PlateCarree())
            ax.set_global()
            gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
            gl.top_labels = False
            gl.right_labels = False
            cbar = plt.colorbar(sc, orientation="horizontal", pad=0.08)
            cbar.set_label(unit)
            ttl = f"Predicted {symbol} ({subset})"
            if alt_tag:
                ttl += f" {alt_tag.replace('_', ' ')}"
            ax.set_title(ttl)
            plt.savefig(os.path.join(subset_dir, f"Scatter_Pred_only_{subset}_{self.target_type}{alt_tag}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            # Linear maps (if available)
            if self.linear_available:
                for lin_label, lin_dict in [("Model_lmax", self.linear_preds["model"]),
                                            ("L_equiv", self.linear_preds["equiv"])]:
                    if subset not in lin_dict:
                        continue

                    lin_vals = lin_dict[subset]

                    # IMPORTANT: lin_vals must align to df AFTER slicing.
                    # This will be true only if you slice df by ROWS that correspond to the same ordering as the saved npy.
                    # If not, you must slice lin_vals with the same mask indices before plotting.

                    plt.figure(figsize=(10, 5))
                    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
                    sc = ax.scatter(df["lon"], df["lat"], c=lin_vals[:len(df)], s=s, alpha=alpha, cmap=cmap,
                                    transform=ccrs.PlateCarree())
                    ax.set_global()
                    gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
                    gl.top_labels = False
                    gl.right_labels = False
                    cbar = plt.colorbar(sc, orientation="horizontal", pad=0.08)
                    cbar.set_label(unit)
                    ttl = f"Linear Prediction ({lin_label}) — {subset}"
                    if alt_tag:
                        ttl += f"\n{alt_tag.replace('_', ' ')}"
                    ax.set_title(ttl)
                    plt.savefig(
                        os.path.join(subset_dir, f"Linear_{lin_label}_{subset}_{self.target_type}{alt_tag}.png"),
                        dpi=300, bbox_inches="tight")
                    plt.close()

    def plot_rmse_by_altitude(self, bins_km=(0, 50, 100, 150, 200, 250, 300, 350,400), subset="A"):
        """
        RMSE vs altitude bins for NN and (optionally) linear baselines.
        Produces a bar chart (one group per bin).
        """

        if "altitude_m" not in self.sample_df.columns:
            raise ValueError("sample_df must include 'altitude_m' for RMSE-by-altitude plots.")

        # columns by target type
        if self.target_type == "acceleration":
            true_col = "dg_total_mGal"
            pred_col = "predicted_dg_total_mGal"
            unit = "mGal"
            symbol = "Δg"
        else:
            true_col = "dU_m2_s2"
            pred_col = "predicted_dU_m2_s2"
            unit = "m²/s²"
            symbol = "ΔU"

        df = self.sample_df_subset[subset].copy()

        # altitude bins
        bins_m = np.array(bins_km, dtype=float) * 1000.0
        bin_ids = np.digitize(df["altitude_m"].to_numpy(), bins_m, right=False) - 1
        n_bins = len(bins_m) - 1

        # helpers
        def rmse(a, b):
            a = np.asarray(a);
            b = np.asarray(b)
            return float(np.sqrt(np.mean((a - b) ** 2)))

        # prepare series to compare
        series = {
            "NN": df[pred_col].to_numpy(),
        }

        if self.linear_available:
            # your loader currently stores magnitude arrays only for linear
            if subset in self.linear_preds["equiv"]:
                series["Linear equiv"] = self.linear_preds["equiv"][subset]

        y_true = df[true_col].to_numpy()

        # compute rmse per bin
        rmse_table = {name: [] for name in series.keys()}
        counts = []

        for b in range(n_bins):
            idx = np.where(bin_ids == b)[0]
            counts.append(len(idx))
            if len(idx) < 50:  # avoid noisy bins
                for name in rmse_table:
                    rmse_table[name].append(np.nan)
                continue

            for name, y_pred in series.items():
                rmse_table[name].append(rmse(y_pred[idx], y_true[idx]))

        # plot
        out_dir = os.path.join(self.output_dir, f"RMSE_by_altitude_{subset}")
        os.makedirs(out_dir, exist_ok=True)

        x = np.arange(n_bins)
        width = 0.8 / max(1, len(rmse_table))

        plt.figure(figsize=(12, 5))
        for i, (name, vals) in enumerate(rmse_table.items()):
            plt.bar(x + i * width, vals, width=width, label=name)

        labels = [f"{bins_km[i]}–{bins_km[i + 1]} km\n(n={counts[i]})" for i in range(n_bins)]
        plt.xticks(x + width * (len(rmse_table) - 1) / 2, labels)
        plt.ylabel(f"RMSE ({unit})")
        plt.title(f"RMSE vs Altitude bins — {symbol} — subset {subset}")
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.legend()

        out = os.path.join(out_dir, f"RMSE_by_altitude_{self.target_type}_{subset}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

