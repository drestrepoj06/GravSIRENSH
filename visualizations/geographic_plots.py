"""
Plotting geographically the predictions, true and analytical model data.
A histogram of true vs predictions is also shown.
jhonr
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

class Plotter:
    def __init__(
        self,
        predictions_path,
        output_dir=None,
        target_type="acceleration",
        vec_view="mag",
    ):
        self.predictions_path = predictions_path
        self.output_dir = output_dir or os.path.dirname(predictions_path)
        self.target_type = target_type.lower()
        self.vec_view = vec_view

        os.makedirs(self.output_dir, exist_ok=True)

        data = np.load(self.predictions_path)
        self.lon = data["lon"].astype(np.float32)
        self.lat = data["lat"].astype(np.float32)
        self.y_true = data["y_true"]
        self.y_pred = data["y_pred"]
        self.y_pred_linear = data["y_pred_linear"] if "y_pred_linear" in data.files else None
        self.is_dist = data["is_dist"].astype(bool)
        self.mode = str(data["mode"])

        self.subsets = {
            "all": np.ones_like(self.is_dist, dtype=bool),
            "dist": self.is_dist,
        }

    def _get_plot_values(self, y):
        if self.mode == "u":
            return y.reshape(-1), "ΔU", "m²/s²"

        if self.vec_view == "components":
            return y, "Δa components", "mGal"
        else:
            mag = np.sqrt((y ** 2).sum(axis=1))
            return mag, "‖Δa‖", "mGal"

    def plot_scatter(self, s=0.5, alpha=0.8, cmap="viridis"):

        def _single_map(lon, lat, vals, title, unit, out_path):
            plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

            sc = ax.scatter(
                lon, lat,
                c=vals, s=s, alpha=alpha, cmap=cmap,
                transform=ccrs.PlateCarree()
            )

            ax.set_global()
            gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
            gl.top_labels = False
            gl.right_labels = False

            cbar = plt.colorbar(sc, orientation="horizontal", pad=0.08)
            cbar.set_label(unit)
            ax.set_title(title)

            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

        for subset_name, mask in self.subsets.items():
            subset_dir = os.path.join(self.output_dir, f"Maps_{subset_name}")
            os.makedirs(subset_dir, exist_ok=True)

            lon = self.lon[mask]
            lat = self.lat[mask]
            yt = self.y_true[mask]
            yp = self.y_pred[mask]
            ylin = self.y_pred_linear[mask] if self.y_pred_linear is not None else None

            true_vals, sym, unit = self._get_plot_values(yt)
            pred_vals, _, _ = self._get_plot_values(yp)

            # -------------------------
            # Single maps: True / Model
            # -------------------------
            _single_map(
                lon, lat, true_vals,
                title=f"True {sym} ({subset_name})",
                unit=unit,
                out_path=os.path.join(subset_dir, f"True_only_{subset_name}_{self.mode}_{self.vec_view}.png")
            )

            _single_map(
                lon, lat, pred_vals,
                title=f"Model Pred {sym} ({subset_name})",
                unit=unit,
                out_path=os.path.join(subset_dir, f"Model_only_{subset_name}_{self.mode}_{self.vec_view}.png")
            )

            # -------------------------
            # Side-by-side: True vs Model
            # -------------------------
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5),
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
            )

            vmin = float(min(true_vals.min(), pred_vals.min()))
            vmax = float(max(true_vals.max(), pred_vals.max()))

            for ax, data_arr, title in zip(
                    axes, [true_vals, pred_vals], [f"True {sym}", f"Model Pred {sym}"]
            ):
                sc = ax.scatter(
                    lon, lat, c=data_arr, s=s, alpha=alpha, cmap=cmap,
                    transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax
                )
                ax.set_global()
                gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
                gl.top_labels = False
                gl.right_labels = False
                cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08)
                cbar.set_label(unit)
                ax.set_title(title)

            out = os.path.join(subset_dir, f"True_vs_Model_{subset_name}_{self.mode}_{self.vec_view}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            # ---- Histogram: True vs Model ----
            plt.figure(figsize=(8, 5))
            bins = 50
            plt.hist(true_vals, bins=bins, alpha=0.6, label="True")
            plt.hist(pred_vals, bins=bins, alpha=0.6, label="Model Pred")
            plt.xlabel(f"{sym} ({unit})")
            plt.ylabel("Frequency")
            plt.title(f"Histogram {subset_name}: True vs Model Pred {sym}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            out = os.path.join(subset_dir, f"Hist_True_vs_Model_{subset_name}_{self.mode}_{self.vec_view}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            # -------------------------
            # Analytical: single + compare
            # -------------------------
            if ylin is not None:
                lin_vals, _, _ = self._get_plot_values(ylin)

                _single_map(
                    lon, lat, lin_vals,
                    title=f"Analytical Pred {sym} ({subset_name})",
                    unit=unit,
                    out_path=os.path.join(subset_dir, f"Analytical_only_{subset_name}_{self.mode}_{self.vec_view}.png")
                )

                fig, axes = plt.subplots(
                    1, 2, figsize=(12, 5),
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
                )

                vmin = float(min(true_vals.min(), lin_vals.min()))
                vmax = float(max(true_vals.max(), lin_vals.max()))

                for ax, data_arr, title in zip(
                        axes, [true_vals, lin_vals], [f"True {sym}", f"Analytical Pred {sym}"]
                ):
                    sc = ax.scatter(
                        lon, lat, c=data_arr, s=s, alpha=alpha, cmap=cmap,
                        transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax
                    )
                    ax.set_global()
                    gl = ax.gridlines(draw_labels=True, linewidth=0, color="none")
                    gl.top_labels = False
                    gl.right_labels = False
                    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08)
                    cbar.set_label(unit)
                    ax.set_title(title)

                out = os.path.join(subset_dir, f"True_vs_Analytical_{subset_name}_{self.mode}_{self.vec_view}.png")
                plt.savefig(out, dpi=300, bbox_inches="tight")
                plt.close()


