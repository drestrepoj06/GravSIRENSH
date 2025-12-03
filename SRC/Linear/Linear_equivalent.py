"""Equivalent pyshtools linear model to the nparams of the trained neural network
jhonr"""

import json
import numpy as np
import pyshtools as pysh
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator

class LinearEquivalentGenerator:
    """
      • Computes L_equiv from NN parameter count and builds linear residual SH model (full − l=2)
      • Computes Lmax from the same lmax used in the embedding
      • Saves g_r, g_theta, g_phi, g_mag as .npy
      • Stores all results in the object
    """

    def __init__(self, run_dir, data_path, altitude=0.0):
        self.run_dir = run_dir
        self.data_path = data_path
        self.altitude = altitude

        self.config_path = os.path.join(run_dir, "config.json")
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        self.lmax = self.config["lmax"]

        params = self.compute_siren_params(self.config)
        self.L_equiv = self.params_to_lmax(params)

        # === Generate linear baseline for model-lmax ===
        (
            self.df_grid_model,
            self.model_dU_grid,
            self.model_lats,
            self.model_lons,
            self.model_clm_full_g,
            self.model_clm_low_g,
            self.model_r0
        ) = self.generate_linear_equiv(self.lmax)

        # === Generate linear baseline for L_equiv ===
        (
            self.df_grid_equiv,
            self.equiv_dU_grid,
            self.equiv_lats,
            self.equiv_lons,
            self.equiv_clm_full_g,
            self.equiv_clm_low_g,
            self.equiv_r0
        ) = self.generate_linear_equiv(self.L_equiv)

        # === Evaluate linear predictions at test coords: model-lmax ===
        (
            self.g_r_lin_model,
            self.g_theta_lin_model,
            self.g_phi_lin_model,
            self.g_mag_lin_model,
            self.mask_model
        ) = self.evaluate_on_test(
            df_grid=self.df_grid_model,
            dU_grid=self.model_dU_grid,
            lats_grid=self.model_lats,
            lons_grid=self.model_lons,
            clm_full_g=self.model_clm_full_g,
            clm_low_g=self.model_clm_low_g,
            r0=self.model_r0,
            L=self.lmax,
            save=True,
            label="model"
        )

        # === Evaluate linear predictions at test coords: L_equiv ===
        (
            self.g_r_lin_equiv,
            self.g_theta_lin_equiv,
            self.g_phi_lin_equiv,
            self.g_mag_lin_equiv,
            self.mask_equiv
        ) = self.evaluate_on_test(
            df_grid=self.df_grid_equiv,
            dU_grid=self.equiv_dU_grid,
            lats_grid=self.equiv_lats,
            lons_grid=self.equiv_lons,
            clm_full_g=self.equiv_clm_full_g,
            clm_low_g=self.equiv_clm_low_g,
            r0=self.equiv_r0,
            L=self.L_equiv,
            save=True,
            label="equiv"
        )

    @staticmethod
    def compute_siren_params(config):
        lmax = config["lmax"]  # local variable, not self.lmax
        hidden = config["hidden_features"]
        layers = config["hidden_layers"]

        input_dim = (lmax + 1) ** 2

        mode = config.get("mode", "U")
        if mode in ["U", "g_indirect"]:
            output_dim = 1
        elif mode == "g_direct":
            output_dim = 2
        elif mode == "g_hybrid":  # ["U_g_direct", "U_g_indirect", "U_g_hybrid"]
            output_dim = 3
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        params = input_dim * hidden + hidden
        for _ in range(layers - 1):
            params += hidden * hidden + hidden
        params += hidden * output_dim + output_dim

        return params

    @staticmethod
    def params_to_lmax(params):
        return max(0, int(np.round(np.sqrt(params) - 1)))

    def _scale_clm(self, clm, scale):
        scaled = clm.copy()
        scaled.coeffs *= scale[np.newaxis, :, np.newaxis]
        return scaled

    def generate_linear_equiv(self, L):

        linear_output_path = os.path.join(
            self.run_dir,
            f"linear_{L}-2.parquet"
        )

        clm_full = pysh.datasets.Earth.EGM2008(lmax=L)
        clm_low = pysh.datasets.Earth.EGM2008(lmax=2)

        r0 = clm_full.r0
        r1 = r0 + self.altitude

        deg_full = np.arange(L + 1)
        deg_low = np.arange(3)

        scale_U_full = (r0 / r1) ** deg_full
        scale_U_low = (r0 / r1) ** deg_low
        scale_g_full = (r0 / r1) ** (deg_full + 2)
        scale_g_low = (r0 / r1) ** (deg_low + 2)

        clm_full_U = self._scale_clm(clm_full, scale_U_full)
        clm_low_U = self._scale_clm(clm_low, scale_U_low)

        clm_full_g = self._scale_clm(clm_full, scale_g_full)
        clm_low_g = self._scale_clm(clm_low, scale_g_low)

        res_full = clm_full_g.expand(lmax=L)
        res_low = clm_low_g.expand(lmax=L)
        res_full_U = clm_full_U.expand(lmax=L)
        res_low_U = clm_low_U.expand(lmax=L)

        pot_full = res_full_U.pot.data
        pot_low = res_low_U.pot.data
        dU = pot_full - pot_low

        gr_full = res_full.rad.data * 1e5
        gr_low = res_low.rad.data * 1e5
        gtheta_full = res_full.theta.data * 1e5
        gtheta_low = res_low.theta.data * 1e5
        gphi_full = res_full.phi.data * 1e5
        gphi_low = res_low.phi.data * 1e5

        dg_r = gr_full - gr_low
        dg_theta = gtheta_full - gtheta_low
        dg_phi = gphi_full - gphi_low
        dg_total = np.sqrt(dg_theta ** 2 + dg_phi ** 2)

        # Build dataframe
        lats = res_full.pot.lats()
        lons = res_full.pot.lons()
        lat_grid = np.repeat(lats, len(lons))
        lon_grid = np.tile(lons, len(lats))

        df = pd.DataFrame({
            "lat": lat_grid.astype("float32"),
            "lon": lon_grid.astype("float32"),
            "dU_m2_s2": dU.ravel().astype("float32"),
            "dg_r_mGal": dg_r.ravel().astype("float32"),
            "dg_theta_mGal": dg_theta.ravel().astype("float32"),
            "dg_phi_mGal": dg_phi.ravel().astype("float32"),
            "dg_total_mGal": dg_total.ravel().astype("float32"),
            "radius_m": np.full(dU.size, r0, dtype="float32")
        })

        df.to_parquet(linear_output_path, index=False)

        return df, dU, lats, lons, clm_full_g, clm_low_g, r0

    def evaluate_on_test(
            self,
            df_grid,
            dU_grid,
            lats_grid,
            lons_grid,
            clm_full_g,
            clm_low_g,
            r0,
            L,
            save=True,
            label=""
            ):

        df_test = pd.read_parquet(self.data_path)
        lat = df_test["lat"].values
        lon = df_test["lon"].values

        mask = np.abs(lat) < 89.9999
        lat_f = lat[mask]
        lon_f = lon[mask]
        r_f = np.full_like(lat_f, r0)

        # Interpolate potential grid to test points
        dU_interp_func = RegularGridInterpolator(
            (lats_grid, lons_grid),
            dU_grid,
            bounds_error=False,
            fill_value=None
        )

        points = np.column_stack((lat_f, lon_f))
        dU = dU_interp_func(points).astype("float32")

        # Gravity at test points
        g_full = clm_full_g.expand(
            lat=lat_f.reshape(-1, 1),
            lon=lon_f.reshape(-1, 1),
            r=r_f.reshape(-1, 1),
            lmax=L,
            degrees=True
        )

        g_low = clm_low_g.expand(
            lat=lat_f.reshape(-1, 1),
            lon=lon_f.reshape(-1, 1),
            r=r_f.reshape(-1, 1),
            lmax=2,
            degrees=True
        )

        g = g_full - g_low

        g_r = (g[:, 0] * 1e5).astype("float32")  # mGal
        g_theta = (g[:, 1] * 1e5).astype("float32")
        g_phi = (g[:, 2] * 1e5).astype("float32")
        g_mag = np.sqrt(g_theta ** 2 + g_phi ** 2).astype("float32")

        if save:
            np.save(f"{self.run_dir}/linear_U_{label}.npy", dU)
            np.save(f"{self.run_dir}/linear_g_r_{label}.npy", g_r)
            np.save(f"{self.run_dir}/linear_g_theta_{label}.npy", g_theta)
            np.save(f"{self.run_dir}/linear_g_phi_{label}.npy", g_phi)
            np.save(f"{self.run_dir}/linear_g_mag_{label}.npy", g_mag)

        return g_r, g_theta, g_phi, g_mag, mask
