"""Equivalent pyshtools linear model to the nparams of the trained neural network
jhonr"""

import json
import numpy as np
import pyshtools as pysh
import pandas as pd
import os
import sys
from SRC.Visualizations.Geographic_plots import GravityDataPlotter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class LinearEquivalentGenerator:
    """
    Automatically:
      • Loads config.json from run_dir
      • Computes L_equiv from NN parameter count
      • Builds linear residual SH model (full − l=2)
      • Saves grid as linear_results.parquet
      • Evaluates gravity at test points (data_path)
      • Saves g_r, g_theta, g_phi, g_mag as .npy
      • Stores all results in the object

    Usage:
        gen = LinearEquivalentGenerator(run_dir, data_path)
    """

    def __init__(self, run_dir, data_path, altitude=0.0):
        self.run_dir = run_dir
        self.data_path = data_path
        self.altitude = altitude

        # Internal paths
        self.config_path = os.path.join(run_dir, "config.json")
        self.linear_output_path = None

        # Load config
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        # Initialize placeholders
        self.clm_U = None
        self.clm_g = None
        self.L_equiv = None

        # Step 1 — run the original linear grid generator (UNCHANGED)
        self.df_grid = self.generate_linear_equiv()

        # Step 3 — now evaluate on test points
        self.g_r_lin, self.g_theta_lin, self.g_phi_lin, \
            self.g_mag_lin, self.mask = self.evaluate_on_test(save=True)

    @staticmethod
    def compute_siren_params(config):
        lmax = config["lmax"]
        hidden = config["hidden_features"]
        layers = config["hidden_layers"]

        input_dim = (lmax + 1)**2

        mode = config.get("mode", "U")
        if mode in ["U", "g_indirect"]:
            output_dim = 1
        elif mode == "g_direct":
            output_dim = 2
        elif mode in ["U_g_direct", "U_g_indirect", "g_hybrid", "U_g_hybrid"]:
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

    def generate_linear_equiv(self):
        params = self.compute_siren_params(self.config)
        L_equiv = self.params_to_lmax(params)
        self.L_equiv = L_equiv
        self.linear_output_path = os.path.join(
            self.run_dir,
            f"linear_{self.L_equiv}-2.parquet"
        )
        self.clm_full_g = None
        self.clm_low_g = None

        print(f"🔢 NN parameters: {params:,}")
        print(f"🎯 Equivalent SH degree: {L_equiv}")

        clm_full = pysh.datasets.Earth.EGM2008(lmax=L_equiv)
        clm_low = pysh.datasets.Earth.EGM2008(lmax=2) # Removed manually degree 2 since the data doesn't have it

        self.r0 = clm_full.r0
        r1 = self.r0 + self.altitude

        deg_full = np.arange(self.L_equiv + 1)
        deg_low = np.arange(2 + 1)

        scale_U_full = (self.r0 / r1) ** deg_full
        scale_U_low = (self.r0 / r1) ** deg_low
        scale_g_full = (self.r0 / r1) ** (deg_full + 2)
        scale_g_low = (self.r0 / r1) ** (deg_low + 2)

        clm_full_U = self._scale_clm(clm_full, scale_U_full)
        clm_low_U = self._scale_clm(clm_low, scale_U_low)
        self.clm_full_g = self._scale_clm(clm_full, scale_g_full)
        self.clm_low_g = self._scale_clm(clm_low, scale_g_low)

        res_full = self.clm_full_g.expand(lmax=self.L_equiv)
        res_low = self.clm_low_g.expand(lmax=self.L_equiv)

        res_full_U = clm_full_U.expand(lmax=self.L_equiv)
        res_low_U = clm_low_U.expand(lmax=self.L_equiv)

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
        dg_total = np.sqrt(
            dg_theta ** 2 + dg_phi ** 2)  # +dg_r**2) the radial component will be added when changing altitude

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
            "radius_m": np.full(dU.size, self.r0, dtype="float32")
        })

        df.to_parquet(self.linear_output_path, index=False)
        print(f"📁 Saved linear SH grid → {self.linear_output_path}")

        return df

    def evaluate_on_test(self, save=True):
        df_test = pd.read_parquet(self.data_path)
        lat = df_test["lat"].values
        lon = df_test["lon"].values

        mask = np.abs(lat) < 89.9999
        lat_f = lat[mask]
        lon_f = lon[mask]

        r_f = np.full_like(lat_f, self.r0)

        # FULL model
        g_full = self.clm_full_g.expand(
            lat=lat_f.reshape(-1, 1),
            lon=lon_f.reshape(-1, 1),
            r=r_f.reshape(-1, 1),
            lmax=self.L_equiv,
            degrees=True,
        )

        # LOW-degree model (degree 0–2)
        g_low = self.clm_low_g.expand(
            lat=lat_f.reshape(-1, 1),
            lon=lon_f.reshape(-1, 1),
            r=r_f.reshape(-1, 1),
            lmax=2,
            degrees=True,
        )

        # RESIDUAL field
        g = g_full - g_low

        g_r = g[:, 0] * 1e5
        g_theta = g[:, 1] * 1e5
        g_phi = g[:, 2] * 1e5
        g_mag = np.sqrt(g_theta ** 2 + g_phi ** 2)

        if save:
            np.save(f"{self.run_dir}/linear_g_r.npy", g_r)
            np.save(f"{self.run_dir}/linear_g_theta.npy", g_theta)
            np.save(f"{self.run_dir}/linear_g_phi.npy", g_phi)
            np.save(f"{self.run_dir}/linear_g_mag.npy", g_mag)

        return g_r, g_theta, g_phi, g_mag, mask

