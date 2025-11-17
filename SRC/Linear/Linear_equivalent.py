"""Equivalent pyshtools linear model to the nparams of the trained neural network
jhonr"""

import json
import numpy as np
import pyshtools as pysh
import pandas as pd
import os

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
        self.linear_output_path = os.path.join(run_dir, "linear_results.parquet")

        # Load config
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        # Initialize placeholders
        self.clm_U = None
        self.clm_g = None
        self.L_equiv = None

        # Automatically run entire pipeline
        self.df_grid = self.generate_linear_equiv()
        (
            self.g_r_lin,
            self.g_theta_lin,
            self.g_phi_lin,
            self.g_mag_lin,
            self.mask
        ) = self.evaluate_on_test(save=True)

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
        elif mode in ["U_g_direct", "U_g_indirect"]:
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

    def generate_linear_equiv(self):
        params = self.compute_siren_params(self.config)
        L_equiv = self.params_to_lmax(params)
        self.L_equiv = L_equiv

        print(f"🔢 NN parameters: {params:,}")
        print(f"🎯 Equivalent SH degree: {L_equiv}")

        clm_full = pysh.datasets.Earth.EGM2008(lmax=L_equiv)

        # Copy and zero degrees 0, 1, 2
        clm_res = clm_full.copy()
        clm_res.coeffs[:, :3, :] = 0.0  # l = 0, 1, 2  → exactly zero

        r0 = clm_res.r0
        r1 = r0 + self.altitude
        degrees = np.arange(L_equiv + 1)

        scale_U = (r0 / r1) ** degrees
        scale_g = (r0 / r1) ** (degrees + 2)

        clm_U = clm_res.copy()
        clm_g = clm_res.copy()

        clm_U.coeffs *= scale_U[np.newaxis, :, np.newaxis]
        clm_g.coeffs *= scale_g[np.newaxis, :, np.newaxis]

        self.clm_U = clm_U
        self.clm_g = clm_g

        # Make grid for inspection
        grid_U = clm_U.expand(lmax=L_equiv)
        grid_g = clm_g.expand(lmax=L_equiv)

        lats = grid_U.pot.lats()
        lons = grid_U.pot.lons()
        lat_grid = np.repeat(lats, len(lons))
        lon_grid = np.tile(lons, len(lats))

        dU = grid_U.pot.data
        dg_r = grid_g.rad.data * 1e5
        dg_theta = grid_g.theta.data * 1e5
        dg_phi = grid_g.phi.data * 1e5
        dg_total = np.sqrt(dg_theta**2 + dg_phi**2)

        df = pd.DataFrame({
            "lat": lat_grid,
            "lon": lon_grid,
            "dU_m2_s2": dU.ravel(),
            "dg_r_mGal": dg_r.ravel(),
            "dg_theta_mGal": dg_theta.ravel(),
            "dg_phi_mGal": dg_phi.ravel(),
            "dg_total_mGal": dg_total.ravel(),
            "radius_m": np.full(dU.size, r0),
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
        r_f = np.full_like(lat_f, self.clm_g.r0, dtype=float)

        g = self.clm_g.expand(
            lat=lat_f.reshape(-1, 1),
            lon=lon_f.reshape(-1, 1),
            r=r_f.reshape(-1, 1),
            lmax=self.L_equiv,
            degrees=True,
        )

        g_r = g[:, 0]*1e5
        g_theta = g[:, 1]*1e5
        g_phi = g[:, 2]*1e5
        g_mag = np.sqrt(g_theta**2 + g_phi**2)
        if save:
            np.save(os.path.join(self.run_dir, "linear_g_r.npy"), g_r)
            np.save(os.path.join(self.run_dir, "linear_g_theta.npy"), g_theta)
            np.save(os.path.join(self.run_dir, "linear_g_phi.npy"), g_phi)
            np.save(os.path.join(self.run_dir, "linear_g_mag.npy"), g_mag)
            print(f"📁 Saved linear gravity components in {self.run_dir}")

        return g_r, g_theta, g_phi, g_mag, mask



