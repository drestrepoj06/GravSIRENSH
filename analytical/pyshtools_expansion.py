"""Equivalent pyshtools linear model to the nparams of the trained hybrid and numerical models
jhonr"""

import re
import numpy as np
import pyshtools as pysh
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import os
import time

# Based on the tutorials of https://shtools.github.io/SHTOOLS/python-examples.html
class Analytical:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.run_name = os.path.basename(os.path.normpath(run_dir))

        self.config = self.parse_config_from_run_dir(self.run_name)

        self.lmax = self.config.get("lmax", None)

        params = self.compute_model_params(self.config)
        self.L_equiv = self.params_to_lmax(params)

        (
            df_grid_equiv, equiv_du_grid, equiv_lats,
            equiv_lons, equiv_clm_full_g, equiv_clm_low_g, equiv_r0
        ) = self.generate_linear_equiv(self.L_equiv)

        self.equiv = {
            "df_grid": df_grid_equiv,
            "du_grid": equiv_du_grid,
            "lats": equiv_lats,
            "lons": equiv_lons,
            "clm_full": equiv_clm_full_g,
            "clm_low": equiv_clm_low_g,
            "r0": equiv_r0,
            "l": self.L_equiv
        }

    @staticmethod
    def parse_config_from_run_dir(run_name):
        config = {}

        patterns = {
            "lmax": r"lmax=([0-9]+)",
            "hidden_layers": r"layers=([0-9]+)",
            "hidden_features": r"neurons=([0-9]+)",
            "mode": r"mode=([^_]+(?:_[^_]+)?)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, run_name)
            if match:
                value = match.group(1)

                if key in {"lmax", "hidden_layers", "hidden_features"}:
                    value = int(value)

                config[key] = value

        if "mode" not in config:
            config["mode"] = "u"

        return config

    @staticmethod
    def compute_model_params(config):
        hidden = config["hidden_features"]
        layers = config["hidden_layers"]

        mode = config.get("mode", "u")
        if mode == "u":
            output_dim = 1
        else:
            output_dim = 3

        if "lmax" in config and config["lmax"] is not None:
            lmax = config["lmax"]
            input_dim = (lmax + 1) ** 2
        else:
            input_dim = 2

        params = input_dim * hidden + hidden
        for _ in range(layers - 1):
            params += hidden * hidden + hidden
        params += hidden * output_dim + output_dim

        return params

    @staticmethod
    def params_to_lmax(params):
        return max(0, int(np.round(np.sqrt(params) - 1)))

    def generate_linear_equiv(self, l):
        clm_full = pysh.datasets.Earth.EGM2008(lmax=l)
        clm_low = pysh.datasets.Earth.EGM2008(lmax=2)

        r0 = clm_full.r0

        res_full = clm_full.expand(lmax=l)
        res_low = clm_low.expand(lmax=l)

        pot_full = res_full.pot.data
        pot_low = res_low.pot.data
        du = pot_full - pot_low

        gr_full = res_full.rad.data * 1e5
        gr_low = res_low.rad.data * 1e5
        gtheta_full = res_full.theta.data * 1e5
        gtheta_low = res_low.theta.data * 1e5
        gphi_full = res_full.phi.data * 1e5
        gphi_low = res_low.phi.data * 1e5

        da_r = gr_full - gr_low
        da_theta = gtheta_full - gtheta_low
        da_phi = gphi_full - gphi_low
        da_total = np.sqrt(da_theta ** 2 + da_phi ** 2 + da_r**2)

        lats = res_full.pot.lats()
        lons = res_full.pot.lons()
        lat_grid = np.repeat(lats, len(lons))
        lon_grid = np.tile(lons, len(lats))

        df = pd.DataFrame({
            "lat": lat_grid.astype("float32"),
            "lon": lon_grid.astype("float32"),
            "dU_m2_s2": du.ravel().astype("float32"),
            "da_r_mGal": da_r.ravel().astype("float32"),
            "da_theta_mGal": da_theta.ravel().astype("float32"),
            "da_phi_mGal": da_phi.ravel().astype("float32"),
            "da_total_mGal": da_total.ravel().astype("float32"),
            "radius_m": np.full(du.size, r0, dtype="float32")
        })

        return df, du, lats, lons, clm_full, clm_low, r0

    def evaluate_on_coords(
            self,
            *,
            lat: np.ndarray,
            lon: np.ndarray,
            du_grid: np.ndarray,
            lats_grid: np.ndarray,
            lons_grid: np.ndarray,
            clm_full,
            clm_low,
            r0: float,
            l: int,
            return_timing: bool = False
    ):
        """
        Returns linear (analytical) predictions at the given coordinates:
        dU [m^2/s^2], and g components [mGal].
        Shapes: dU (N,), a_theta/a_phi/a_r (N,)
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        r_f = np.full_like(lat, r0, dtype=np.float64)

        interp = RegularGridInterpolator(
            (lats_grid, lons_grid), du_grid, bounds_error=False, fill_value=None
        )
        du = interp(np.column_stack((lat, lon))).astype("float32")
        t0 = time.perf_counter() ## Time only reported for acceleration, as pyshtools does not perform expansions for potential values
        a_full = clm_full.expand(lat=lat, lon=lon, r=r_f, lmax=l, degrees=True)
        t_full = time.perf_counter() - t0
        a_low = clm_low.expand(lat=lat, lon=lon, r=r_f, lmax=2, degrees=True)
        a = a_full - a_low

        timing = {
            "n_points": int(len(lat)),
            "t_expand_full_s": float(t_full)
        }

        # convert to mGal
        a_r = (a[:, 0] * 1e5).astype("float32") # in mGal
        a_theta = (a[:, 1] * 1e5).astype("float32")
        a_phi = (a[:, 2] * 1e5).astype("float32")

        if return_timing:
            timing = {
                "n_points": int(len(lat)),
                "t_total_s": float(t_full),
                "seconds_per_sample": float(t_full / len(lat)),
                "samples_per_second": float(len(lat) / t_full),
            }
            return du, a_theta, a_phi, a_r, timing

        return du, a_theta, a_phi, a_r
