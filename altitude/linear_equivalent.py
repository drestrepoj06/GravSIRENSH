"""Equivalent pyshtools linear model to the nparams of the trained neural network and with the same
lmax of the embedding
jhonr"""

import json
import numpy as np
import pyshtools as pysh
import pandas as pd
import os
import time
from scipy.interpolate import RegularGridInterpolator

class LinearEquivalentGenerator:

    def __init__(self, run_dir, data_path):
        self.run_dir = run_dir
        self.data_path = data_path

        with open(os.path.join(run_dir, "config.json")) as f:
            self.config = json.load(f)

        self.is_pinn = ("lmax" not in self.config) or (self.config.get("model_type", "").lower() == "pinn")
        self.lmax = self.config.get("lmax", None)

        params = self.compute_model_params(self.config)
        self.L_equiv = self.params_to_lmax(params)

        if self.lmax is not None:
            clm_full, clm_low, r0 = self.load_clm_pair(self.lmax)
            self.model = {"clm_full": clm_full, "clm_low": clm_low, "r0": r0, "L": self.lmax}
        else:
            self.model = None

        clm_full, clm_low, r0 = self.load_clm_pair(self.L_equiv)
        self.equiv = {"clm_full": clm_full, "clm_low": clm_low, "r0": r0, "L": self.L_equiv}

    @staticmethod
    def compute_model_params(config):
        hidden = config["hidden_features"]
        layers = config["hidden_layers"]

        mode = config.get("mode", "U")
        if mode in ["U", "g_indirect"]:
            output_dim = 1
        elif mode == "g_direct":
            output_dim = 3
        else:
            raise ValueError(f"Unknown mode '{mode}'")

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

    def load_clm_pair(self, L: int):
        clm_full = pysh.datasets.Earth.EGM2008(lmax=L)
        clm_low = pysh.datasets.Earth.EGM2008(lmax=2)
        r0 = clm_full.r0
        return clm_full, clm_low, r0

    @staticmethod
    def params_to_lmax(params):
        return max(0, int(np.round(np.sqrt(params) - 1)))

    def _scale_clm(self, clm, scale):
        scaled = clm.copy()
        scaled.coeffs *= scale[np.newaxis, :, np.newaxis]
        return scaled

    def _dh_axes_from_lmax(self, lmax: int, sampling: int = 2, extend: bool = False):
        """
        Returns lat_axis (descending) and lon_axis (0..360-step)
        matching MakeGravGridDH output shape when extend=False.
        Supports lmax=0.
        """
        lmax = int(lmax)
        if lmax < 0:
            raise ValueError(f"lmax must be >= 0, got {lmax}")
        if sampling not in (1, 2):
            raise ValueError(f"sampling must be 1 or 2, got {sampling}")

        n = 2 * lmax + 2  # OK also for lmax=0 -> n=2
        nlat = n + (1 if extend else 0)
        nlon = (n if sampling == 1 else 2 * n) + (1 if extend else 0)

        dlat = 180.0 / n
        lat = 90.0 - np.arange(nlat) * dlat  # descending

        dlon = (360.0 / n) if sampling == 1 else (180.0 / n)
        lon = np.arange(nlon) * dlon

        return lat.astype(np.float64), lon.astype(np.float64)

    def _dU_grid_at_radius(self, clm_full, clm_low, L: int, r_m: float, *, sampling: int = 2):
        """
        Returns dU_grid(lat, lon) at a single radius r_m (meters), in SI units (m^2/s^2).
        Supports L=0.
        """
        L = int(L)
        if L < 0:
            raise ValueError(f"L must be >= 0, got {L}")

        # MakeGravGridDH expects cilm shaped (2, lmax+1, lmax+1) 4π-normalized
        c_full = clm_full.coeffs
        c_low = clm_low.coeffs

        # IMPORTANT: lmax_calc must be <= lmax (grid resolution) and within coeff bounds
        # Full: you currently compute up to L (as before)
        lmax_calc_full = min(L, int(clm_full.lmax))

        # Low: you were using lmax_calc=2; make it safe for small L
        lmax_calc_low = min(2, L, int(clm_low.lmax))  # <-- key change for L=0

        *_, pot_full = pysh.gravmag.MakeGravGridDH(
            c_full, clm_full.gm, clm_full.r0,
            lmax=L,
            sampling=sampling,
            lmax_calc=lmax_calc_full,
            omega=0.0,
            normal_gravity=0,
            extend=False,
            a=float(r_m),
            f=0.0,
        )

        *_, pot_low = pysh.gravmag.MakeGravGridDH(
            c_low, clm_low.gm, clm_low.r0,
            lmax=L,
            sampling=sampling,
            lmax_calc=lmax_calc_low,  # <-- key change
            omega=0.0,
            normal_gravity=0,
            extend=False,
            a=float(r_m),
            f=0.0,
        )

        return (pot_full - pot_low).astype(np.float32)

    def _build_dU_interp3d(self, clm_full, clm_low, L: int, r_grid: np.ndarray, *, sampling: int = 2):
        """
        Builds RegularGridInterpolator on (r, lat, lon) for dU.
        r_grid must be strictly increasing.
        Supports L=0.
        """
        L = int(L)

        lat_desc, lon_0_360 = self._dh_axes_from_lmax(L, sampling=sampling, extend=False)
        lat_asc = lat_desc[::-1]

        dU_cube = np.empty((len(r_grid), len(lat_asc), len(lon_0_360)), dtype=np.float32)

        for k, r_m in enumerate(r_grid):
            dU_grid_desc = self._dU_grid_at_radius(clm_full, clm_low, L, float(r_m), sampling=sampling)
            dU_cube[k] = dU_grid_desc[::-1, :]

        interp3d = RegularGridInterpolator(
            (r_grid.astype(np.float64), lat_asc, lon_0_360),
            dU_cube,
            bounds_error=False,
            fill_value=None,
        )
        return interp3d

    def generate_linear_equiv_at_points(self, L: int, df_points: pd.DataFrame, label=""):
        linear_output_path = os.path.join(self.run_dir, f"linear_{L}-2_{label}.parquet")

        clm_full = pysh.datasets.Earth.EGM2008(lmax=L)
        clm_low = pysh.datasets.Earth.EGM2008(lmax=2)

        r0 = clm_full.r0

        lat = df_points["lat"].to_numpy(dtype=np.float64)
        lon = df_points["lon"].to_numpy(dtype=np.float64)
        r = (r0 + df_points["altitude_m"].to_numpy(dtype=np.float64))
        g_full = clm_full.expand(lat=lat, lon=lon, r=r, lmax=L, degrees=True)
        g_low = clm_low.expand(lat=lat, lon=lon, r=r, lmax=2, degrees=True)

        g = g_full - g_low  # m/s^2

        dg_r = (g[:, 0] * 1e5).astype("float32")
        dg_theta = (g[:, 1] * 1e5).astype("float32")
        dg_phi = (g[:, 2] * 1e5).astype("float32")
        dg_total = np.sqrt(dg_theta ** 2 + dg_phi ** 2 + dg_r ** 2).astype("float32")

        out = pd.DataFrame({
            "lat": lat.astype("float32"),
            "lon": lon.astype("float32"),
            "altitude_m": (r - r0).astype("float32"),
            "radius_m": r.astype("float32"),
            #"dU_m2s2": dU.astype("float32"),
            "dg_r_mGal": dg_r,
            "dg_theta_mGal": dg_theta,
            "dg_phi_mGal": dg_phi,
            "dg_total_mGal": dg_total,
        })

        out.to_parquet(linear_output_path, index=False)
        return out

    def evaluate_on_test_points(
            self,
            clm_full,
            clm_low,
            r0,
            L,
            A_idx, F_idx, C_idx,
            save=True,
            label="",
    ):
        df_test = pd.read_parquet(self.data_path)
        mask = np.abs(df_test["lat"].values) < 89.9999
        df_test = df_test[mask].reset_index(drop=True)

        lat_f = df_test["lat"].values
        lon_f = df_test["lon"].values
        r_f = (r0 + df_test["altitude_m"].values).astype(lat_f.dtype)

        # --- build r_grid from the unique altitude levels in the test dataframe ---
        alt = df_test["altitude_m"].to_numpy(dtype=np.float64)
        nr = 200
        alt_levels = np.linspace(alt.min(), alt.max(), nr, dtype=np.float64)
        r_grid = (r0 + alt_levels).astype(np.float64)
        # RegularGridInterpolator requires strictly increasing
        # (alt_levels sorted -> r_grid sorted)

        # Wrap longitudes for DH grid convention (0..360)
        lon_f_0_360 = (lon_f % 360.0).astype(np.float64)

        interp3d = self._build_dU_interp3d(clm_full, clm_low, L, r_grid, sampling=2)
        dU = interp3d(np.column_stack((r_f.astype(np.float64), lat_f.astype(np.float64), lon_f_0_360))).astype(
            "float32")

        t0 = time.perf_counter()
        g_full = clm_full.expand(lat=lat_f, lon=lon_f, r=r_f, lmax=L, degrees=True)
        t_full = time.perf_counter() - t0
        t0 = time.perf_counter()
        g_low = clm_low.expand(lat=lat_f, lon=lon_f, r=r_f, lmax=2, degrees=True)
        t_low = time.perf_counter() - t0

        timing = {
            "n_points": int(len(lat_f)),
            "t_expand_full_s": float(t_full),
            "t_expand_low_s": float(t_low),
            "t_total_s": float(t_full + t_low),
        }

        g = g_full - g_low

        g_r = (g[:, 0] * 1e5).astype("float32")
        g_theta = (g[:, 1] * 1e5).astype("float32")
        g_phi = (g[:, 2] * 1e5).astype("float32")

        g_mag = np.sqrt(g_theta ** 2 + g_phi ** 2 + g_r ** 2).astype("float32")

        subsets = {"A": A_idx, "F": F_idx, "C": C_idx}
        out = {}

        for s, idx in subsets.items():
            out[s] = {
                "dU": dU[idx],
                "g_r": g_r[idx],
                "g_theta": g_theta[idx],
                "g_phi": g_phi[idx],
                "g_mag": g_mag[idx],
            }

            if save:
                np.save(f"{self.run_dir}/linear_U_{s}_{label}.npy", dU[idx])
                np.save(f"{self.run_dir}/linear_g_r_{s}_{label}.npy", g_r[idx])
                np.save(f"{self.run_dir}/linear_g_theta_{s}_{label}.npy", g_theta[idx])
                np.save(f"{self.run_dir}/linear_g_phi_{s}_{label}.npy", g_phi[idx])
                np.save(f"{self.run_dir}/linear_g_mag_{s}_{label}.npy", g_mag[idx])

        return out, timing
