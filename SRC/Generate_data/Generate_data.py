"""Generate n random samples on the sphere that contain lon, lat, r, potential and acceleration from
EGM2008
jhonr"""

import os
import numpy as np
import pandas as pd
import pyshtools as pysh

class GravityDataGenerator:
    def __init__(self, lmax_full, lmax_base, n_samples, mode="train", output_dir="Data", altitude=0.0):
        self.lmax_full = lmax_full
        self.lmax_base = lmax_base
        self.n_samples = n_samples
        self.mode = mode.lower()
        self.output_dir = output_dir
        self.altitude = altitude

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(
            self.output_dir,
            f"Samples_{lmax_full}_{self._format_samples(n_samples)}_r{int(self.altitude)}_{self.mode}.parquet"
        )

    @staticmethod
    def _format_samples(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M".rstrip("0").rstrip(".")
        elif n >= 1_000:
            return f"{n / 1_000:.0f}k"
        else:
            return str(n)


    def _scale_clm(self, clm, scale):
        scaled = clm.copy()
        scaled.coeffs *= scale[np.newaxis, :, np.newaxis]
        return scaled

    def _compute_fields(self):
        clm_full = pysh.datasets.Earth.EGM2008(lmax=self.lmax_full)
        clm_low = pysh.datasets.Earth.EGM2008(lmax=self.lmax_base)

        self.r0 = clm_full.r0  # ✅ store as class attribute
        r1 = self.r0 + self.altitude

        deg_full = np.arange(self.lmax_full + 1)
        deg_low = np.arange(self.lmax_base + 1)

        scale_V_full = (self.r0 / r1) ** deg_full
        scale_V_low = (self.r0 / r1) ** deg_low
        scale_g_full = (self.r0 / r1) ** (deg_full + 2)
        scale_g_low = (self.r0 / r1) ** (deg_low + 2)

        clm_full_V = self._scale_clm(clm_full, scale_V_full)
        clm_low_V = self._scale_clm(clm_low, scale_V_low)
        clm_full_g = self._scale_clm(clm_full, scale_g_full)
        clm_low_g = self._scale_clm(clm_low, scale_g_low)

        res_full = clm_full_g.expand(lmax=self.lmax_full)
        res_low = clm_low_g.expand(lmax=self.lmax_full)

        res_full_V = clm_full_V.expand(lmax=self.lmax_full)
        res_low_V = clm_low_V.expand(lmax=self.lmax_full)

        return res_full, res_low, res_full_V, res_low_V

    def _sample_points(self, df):
        if len(df) <= self.n_samples:
            return df

        if self.mode == "train":
            weights = np.abs(np.cos(np.radians(df["lat"].values)))
            return df.sample(n=self.n_samples, weights=weights, random_state=42).reset_index(drop=True)

        elif self.mode == "test":
            lat_fib, lon_fib, _ = self.fibonacci_spiral_sphere(self.n_samples, self.r0)
            from scipy.spatial import cKDTree
            # From the pysh.expand grid, give the closest coordinates to a Fibonacci grid
            coords_grid = np.vstack((df["lat"].values, df["lon"].values)).T
            tree = cKDTree(coords_grid)
            coords_fib = np.vstack((lat_fib, lon_fib)).T
            _, idx = tree.query(coords_fib, k=1)

            df_test = df.iloc[idx].reset_index(drop=True)
            return df_test

        else:
            raise ValueError("mode must be 'train' or 'test'")

    # From https://github.com/MartinAstro/GravNN/blob/master/GravNN/Trajectories/FibonacciDist.py
    # Line 10
    def fibonacci_spiral_sphere(self, num_points, r):
        gr = (np.sqrt(5.0) + 1.0) / 2.0  # golden ratio
        ga = (2.0 - gr) * (2.0 * np.pi)  # golden angle

        lat = np.arcsin(-1.0 + 2.0 * np.arange(num_points) / num_points)
        lon = ga * np.arange(num_points)

        x = r * np.cos(lon) * np.cos(lat)
        y = r * np.sin(lon) * np.cos(lat)
        z = r * np.sin(lat)

        lat_deg = np.degrees(lat)
        lon_deg = np.degrees(lon % (2 * np.pi))
        return lat_deg, lon_deg, np.vstack((x, y, z)).T

    def generate(self):
        res_full, res_low, res_full_V, res_low_V = self._compute_fields()
        r0 = self.r0

        pot_full = res_full_V.pot.data
        pot_low = res_low_V.pot.data
        dV = pot_full - pot_low

        gr_full = res_full.rad.data * 1e5
        gr_low = res_low.rad.data * 1e5
        gtheta_full = res_full.theta.data * 1e5
        gtheta_low = res_low.theta.data * 1e5
        gphi_full = res_full.phi.data * 1e5
        gphi_low = res_low.phi.data * 1e5

        dg_r = gr_full - gr_low
        dg_theta = gtheta_full - gtheta_low
        dg_phi = gphi_full - gphi_low
        dg_total = np.sqrt(dg_r**2 + dg_theta**2 + dg_phi**2)

        lats = res_full.pot.lats()
        lons = res_full.pot.lons()
        lat_grid = np.repeat(lats, len(lons))
        lon_grid = np.tile(lons, len(lats))

        df = pd.DataFrame({
            "lat": lat_grid.astype("float32"),
            "lon": lon_grid.astype("float32"),
            "dV_m2_s2": dV.ravel().astype("float32"),
            "dg_r_mGal": dg_r.ravel().astype("float32"),
            "dg_theta_mGal": dg_theta.ravel().astype("float32"),
            "dg_phi_mGal": dg_phi.ravel().astype("float32"),
            "dg_total_mGal": dg_total.ravel().astype("float32"),
            "radius_m": np.full(dV.size, r0, dtype="float32")
        })

        df = self._sample_points(df)
        df.to_parquet(self.output_file, index=False)
        print(f"✅ Saved: {self.output_file}")
        return df