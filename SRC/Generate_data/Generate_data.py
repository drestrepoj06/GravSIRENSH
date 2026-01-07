"""Generate n random samples on the sphere that contain lon, lat, r, potential and acceleration from
EGM2008. For train, n = 5 M. For test, n = 250 K distributed in a Fibonacci grid.
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
        self._clm_full = pysh.datasets.Earth.EGM2008(lmax=self.lmax_full)
        self._clm_low = pysh.datasets.Earth.EGM2008(lmax=self.lmax_base)
        self.r0 = self._clm_full.r0

        os.makedirs(self.output_dir, exist_ok=True)

        n_final = len(self.samples) if hasattr(self, "samples") else n_samples

        sample_tag = self._format_samples(n_final)

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

    def _compute_fields_on_grid(self):
        """
        Compute dU and dg components on the FULL lat/lon grid for the current self.altitude.
        Returns float32 arrays shaped (nlat, nlon).
        """
        # Use cached coeffs if you have them; otherwise replace with pysh.datasets.Earth.EGM2008(...)
        clm_full = self._clm_full
        clm_low = self._clm_low

        self.r0 = float(clm_full.r0)
        r1 = self.r0 + float(self.altitude)

        deg_full = np.arange(self.lmax_full + 1, dtype=float)
        deg_low = np.arange(self.lmax_base + 1, dtype=float)

        scale_U_full = (self.r0 / r1) ** deg_full
        scale_U_low = (self.r0 / r1) ** deg_low
        scale_g_full = (self.r0 / r1) ** (deg_full + 2)
        scale_g_low = (self.r0 / r1) ** (deg_low + 2)

        clm_full_U = self._scale_clm(clm_full, scale_U_full)
        clm_low_U = self._scale_clm(clm_low, scale_U_low)
        clm_full_g = self._scale_clm(clm_full, scale_g_full)
        clm_low_g = self._scale_clm(clm_low, scale_g_low)

        res_full_g = clm_full_g.expand(lmax=self.lmax_full)
        res_low_g = clm_low_g.expand(lmax=self.lmax_full)
        res_full_U = clm_full_U.expand(lmax=self.lmax_full)
        res_low_U = clm_low_U.expand(lmax=self.lmax_full)

        # Extract numeric arrays
        dU = (res_full_U.pot.data - res_low_U.pot.data).astype("float32")

        dg_r = ((res_full_g.rad.data - res_low_g.rad.data) * 1e5).astype("float32")
        dg_theta = ((res_full_g.theta.data - res_low_g.theta.data) * 1e5).astype("float32")
        dg_phi = ((res_full_g.phi.data - res_low_g.phi.data) * 1e5).astype("float32")

        return dU, dg_r, dg_theta, dg_phi


    def _sample_points(self, df):

        n = len(df)
        N = self.n_samples

        if self.mode == "train":
            weights = np.abs(np.cos(np.radians(df["lat"].values)))

            if n >= N:
                return df.sample(n=N, weights=weights, random_state=42).reset_index(drop=True)

            df_all = df.copy()
            df_extra = df.sample(n=(N - n), weights=weights, replace=True, random_state=42)
            return pd.concat([df_all, df_extra], ignore_index=True)

        elif self.mode == "test":
            lat_fib, lon_fib, _ = self.fibonacci_spiral_sphere(N, self.r0)

            from scipy.spatial import cKDTree
            coords_grid = np.vstack((df["lat"].values, df["lon"].values)).T
            tree = cKDTree(coords_grid)

            coords_fib = np.vstack((lat_fib, lon_fib)).T
            _, idx = tree.query(coords_fib, k=1)

            df_test = df.iloc[idx].reset_index(drop=True)

            if len(df_test) < N:
                df_extra = df_test.sample(n=(N - len(df_test)), replace=True, random_state=42)
                df_test = pd.concat([df_test, df_extra], ignore_index=True)
            elif len(df_test) > N:
                df_test = df_test.iloc[:N].reset_index(drop=True)

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

    def generate_uniform_altitude_file(
            self,
            *,
            max_alt_m=420_000.0,
            n_alt_shells=200,
            seed=42,
    ):
        rng = np.random.default_rng(seed)
        N = int(self.n_samples)

        saved_alt = float(self.altitude)

        # ------------------------------------------------------------
        # 1) Build grid ONCE (alt=0 only to obtain lats/lons)
        # ------------------------------------------------------------
        self.altitude = 0.0

        deg_full = np.arange(self.lmax_full + 1, dtype=float)
        scale_ones = np.ones_like(deg_full, dtype=float)
        clm_full_U0 = self._scale_clm(self._clm_full, scale_ones)

        res0 = clm_full_U0.expand(lmax=self.lmax_full)
        lats0 = res0.pot.lats()
        lons0 = res0.pot.lons()

        nlat = len(lats0)
        nlon = len(lons0)

        lat_grid = np.repeat(lats0, nlon).astype("float32")
        lon_grid = np.tile(lons0, nlat).astype("float32")

        lat_i_grid = np.repeat(np.arange(nlat, dtype=np.int32), nlon)
        lon_i_grid = np.tile(np.arange(nlon, dtype=np.int32), nlat)

        df_grid = pd.DataFrame({
            "lat": lat_grid,
            "lon": lon_grid,
            "lat_i": lat_i_grid,
            "lon_i": lon_i_grid,
        })
        df_grid = df_grid[np.abs(df_grid["lat"].values) < 89.9999].reset_index(drop=True)

        df_ll = self._sample_points(df_grid)

        lat = df_ll["lat"].to_numpy(dtype="float32")
        lon = (df_ll["lon"].to_numpy(dtype="float32") % 360.0).astype("float32")

        lat_i = df_ll["lat_i"].to_numpy(np.int32)
        lon_i = df_ll["lon_i"].to_numpy(np.int32)
        flat_i = lat_i * nlon + lon_i  # flat grid indices for each sampled point

        # Restore altitude state
        self.altitude = saved_alt

        # ------------------------------------------------------------
        # 2) Sample continuous altitude
        # ------------------------------------------------------------
        altitude = rng.uniform(0.0, float(max_alt_m), size=N).astype("float32")

        if int(n_alt_shells) < 2:
            raise ValueError("Interpolation requires n_alt_shells >= 2.")

        shells = np.linspace(0.0, float(max_alt_m), int(n_alt_shells)).astype("float32")

        # ------------------------------------------------------------
        # 3) Bracketing shells (lower/upper) + alpha
        # ------------------------------------------------------------
        lower_idx = np.searchsorted(shells, altitude, side="right") - 1
        lower_idx = np.clip(lower_idx, 0, len(shells) - 1)

        upper_idx = np.clip(lower_idx + 1, 0, len(shells) - 1)

        h_lo = shells[lower_idx]
        h_hi = shells[upper_idx]

        denom = (h_hi - h_lo).astype("float32")
        alpha = np.zeros(N, dtype="float32")
        m = denom > 0
        alpha[m] = ((altitude[m] - h_lo[m]) / denom[m]).astype("float32")

        # ------------------------------------------------------------
        # 4) Allocate lower/upper arrays
        # ------------------------------------------------------------
        dU_lo = np.empty(N, dtype="float32")
        dg_r_lo = np.empty(N, dtype="float32")
        dg_theta_lo = np.empty(N, dtype="float32")
        dg_phi_lo = np.empty(N, dtype="float32")

        dU_hi = np.empty(N, dtype="float32")
        dg_r_hi = np.empty(N, dtype="float32")
        dg_theta_hi = np.empty(N, dtype="float32")
        dg_phi_hi = np.empty(N, dtype="float32")

        # All shell indices we must compute
        shells_needed = np.unique(np.concatenate([lower_idx, upper_idx]))

        # ------------------------------------------------------------
        # 5) Compute each needed shell ONCE on the full grid, then sample
        # ------------------------------------------------------------
        for k in shells_needed:
            idx_need = np.where((lower_idx == k) | (upper_idx == k))[0]
            if idx_need.size == 0:
                continue

            self.altitude = float(shells[k])

            dU_grid, dg_r_grid, dg_theta_grid, dg_phi_grid = self._compute_fields_on_grid()

            # Flat-sample the grid for these points
            fi = flat_i[idx_need]
            dU_vals = dU_grid.ravel()[fi]
            dg_r_vals = dg_r_grid.ravel()[fi]
            dg_theta_vals = dg_theta_grid.ravel()[fi]
            dg_phi_vals = dg_phi_grid.ravel()[fi]

            # Scatter into lower/upper arrays
            is_lo = (lower_idx[idx_need] == k)
            is_hi = (upper_idx[idx_need] == k)

            if np.any(is_lo):
                sel = idx_need[is_lo]
                dU_lo[sel] = dU_vals[is_lo]
                dg_r_lo[sel] = dg_r_vals[is_lo]
                dg_theta_lo[sel] = dg_theta_vals[is_lo]
                dg_phi_lo[sel] = dg_phi_vals[is_lo]

            if np.any(is_hi):
                sel = idx_need[is_hi]
                dU_hi[sel] = dU_vals[is_hi]
                dg_r_hi[sel] = dg_r_vals[is_hi]
                dg_theta_hi[sel] = dg_theta_vals[is_hi]
                dg_phi_hi[sel] = dg_phi_vals[is_hi]

        # ------------------------------------------------------------
        # 6) Interpolate to true altitude
        # ------------------------------------------------------------
        dU_out = (dU_lo + alpha * (dU_hi - dU_lo)).astype("float32")
        dg_r_out = (dg_r_lo + alpha * (dg_r_hi - dg_r_lo)).astype("float32")
        dg_theta_out = (dg_theta_lo + alpha * (dg_theta_hi - dg_theta_lo)).astype("float32")
        dg_phi_out = (dg_phi_lo + alpha * (dg_phi_hi - dg_phi_lo)).astype("float32")

        dg_total_out = np.sqrt(dg_r_out ** 2 + dg_theta_out ** 2 + dg_phi_out ** 2).astype("float32")

        # ------------------------------------------------------------
        # 7) Save
        # ------------------------------------------------------------
        r0 = float(self.r0)
        radius = (r0 + altitude).astype("float32")

        df = pd.DataFrame({
            "lat": lat,
            "lon": lon,
            "altitude_m": altitude,
            "radius_m": radius,
            "r0_ref_m": np.full(N, r0, dtype="float32"),

            "dU_m2_s2": dU_out,
            "dg_r_mGal": dg_r_out,
            "dg_theta_mGal": dg_theta_out,
            "dg_phi_mGal": dg_phi_out,
            "dg_total_mGal": dg_total_out,

            "shell_lower_m": h_lo.astype("float32"),
            "shell_upper_m": h_hi.astype("float32"),
            "alpha_r": alpha.astype("float32"),
        })

        out_path = os.path.join(
            self.output_dir,
            f"Samples_{self.lmax_full}-{self.lmax_base}_{self._format_samples(N)}_altUniform0-{int(max_alt_m)}_{self.mode}_shells_{int(n_alt_shells)}_interp.parquet"
        )
        df.to_parquet(out_path, index=False)

        self.altitude = saved_alt
        return df

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    lmax_full = 2190
    lmax_base = 2
    max_altitude = 420_000.0


    generator_train = GravityDataGenerator(
            lmax_full=lmax_full,
            lmax_base=lmax_base,
            n_samples=5_000_000,
            mode="train",
            output_dir=data_dir,
            altitude=0.0,
    )

    generator_train.generate_uniform_altitude_file(
            max_alt_m=max_altitude,
            n_alt_shells=2,
            seed=42,
    )

    generator_test = GravityDataGenerator(
            lmax_full=lmax_full,
            lmax_base=lmax_base,
            n_samples=250_000,
            mode="test",
            output_dir=data_dir,
            altitude=0.0,
    )

    generator_test.generate_uniform_altitude_file(
            max_alt_m=max_altitude,
            n_alt_shells=2,
            seed=123,
    )


if __name__ == "__main__":
    main()