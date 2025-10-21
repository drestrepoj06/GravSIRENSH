"""Creation of spherical harmonic basis functions and store them in cache
jhonr"""

import numpy as np
import pandas as pd
import pyshtools as pysh
import os

class SHEmbedding:
    def __init__(self, lmax: int = 10, normalization: str = "4pi", cache_path: str = None):
        self.lmax = lmax
        self.normalization = normalization
        self.cache_path = cache_path  # optional .npy or .parquet file

    def compute_basis(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        n_points = len(lat)
        n_basis = (self.lmax + 1) ** 2
        Y = np.zeros((n_points, n_basis))
        theta = 90.0 - lat

        for i, (th, ph) in enumerate(zip(theta, lon)):
            ylm = pysh.expand.spharm(
                self.lmax, th, ph,
                normalization=self.normalization,
                kind="real"
            )

            y_list = []
            for l in range(self.lmax + 1):
                for m in range(l, 0, -1):
                    y_list.append(ylm[1, l, m])
                y_list.append(ylm[0, l, 0])
                for m in range(1, l + 1):
                    y_list.append(ylm[0, l, m])
            Y[i, :] = np.array(y_list)
        return Y

    def from_dataframe(self, df: pd.DataFrame, lon_col='lon', lat_col='lat', use_cache=True) -> np.ndarray:
        lon = df[lon_col].values
        lat = df[lat_col].values

        # --- Load from cache if available ---
        if use_cache and self.cache_path and os.path.exists(self.cache_path):
            print(f"📂 Loading cached SH basis from {self.cache_path}")
            return np.load(self.cache_path)

        print("⚙️ Computing spherical harmonic basis...")
        Y = self.compute_basis(lon, lat)

        if self.cache_path:
            np.save(self.cache_path, Y)
            print(f"💾 Cached SH basis saved to {self.cache_path}")

        return Y

