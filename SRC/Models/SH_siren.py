"""
SH-SIREN: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Models.SH_embedding import SHEmbedding
from SRC.Models.Siren import SIRENNet
import numpy as np
import os

# Scaling acceleration outputs in the range [-1, 1],
# based on the scaling preferred for SIRENNETS: https://github.com/vsitzmann/siren
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/explore_siren.ipynb
class SHSirenScaler:

    def __init__(self, r_scale=None, a_min=None, a_max=None):
        self.r_scale = r_scale
        self.a_min = a_min
        self.a_max = a_max

    def scale_inputs(self, lon, lat, r):
        if self.r_scale:
            r = r / self.r_scale
        return lon, lat, r

    def fit_acceleration(self, a_components):
        a_all = np.concatenate([np.ravel(a) for a in np.atleast_2d(a_components).T], axis=0)
        self.a_min = np.min(a_all)
        self.a_max = np.max(a_all)
        return self

    def scale_acceleration(self, a_components):
        if self.a_min is None or self.a_max is None:
            raise ValueError("Call fit_acceleration() before scaling.")
        return 2 * (a_components - self.a_min) / (self.a_max - self.a_min) - 1

    def unscale_acceleration(self, a_scaled):
        if self.a_min is None or self.a_max is None:
            raise ValueError("Scaler not fitted.")
        return (a_scaled + 1) * 0.5 * (self.a_max - self.a_min) + self.a_min


# Based on the code https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/locationencoder.py
# But only for the encoder SH + Siren network

class SH_SIREN(nn.Module):

    def __init__(self, lmax=10, hidden_features=128, hidden_layers=4, out_features=1,
                 first_omega_0=30.0, hidden_omega_0=1.0, device='cuda',
                 normalization="4pi", cache_path=None, scaler=None):
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.lmax = lmax
        self.normalization = normalization
        self.cache_path = cache_path

        self.embedding = SHEmbedding(lmax=lmax, normalization=normalization, cache_path=cache_path)

        n_basis = (lmax + 1) ** 2

        self.siren = SIRENNet(
            in_features=n_basis,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,   # <-- 3 components: a_r, a_theta, a_phi
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        ).to(device)

    def prepare_input(self, df, lon_col='lon', lat_col='lat', use_cache=True):
        if use_cache:
            if "orig_index" in df.columns:
                idx = df["orig_index"].values
            else:
                idx = df.index.values
            base, ext = os.path.splitext(self.embedding.cache_path or "cache_basis")
            if not ext:
                ext = ".npy"
            cache_file = f"{base}_lmax{self.lmax}{ext}"  # ✅ ensures correct suffix
            Y_memmap = np.load(cache_file, mmap_mode="r")
            Y = Y_memmap[idx]
        else:
            Y = self.embedding.from_dataframe(df, lon_col=lon_col, lat_col=lat_col, use_cache=False)

        Y_torch = torch.tensor(Y, dtype=torch.float32, device=self.device)
        return Y_torch

    def forward(self, df=None, Y=None, return_gradients=False):
        if Y is None:
            if df is None:
                raise ValueError("Provide either DataFrame (df) or precomputed SH basis (Y).")
            Y = self.prepare_input(df, use_cache=True)

        if return_gradients:
            raise RuntimeError("Gradients unavailable: SH basis is precomputed and non-differentiable.")

        a_scaled = self.siren(Y)  # this is what we train on (requires grad)
        return a_scaled



