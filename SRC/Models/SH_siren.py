"""
SH-SIREN: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Models.SH_embedding import SHEmbedding
from SRC.Models.Siren import SIRENNet
import numpy as np

# Scaling potential outputs in the range [-1, 1],
# based on the code https://github.com/MartinAstro/GravNN/blob/master/GravNN/Networks/Data.py Line 91
# And https://github.com/MartinAstro/GravNN/blob/5debb42013097944c0398fe5b570d7cd9ebd43bd/GravNN/Preprocessors/UniformScaler.py

class SHSirenScaler:

    def __init__(self, r_scale=None, U_min=None, U_max=None):
        self.r_scale = r_scale

        self.U_min = U_min
        self.U_max = U_max

        # Derived acceleration scale (if needed)
        if self.U_max is not None and self.U_min is not None and r_scale is not None:
            u_scale = max(abs(self.U_min), abs(self.U_max))
            self.a_scale = u_scale / r_scale
        else:
            self.a_scale = None

    def scale_inputs(self, lon, lat, r):
        if self.r_scale:
            r = r / self.r_scale
        return lon, lat, r

    def fit_potential(self, U):
        self.U_min = np.min(U)
        self.U_max = np.max(U)
        return self

    def scale_potential(self, U):
        if self.U_min is None or self.U_max is None:
            raise ValueError("Call fit_potential(U) before scaling.")
        U_scaled = 2 * (U - self.U_min) / (self.U_max - self.U_min) - 1
        return U_scaled

    def unscale_potential(self, U_scaled):
        if self.U_min is None or self.U_max is None:
            raise ValueError("Scaler not fitted.")
        U = (U_scaled + 1) * 0.5 * (self.U_max - self.U_min) + self.U_min
        return U

    def unscale_acceleration(self, grads):
        if self.a_scale:
            return tuple(g * self.a_scale for g in grads)
        return grads


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
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        ).to(device)

    def prepare_input(self, df, lon_col='lon', lat_col='lat', use_cache=True):
        Y = self.embedding.from_dataframe(df, lon_col=lon_col, lat_col=lat_col, use_cache=use_cache)
        Y_torch = torch.from_numpy(Y).float().to(self.device)
        return Y_torch

    def forward(self, df=None, Y=None, return_gradients=False):
        if Y is None:
            if df is None:
                raise ValueError("Provide either DataFrame (df) or precomputed SH basis (Y).")
            Y = self.prepare_input(df)

        if return_gradients:
            raise RuntimeError("Gradients unavailable: SH basis is precomputed and non-differentiable.")

        U_scaled = self.siren(Y)

        if self.scaler is not None:
            U_phys = self.scaler.unscale_potential(U_scaled.detach().cpu().numpy())
            return torch.from_numpy(U_phys).float().to(self.device)
        return U_scaled


