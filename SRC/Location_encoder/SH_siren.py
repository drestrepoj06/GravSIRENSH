"""
SH-SIREN: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Location_encoder.SH_embedding import SphericalHarmonics
from SRC.Location_encoder.Siren import SIRENNet
import numpy as np

# Scaling on potential, based on the code https://github.com/MartinAstro/GravNN/blob/master/GravNN/Networks/Data.py Line 91
# And https://github.com/MartinAstro/GravNN/blob/5debb42013097944c0398fe5b570d7cd9ebd43bd/GravNN/Preprocessors/UniformScaler.py

class SHSirenScaler:
    def __init__(self, r_scale=None, U_min=None, U_max=None):
        self.r_scale = r_scale
        self.U_min = U_min
        self.U_max = U_max
        self.a_scale = None

    def fit_potential(self, U):
        self.U_min = np.min(U)
        self.U_max = np.max(U)
        self.u_scale = 0.5 * (self.U_max - self.U_min)
        return self

    def scale_potential(self, U):
        return 2 * (U - self.U_min) / (self.U_max - self.U_min) - 1

    def unscale_potential(self, U_scaled):
        return (U_scaled + 1) * 0.5 * (self.U_max - self.U_min) + self.U_min

    def unscale_acceleration(self, grads):
        if self.U_min is None or self.U_max is None:
            raise ValueError("Scaler not fitted.")
        S = 0.5 * (self.U_max - self.U_min)
        deg2rad = np.pi / 180.0
        # grads = (dU_dlon, dU_dlat, dU_dr)
        dU_dlon, dU_dlat, *rest = grads

        dU_dlon_phys = dU_dlon * (S * deg2rad)
        dU_dlat_phys = dU_dlat * (S * deg2rad)
        if rest:
            dU_dr = rest[0]
            dU_dr_phys = torch.zeros_like(dU_dr)  # no radial derivative at surface
        else:
            dU_dr_phys = None
        return (dU_dlon_phys, dU_dlat_phys, dU_dr_phys)


# Based on the code https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/locationencoder.py
# But only for the encoder SH + Siren network
# For the output of potential + autograd acceleration, see "Martin, J., & Schaub, H. (2022a). Physics-informed neural networks for gravity field 
# modeling of small bodies. Celestial Mechanics and Dynamical Astronomy, 134(5), 46. https://doi.org/10.1007/s10569-022-10101-8"

class SH_SIREN(nn.Module):
    def __init__(self, lmax=10, hidden_features=128, hidden_layers=4, out_features=1,
                 first_omega_0=30, hidden_omega_0=1.0, device='cuda',
                 normalize_input=True, global_scale=None, scaler=None):
        """
        Combined model that uses a differentiable SH embedding (r, lon, lat)
        as input to a SIREN network that predicts potential U(r,lon,lat) with the 
        option to compute its gradients.

        Parameters
        ----------
        scaler : SHSirenScaler, optional
            Defines how to non-dimensionalize and recover physical units.
        """
        super().__init__()
        self.device = device
        self.embedding = SphericalHarmonics(lmax=lmax, device=device)
        n_basis = (lmax + 1) ** 2

        self.siren = SIRENNet(
            in_features=n_basis,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        ).to(device)

        self.normalize_input = normalize_input
        self.global_scale = global_scale
        self.scaler = scaler  # optional SHSirenScaler instance

    def forward(self, lon, lat, r=None, return_gradients=False, physical_units=True):
        """
        Parameters
        ----------
        lon, lat : [N] tensors in degrees
        r : [N] tensor (meters or scaled)
        return_gradients : bool
            Whether to compute and return dU/dlon, dU/dlat, dU/dr
        physical_units : bool
            If True, return gradients in m/s² (physical accelerations)
            instead of dimensionless values.

        Returns
        -------
        U : [N,1] tensor (potential)
        grads : tuple(dU/dlon, dU/dlat, dU/dr)
        """
        lon = torch.as_tensor(lon, dtype=torch.float32, device=self.device)
        lat = torch.as_tensor(lat, dtype=torch.float32, device=self.device)
        
        if r is None:
            r = torch.ones_like(lon) * getattr(self.embedding, "r_ref", 1.0)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)

        if return_gradients:
            lon.requires_grad_(True)
            lat.requires_grad_(True)

        r_used = r

        lonlatr = torch.stack([lon, lat, r_used], dim=-1)
        Y = self.embedding(lonlatr)
        U_scaled = self.siren(Y)

        if not return_gradients:
            return U_scaled

        dU_dlon, dU_dlat = torch.autograd.grad(
            U_scaled.sum(), [lon, lat], create_graph=True
        )
        # fabricate a zero tensor for dU/dr for compatibility
        dU_dr = torch.zeros_like(r)
        grads_scaled = (dU_dlon, dU_dlat, dU_dr)

        if physical_units and self.scaler is not None:
            grads_phys = self.scaler.unscale_acceleration(grads_scaled)
            U_phys = self.scaler.unscale_potential(U_scaled)
            return U_phys, grads_phys
        else:
            return U_scaled, grads_scaled


