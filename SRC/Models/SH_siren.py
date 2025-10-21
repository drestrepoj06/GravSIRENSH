"""
SH-SIREN: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Models.SH_embedding import SphericalHarmonics
from SRC.Models.Siren import SIRENNet
import numpy as np

# Scaling potential outputs in the range [-1, 1],
# based on the code https://github.com/MartinAstro/GravNN/blob/master/GravNN/Networks/Data.py Line 91
# And https://github.com/MartinAstro/GravNN/blob/5debb42013097944c0398fe5b570d7cd9ebd43bd/GravNN/Preprocessors/UniformScaler.py

class SHSirenScaler:
    """
    Hybrid scaler for SIREN-based gravity models.
    - Keeps physical consistency (r and acceleration scaling)
    - Applies uniform [-1, 1] scaling to the potential for stable training

    Equations:
      r' = r / r_scale
      U' = 2 * (U - U_min) / (U_max - U_min) - 1
      a' = a * (u_scale / r_scale)
    """

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
        """Normalize radius; lon/lat unchanged (still in degrees or radians)."""
        if self.r_scale:
            r = r / self.r_scale
        return lon, lat, r

    def fit_potential(self, U):
        """Compute and store the min/max for uniform scaling."""
        self.U_min = np.min(U)
        self.U_max = np.max(U)
        return self

    def scale_potential(self, U):
        """Uniformly scale potential to [-1, 1]."""
        if self.U_min is None or self.U_max is None:
            raise ValueError("Call fit_potential(U) before scaling.")
        U_scaled = 2 * (U - self.U_min) / (self.U_max - self.U_min) - 1
        return U_scaled

    def unscale_potential(self, U_scaled):
        """Recover potential in physical units."""
        if self.U_min is None or self.U_max is None:
            raise ValueError("Scaler not fitted.")
        U = (U_scaled + 1) * 0.5 * (self.U_max - self.U_min) + self.U_min
        return U

    def unscale_acceleration(self, grads):
        """Unscale gradients (∇U) to physical accelerations."""
        if self.a_scale:
            return tuple(g * self.a_scale for g in grads)
        return grads


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
            r.requires_grad_(True)

        if self.scaler is not None:
            _, _, r_bar = self.scaler.scale_inputs(lon, lat, r)  
            r_phys = r_bar * self.scaler.r_scale                
        else:
            r_phys = r

        lonlatr = torch.stack([lon, lat, r_phys], dim=-1)
        Y = self.embedding(lonlatr)

        U_scaled = self.siren(Y)

        if not return_gradients:
            return U_scaled

        dU_dlon, dU_dlat, dU_dr = torch.autograd.grad(
            U_scaled.sum(), [lon, lat, r], create_graph=True
        )

        grads_scaled = (dU_dlon, dU_dlat, dU_dr)

        if physical_units and self.scaler is not None:
            grads_phys = self.scaler.unscale_acceleration(grads_scaled)
            U_phys = self.scaler.unscale_potential(U_scaled)
            return U_phys, grads_phys
        else:
            return U_scaled, grads_scaled


