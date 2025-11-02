"""
SH-SIREN: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Location_encoder.SH_embedding import SHEmbedding
from SRC.Location_encoder.Siren import SIRENNet
import numpy as np

# Scaling target outputs in the range [-1, 1],
# based on the scaling preferred for SIRENNETS: https://github.com/vsitzmann/siren
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/explore_siren.ipynb
class SHSirenScaler:
    """
    Handles scaling of potential U and rescaling of its gradients
    (autograd-compatible for computing physical accelerations).
    """
    def __init__(self, r_scale=None, U_min=None, U_max=None):
        self.r_scale = r_scale
        self.U_min = U_min
        self.U_max = U_max
        self.a_scale = None

    # --- fit & scaling of potential ------------------------------------------
    def fit_potential(self, U_np):
        self.U_min = float(np.min(U_np))
        self.U_max = float(np.max(U_np))
        self.u_scale = 0.5 * (self.U_max - self.U_min)
        return self

    def scale_potential(self, U):
        return 2 * (U - self.U_min) / (self.U_max - self.U_min) - 1

    def unscale_potential(self, U_scaled):
        return (U_scaled + 1) * 0.5 * (self.U_max - self.U_min) + self.U_min

    # --- unscaling of gradients ----------------------------------------------
    def unscale_acceleration(self, grads):
        """
        grads: tuple of torch tensors (dU_dlon, dU_dlat, [dU_dr]) in model space.
        Returns gradients in physical units (same shape).
        """
        if self.U_min is None or self.U_max is None:
            raise ValueError("Scaler not fitted. Call fit_potential(U) first.")

        S = 0.5 * (self.U_max - self.U_min)
        deg2rad = np.pi / 180.0

        dU_dlon, dU_dlat, *rest = grads

        # longitude and latitude derivatives → multiply by angular to linear conversion
        dU_dlon_phys = dU_dlon * (S * deg2rad)
        dU_dlat_phys = dU_dlat * (S * deg2rad)

        if rest:
            dU_dr = rest[0]
            # If you’re working on a spherical shell (fixed r), keep zero radial derivative
            if dU_dr is not None:
                dU_dr_phys = dU_dr * (S / (self.r_scale or 1.0))
            else:
                dU_dr_phys = torch.zeros_like(dU_dlon)
        else:
            dU_dr_phys = None

        return (dU_dlon_phys, dU_dlat_phys, dU_dr_phys)

# Based on the code https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/locationencoder.py
# But only for the encoder SH + Siren network

class SH_SIREN(nn.Module):
    def __init__(self, lmax=10, hidden_features=128, hidden_layers=4, out_features=1,
                 first_omega_0=30.0, hidden_omega_0=1.0, device='cuda',
                 normalization="4pi", cache_path=None, scaler=None,
                 df=None):
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.lmax = lmax
        self.normalization = normalization
        self.cache_path = cache_path

        # Create embedding
        self.embedding = SHEmbedding(
            lmax=lmax,
            normalization=normalization,
            cache_path=cache_path,
            use_theta_lut=True,
            n_theta=18001,
        )

        # Define SIREN
        n_basis = (lmax + 1) ** 2
        self.siren = SIRENNet(
            in_features=n_basis,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        ).to(device)

    # -----------------------------
    # Forward pipeline
    # -----------------------------
    def forward(self, lon, lat, return_gradients=False,r = None, idx = None):
        """
        lon, lat: torch tensors in degrees
        return_gradients: if True, return U and grads (autograd)
        """
        # build differentiable embedding
        Y = self.embedding(lon, lat, r, idx=idx)
        Y = Y.to(self.device)

        # forward pass
        U_scaled = self.siren(Y)

        if not return_gradients:
            return U_scaled

        # compute autograd gradients (acceleration)
        grads = torch.autograd.grad(
            outputs=U_scaled,
            inputs=[lon, lat, r] if r is not None else [lon, lat],
            grad_outputs=torch.ones_like(U_scaled),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )

        # rescale to physical units
        grads_phys = self.scaler.unscale_acceleration(grads)
        return U_scaled, grads_phys



