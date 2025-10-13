# -*- coding: utf-8 -*-
"""
SH-SIREN: Combines Spherical Harmonics embedding with a SIREN network.
"""

import torch
import torch.nn as nn
from SRC.Models.SH_embedding import SHEmbedding
from SRC.Models.Siren import SIRENNet

class SH_SIREN(nn.Module):
    def __init__(self, lmax=10, hidden_features=128, hidden_layers=4, out_features=1,
                 first_omega_0=30, hidden_omega_0=1.0):
        """
        Combined model that uses spherical harmonic embeddings as input to a SIREN network.
        """
        super().__init__()
        self.embedding = SHEmbedding(lmax=lmax)
        n_basis = (lmax + 1)**2

        self.siren = SIRENNet(
            in_features=n_basis,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )

    def forward(self, lon, lat):
        """
        Compute SH embedding and forward through SIREN.
        Efficient hybrid CPU–GPU version.
        """
        # Ensure tensors
        if not isinstance(lon, torch.Tensor):
            lon = torch.tensor(lon, dtype=torch.float32)
        if not isinstance(lat, torch.Tensor):
            lat = torch.tensor(lat, dtype=torch.float32)
    
        # --- CPU compute only once per batch ---
        with torch.no_grad():  # embedding has no gradients
            Y_np = self.embedding.compute_basis(
                lon.detach().cpu().numpy(),
                lat.detach().cpu().numpy()
            )
    
        # --- Convert back to GPU once per batch ---
        Y = torch.from_numpy(Y_np).float().to(lon.device)
    
        # Forward through the SIREN on GPU
        return self.siren(Y)
