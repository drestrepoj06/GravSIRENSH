"""Creation of closed-form spherical harmonics in a torch-based manner
jhonr"""

import math
import torch
from torch import nn

# Closed form of https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/pe/spherical_harmonics_closed_form.py
def associated_legendre_polynomial(l, m, x):
    """Compute P_l^m(x) using the stable recursion (real-valued)."""
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt(torch.clamp((1 - x) * (1 + x), min=0))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def SH_renormalization(l, m, fourpi=True):
    """Normalization constant for spherical harmonics."""
    base = math.sqrt(
        (2.0 * l + 1.0)
        * math.factorial(l - abs(m))
        / (4.0 * math.pi * math.factorial(l + abs(m)))
    )
    if fourpi:
        base *= math.sqrt(4.0 * math.pi)
    return base

def SH(m, l, phi, theta):
    """Real-valued spherical harmonic Y_lm(phi, theta) following 4π normalization."""
    cos_theta = torch.clamp(torch.cos(theta), -1.0 + 1e-7, 1.0 - 1e-7)
    P_lm = associated_legendre_polynomial(l, abs(m), cos_theta)
    K_lm = SH_renormalization(l, m)

    if m == 0:
        return K_lm * P_lm
    elif m > 0:
        return math.sqrt(2.0) * K_lm * torch.cos(m * phi) * P_lm
    else:  # m < 0
        return math.sqrt(2.0) * K_lm * torch.sin(-m * phi) * P_lm


# Differentiable SH based on https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/pe/spherical_harmonics.py
# And adding a new radius input, apart from the lat and lon coordinates
class SphericalHarmonics(nn.Module):

    def __init__(self, lmax: int = 10, exclude_degrees: int = 0,
                 device: str = "cuda", r_ref: float = 6378136.3):
        super().__init__()
        self.lmax = lmax
        self.exclude_degrees = exclude_degrees
        self.device = device
        self.r_ref = r_ref

        # Build degree list for ALL harmonics (full basis)
        deg_list = []
        for l in range(lmax + 1):
            deg_list.extend([l] * (2 * l + 1))

        deg_list = torch.tensor(deg_list, dtype=torch.float32)

        if exclude_degrees is None:
            mask = torch.ones_like(deg_list, dtype=torch.bool)
        else:
            mask = deg_list > exclude_degrees

        # Store mask and filtered degree indices
        self.register_buffer("deg_idx_full", deg_list)
        self.register_buffer("mask", mask)
        self.register_buffer("deg_idx", deg_list[mask])

        # Final embedding dimension AFTER removal
        self.embedding_dim = int(mask.sum().item())
        print(f"🔹 Effective embedding size: {self.embedding_dim}")

    def forward(self, lonlatr: torch.Tensor) -> torch.Tensor:
        if lonlatr.shape[1] == 2:
            lon, lat = lonlatr[:, 0], lonlatr[:, 1]
            r = torch.ones_like(lon) * self.r_ref
        else:
            lon, lat, r = lonlatr[:, 0], lonlatr[:, 1], lonlatr[:, 2]

        phi = torch.deg2rad(lon + 180.0)
        theta = torch.deg2rad(90.0 - lat)
        theta = torch.clamp(theta, 1e-7, math.pi - 1e-7)

        # Build FULL spherical harmonics
        Y_full = []
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                Y_full.append(SH(m, l, phi, theta))
        Y_full = torch.stack(Y_full, dim=-1)  # (N, full_dim)

        # Filter out low degrees
        Y = Y_full[:, self.mask]

        # Apply radial factor only for remaining degrees
        radial_factor = (self.r_ref / r).unsqueeze(1) ** self.deg_idx
        Y = Y * radial_factor

        return Y