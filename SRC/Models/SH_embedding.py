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
        base *= math.sqrt(4.0 * math.pi)  # convert orthonormal → 4π normalized
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
    """
    Differentiable spherical harmonics embedding with optional radial scaling (r_ref/r)^l.
    """

    def __init__(self, lmax: int = 10, device: str = "cuda", r_ref: float = 6371e3):
        """
        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        device : str
            Device for computation.
        r_ref : float
            Reference radius (m).
        """
        super().__init__()
        self.lmax = lmax
        self.device = device
        self.r_ref = r_ref
        self.embedding_dim = (lmax + 1) ** 2

        # precompute degree indices for each harmonic term
        deg_list = []
        for l in range(lmax + 1):
            deg_list.extend([l] * (2 * l + 1))
        self.register_buffer("deg_idx", torch.tensor(deg_list, dtype=torch.float32))

    def forward(self, lonlatr: torch.Tensor) -> torch.Tensor:
        """
        Compute SH embedding for input [lon, lat, r] (degrees, meters).

        Parameters
        ----------
        lonlatr : (N, 3) tensor [lon, lat, r]
                  or (N, 2) tensor [lon, lat] → assumes r = r_ref

        Returns
        -------
        (N, (lmax+1)^2) tensor with real SH values scaled by (r_ref / r)^l.
        """
        if lonlatr.shape[1] == 2:
            lon, lat = lonlatr[:, 0], lonlatr[:, 1]
            r = torch.ones_like(lon) * self.r_ref
        else:
            lon, lat, r = lonlatr[:, 0], lonlatr[:, 1], lonlatr[:, 2]

        # Convert degrees → radians
        phi = torch.deg2rad(lon + 180.0)   # 0–360° longitude
        theta = torch.deg2rad(90.0 - lat)  # 0–180° colatitude
        theta = torch.clamp(theta, 1e-7, math.pi - 1e-7) 

        # Angular harmonics
        Y = []
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                y = SH(m, l, phi, theta)
                Y.append(y)
        Y = torch.stack(Y, dim=-1).to(self.device)

        # Radial dependence (r_ref / r)^l
        radial_factor = (self.r_ref / r).unsqueeze(1) ** self.deg_idx
        Y = Y * radial_factor

        return Y

    def from_dataframe(self, df, lon_col="lon", lat_col="lat", r_col="radius_m"):
        """Convenience method to build embedding directly from a DataFrame."""
        lon = torch.tensor(df[lon_col].values, dtype=torch.float32, device=self.device)
        lat = torch.tensor(df[lat_col].values, dtype=torch.float32, device=self.device)
        r   = torch.tensor(df[r_col].values, dtype=torch.float32, device=self.device)
        lonlatr = torch.stack([lon, lat, r], dim=-1)
        return self.forward(lonlatr)
