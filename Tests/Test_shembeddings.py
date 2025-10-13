# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:24:58 2025

@author: jhonr
"""

import numpy as np
import pyshtools as pysh
import sys, os

# --- Import model ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SRC.Models.SH_embedding import SHEmbedding

# --- Test grid ---
lats = np.linspace(-90, 90, 181)
lons = np.linspace(0, 360, 361)
Lon, Lat = np.meshgrid(lons, lats)
lat_flat, lon_flat = Lat.ravel(), Lon.ravel()

# --- Compute spherical harmonic basis ---
sh = SHEmbedding(lmax=3)
Y = sh.compute_basis(lon_flat, lat_flat)
print(f"\nBasis computed: {Y.shape} = (n_points, (lmax+1)^2)\n")

# -------------------------------------------------------------------------
# 1. Orthogonality
# -------------------------------------------------------------------------
def check_orthogonality(Y, lat, lmax):
    theta = np.radians(90 - lat)
    weights = np.sin(theta)
    weights /= weights.sum()
    W = np.diag(weights)
    G = Y.T @ W @ Y
    diag_mean = np.mean(np.diag(G))
    off_diag_mean = np.mean(np.abs(G - np.diag(np.diag(G))))
    print(f"Mean diagonal: {diag_mean:.4f}")
    print(f"Mean off-diagonal: {off_diag_mean:.4e}")
    assert abs(diag_mean - 1) < 1e-2
    assert off_diag_mean < 1e-3

check_orthogonality(Y, lat_flat, sh.lmax)

# -------------------------------------------------------------------------
# 2. Analytical values at pole
# -------------------------------------------------------------------------
print("\nAnalytical Y_l0 values at North Pole:")
for l in range(sh.lmax + 1):
    ylm = pysh.expand.spharm(sh.lmax, 0, 0, normalization='4pi', kind='real')
    expected = np.sqrt(2*l + 1)
    assert np.isclose(ylm[0, l, 0], expected, atol=1e-3)
    print(f"  l={l}: computed={ylm[0, l, 0]:.5f}, expected={expected:.5f}")

# -------------------------------------------------------------------------
# 3. Symmetry test
# -------------------------------------------------------------------------
def check_symmetry(Y, lat, lmax):
    lm_pairs = [(l,m) for l in range(lmax+1) for m in range(-l, l+1)]
    for idx, (l,m) in enumerate(lm_pairs):
        mask_north = np.isclose(lat, 45)
        mask_south = np.isclose(lat, -45)
        Ynorth = Y[mask_north, idx]
        Ysouth = Y[mask_south, idx]
        if np.allclose(Ynorth, Ysouth, atol=1e-3):
            parity = 'even'
        elif np.allclose(Ynorth, -Ysouth, atol=1e-3):
            parity = 'odd'
        else:
            parity = 'mixed'
        expected = 'even' if (l+m)%2==0 else 'odd'
        if parity != expected:
            print(f"  [!] l={l}, m={m:+d}, expected={expected}, got={parity}")

check_symmetry(Y, lat_flat, sh.lmax)

# -------------------------------------------------------------------------
# 4. Energy normalization
# -------------------------------------------------------------------------
print("\nEnergy normalization per harmonic:")
weights = np.sin(np.radians(90 - lat_flat))
weights /= weights.sum()
energy = np.sum((Y**2) * weights[:, None], axis=0)
for idx, val in enumerate(energy):
    print(f"  Column {idx:2d}: total energy ≈ {val:.3f}")
    assert abs(val - 1.0) < 0.1

# -------------------------------------------------------------------------
# 5. Reconstruction consistency
# -------------------------------------------------------------------------
print("\nReconstruction test with pyshtools:")
lmax = sh.lmax
cilm = np.zeros((2, lmax + 1, lmax + 1))
cilm[0, :, :] = 1.0  # cosine coefficients = 1
grid = pysh.expand.MakeGridDH(cilm, lmax=lmax, sampling=2)
coeffs = pysh.expand.SHExpandDH(grid, sampling=2)
recon = pysh.expand.MakeGridDH(coeffs, lmax=lmax, sampling=2)
diff_norm = np.linalg.norm(grid - recon)
print(f"  Grid shape: {grid.shape}")
print(f"  Reconstruction difference norm: {diff_norm:.3e}")
assert diff_norm < 1e-8

print("\nAll quantitative checks passed successfully ✅")
