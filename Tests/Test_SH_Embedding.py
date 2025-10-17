"""
Validation of Torch closed-form spherical harmonics for different criteria, and against PySHTOOLS, 
when 4pi normalization is used.
jhonr
"""

import numpy as np
import pyshtools as pysh
import torch
import sys, os

# --- Import model ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SRC.Models.SH_embedding import SphericalHarmonics  

# --- Device setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Test grid ---
lats = np.linspace(-90, 90, 181)
lons = np.linspace(0, 360, 361)
Lon, Lat = np.meshgrid(lons, lats)
lat_flat, lon_flat = Lat.ravel(), Lon.ravel()

# --- Compute spherical harmonic basis (Torch closed-form) ---
lmax = 3
sh = SphericalHarmonics(lmax=lmax, device=device)

coords = torch.tensor(np.vstack((lon_flat, lat_flat)).T, dtype=torch.float32, device=device)
Y_torch = sh(coords)                # differentiable Torch tensor
Y = Y_torch.detach().cpu().numpy()  # for numerical tests

print(f"\nBasis computed: {Y.shape} = (n_points, (lmax+1)^2)\n")

# -------------------------------------------------------------------------
# 1. Orthogonality test
# -------------------------------------------------------------------------
def check_orthogonality(Y, lat, lons, lmax):
    theta = np.radians(90 - lat)
    dtheta = np.radians(180 / (len(np.unique(lat)) - 1))
    dphi   = np.radians(360 / (len(np.unique(lons)) - 1))
    weights = np.sin(theta) * dtheta * dphi

    G = Y.T @ (Y * weights[:, None])
    diag_mean = np.mean(np.diag(G))
    off_diag_mean = np.mean(np.abs(G - np.diag(np.diag(G))))
    print(f"Mean diagonal (should ≈ 4π): {diag_mean:.4f}")
    print(f"Mean off-diagonal: {off_diag_mean:.4e}")
    assert abs(diag_mean - 4*np.pi) < 1e-1
    assert off_diag_mean < 1e-2

check_orthogonality(Y, lat_flat, lons, lmax)

# -------------------------------------------------------------------------
# 2) Symmetry: assert parity Y_l^m(θ,φ) vs Y_l^m(π-θ,φ)
def check_symmetry(Y, lat, lmax):
    # parity: Y_l^m(π-θ, φ) = (-1)^(l+m) Y_l^m(θ, φ)
    lm_pairs = [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    mask_north = np.isclose(lat, 45)
    mask_south = np.isclose(lat, -45)

    n_bad = 0
    for idx, (l, m) in enumerate(lm_pairs):
        Ynorth = Y[mask_north, idx]
        Ysouth = Y[mask_south, idx]
        sign = 1.0 if ((l + m) % 2 == 0) else -1.0
        if not np.allclose(Ynorth, sign*Ysouth, atol=1e-3, rtol=0):
            n_bad += 1

    print(f"Symmetry parity mismatches: {n_bad}")
    assert n_bad == 0, "Found parity mismatches in Y_lm."

check_symmetry(Y, lat_flat, lmax)

# 3) Energy: EXPECT 4π (because normalization='4pi')
def check_energy(Y, lat_flat, lon_flat, lmax):
    n_lat = len(np.unique(lat_flat))
    n_lon = len(np.unique(lon_flat))
    lats = np.unique(lat_flat)
    dtheta = np.radians(180 / (n_lat - 1))
    dphi   = np.radians(360 / (n_lon - 1))

    # one weight per latitude band, scaled so Σ w_lat dθ dφ = 4π
    weights_lat = np.sin(np.radians(90 - lats))
    weights_lat *= (4.0 * np.pi) / (np.sum(weights_lat) * dtheta * dphi)

    # reshape → average over longitude to avoid overcounting
    Y_reshaped = Y.reshape(n_lat, n_lon, -1)
    Y2_mean_lon = np.mean(Y_reshaped**2, axis=1)

    # integrate over latitude bands
    energy = np.sum(Y2_mean_lon * (weights_lat[:, None] * dtheta * dphi), axis=0)

    target = 4.0 * np.pi
    print("\nEnergy normalization per harmonic (expect ~4π):")
    for idx, val in enumerate(energy):
        print(f"  Column {idx:2d}: total energy ≈ {val:.6f}")
        assert abs(val - target) < 1e-1  # tolerate grid quadrature error
check_energy(Y, lat_flat, lon_flat, lmax)

# -------------------------------------------------------------------------
# 4. Direct comparison vs PySHTOOLS at random points
# -------------------------------------------------------------------------
print("\nComparing Torch vs PySHTOOLS for random angles:")

rng = np.random.default_rng(42)
n_tests = 5
for i in range(n_tests):
    lon = float(rng.uniform(0, 360))
    lat = float(rng.uniform(-90, 90))

    # Torch evaluation
    coords_t = torch.tensor([[lon, lat]], dtype=torch.float32, device=device)
    Y_torch = sh(coords_t).detach().cpu().numpy().ravel()

    # PySHTOOLS reference
    ylm = pysh.expand.spharm(lmax, 90 - lat, lon, normalization="4pi", kind="real")
    Y_ref = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if m < 0:
                Y_ref.append(ylm[1, l, -m])   # sin component
            else:
                Y_ref.append(ylm[0, l, m])   # cos component
    Y_ref = np.array(Y_ref)

    # --- Diagnostic: correlation per harmonic ---
    corrs = []
    for j in range(len(Y_ref)):
        c = np.corrcoef([Y_torch[j]], [Y_ref[j]])[0, 1] if len(Y_ref) > 1 else np.nan
        corrs.append(c)
    print("Torch vs PySHTOOLS values:\n", np.round(np.c_[Y_torch, Y_ref], 5))

    # --- Relative error ---
    rel_err = np.abs(Y_torch - Y_ref) / (np.abs(Y_ref) + 1e-12)
    print(f"  (lon={lon:.1f}, lat={lat:.1f})  mean rel.err = {rel_err.mean():.2e}\n")

assert rel_err.mean() < 1e-5, "Torch SH values deviate too much from PySHTOOLS"

print("\n✅ All Torch vs PySHTOOLS validation checks passed successfully!\n")

# def check_energy_pyshtools(lmax=3, normalization="ortho"):
#     """
#     Robust energy check for PySHTOOLS real spherical harmonics (kind='real').
#     Avoids internal broadcasting pitfalls by evaluating spharm at scalar angles.
#     """
#     # 1° grid
#     lons = np.linspace(0.0, 360.0, 361)     # φ in degrees
#     lats = np.linspace(90.0, -90.0, 181)
#     thetas = 90.0 - lats                    # θ = colatitude in degrees
#     ntheta, nphi = len(thetas), len(lons)

#     # Allocate harmonic grids: [2, lmax+1, lmax+1, ntheta, nphi]
#     Y = np.zeros((2, lmax+1, lmax+1, ntheta, nphi), dtype=float)

#     # === Build the grid safely (scalar calls) ===
#     # For each grid node, call spharm once and place the result.
#     for i, theta in enumerate(thetas):
#         for j, phi in enumerate(lons):
#             y_ij = pysh.expand.spharm(
#                 lmax=lmax,
#                 theta=theta,   # scalar
#                 phi=phi,       # scalar
#                 normalization=normalization,
#                 kind="real",
#                 degrees=True
#             )                 # shape (2, lmax+1, lmax+1)
#             Y[:, :, :, i, j] = y_ij

#     # Integration weights
#     dtheta = np.radians(thetas[1] - thetas[0])
#     dphi   = np.radians(lons[1]   - lons[0])
#     # sinθ depends only on θ; broadcast across φ
#     weights = np.sin(np.radians(thetas))[:, None]

#     print(f"\nEnergy normalization per harmonic (normalization='{normalization}'):\n")

#     # Check energy per (l,m)
#     for l in range(lmax + 1):
#         for m in range(-l, l + 1):
#             # cos-part for m>=0, sin-part for m<0 (PySHTOOLS real form)
#             if m < 0:
#                 Ylm = Y[1, l, abs(m)]     # sine component
#             else:
#                 Ylm = Y[0, l, m]          # cosine component

#             energy = np.sum((Ylm**2) * weights) * dtheta * dphi
#             print(f"  l={l:2d}, m={m:3d}: energy ≈ {energy:.6f}")

#             # Expected energies by normalization
#             if normalization == "ortho":
#                 target = 1.0
#             elif normalization == "4pi":
#                 target = 4.0 * np.pi
#             elif normalization == "schmidt":
#                 target = 2.0 * np.pi / (2*l + 1)
#             else:
#                 # Fallback: don't assert for unknown schemes
#                 target = None

#             if target is not None:
#                 assert abs(energy - target) < 1e-2, (
#                     f"Energy mismatch for l={l}, m={m}: got {energy:.6f}, "
#                     f"expected {target:.6f}"
#                 )

#     print("\n✅ All harmonics match the expected energy for the chosen normalization.\n")

# check_energy_pyshtools(lmax=3, normalization="ortho")
