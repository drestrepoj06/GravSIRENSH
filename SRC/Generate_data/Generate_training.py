"""Generate 5 M random samples on the sphere that contain lon, lat, r, potential and acceleration from
EGM2008
jhonr"""

import os
os.environ["OMP_NUM_THREADS"] = "8" 
import pyshtools as pysh
import pandas as pd
import numpy as np

LMAX_FULL = 2190
LMAX_BASE = 2
N_SAMPLES = 5_000_000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, f"Samples_{LMAX_FULL}_5M_r0_complete.parquet")

clm_full = pysh.datasets.Earth.EGM2008(lmax=LMAX_FULL)
clm_low  = pysh.datasets.Earth.EGM2008(lmax=LMAX_BASE)

r0 = clm_full.r0
r1 = r0 # + 420000.0   # 420 km altitude

deg_full = np.arange(LMAX_FULL + 1)
deg_low  = np.arange(LMAX_BASE + 1)

scale_V_full = (r0 / r1) ** deg_full
scale_V_low  = (r0 / r1) ** deg_low

scale_g_full = (r0 / r1) ** (deg_full + 2)
scale_g_low  = (r0 / r1) ** (deg_low + 2)

scale_V_full_3d = scale_V_full[np.newaxis, :, np.newaxis]
scale_V_low_3d  = scale_V_low[np.newaxis, :, np.newaxis]
scale_g_full_3d = scale_g_full[np.newaxis, :, np.newaxis]
scale_g_low_3d  = scale_g_low[np.newaxis, :, np.newaxis]

clm_full_scaled_V = clm_full.copy()
clm_low_scaled_V  = clm_low.copy()
clm_full_scaled_g = clm_full.copy()
clm_low_scaled_g  = clm_low.copy()

clm_full_scaled_V.coeffs *= scale_V_full_3d
clm_low_scaled_V.coeffs  *= scale_V_low_3d

clm_full_scaled_g.coeffs *= scale_g_full_3d
clm_low_scaled_g.coeffs  *= scale_g_low_3d

res_full = clm_full_scaled_g.expand(lmax=LMAX_FULL)
res_low  = clm_low_scaled_g.expand(lmax=LMAX_FULL)

res_full_V = clm_full_scaled_V.expand(lmax=LMAX_FULL)
res_low_V  = clm_low_scaled_V.expand(lmax=LMAX_FULL)

pot_full = res_full_V.pot.data
pot_low = res_low_V.pot.data

dV = pot_full - pot_low

gr_full     = res_full.rad.data     * 1e5
gtheta_full = res_full.theta.data   * 1e5
gphi_full   = res_full.phi.data     * 1e5
gtotal_full = res_full.total.data   * 1e5

gr_low     = res_low.rad.data     * 1e5
gtheta_low = res_low.theta.data   * 1e5
gphi_low   = res_low.phi.data     * 1e5
gtotal_low = res_low.total.data   * 1e5

dg_r     = gr_full - gr_low
dg_theta = gtheta_full - gtheta_low
dg_phi   = gphi_full - gphi_low
dg_total = np.sqrt(dg_r**2 + dg_theta**2 + dg_phi**2)

lats = res_full.pot.lats()
lons = res_full.pot.lons()
lat_grid = np.repeat(lats, len(lons))
lon_grid = np.tile(lons, len(lats))

df = pd.DataFrame({
    "lat": lat_grid.astype("float32"),
    "lon": lon_grid.astype("float32"),
    "V_full_m2_s2": pot_full.ravel().astype("float32"),
    "V_low_m2_s2": pot_low.ravel().astype("float32"),
    "dV_m2_s2": dV.ravel().astype("float32"),
    "gr_full_mGal": (gr_full.ravel()).astype("float32"),
    "gr_low_mGal": (gr_low.ravel()).astype("float32"),
    "dg_r_mGal": dg_r.ravel().astype("float32"),
    "gtheta_full_mGal": (gtheta_full.ravel()).astype("float32"),
    "gtheta_low_mGal": (gtheta_low.ravel()).astype("float32"),
    "dg_theta_mGal": dg_theta.ravel().astype("float32"),
    "gphi_full_mGal": (gphi_full.ravel()).astype("float32"),
    "gphi_low_mGal": (gphi_low.ravel()).astype("float32"),
    "dg_phi_mGal": dg_phi.ravel().astype("float32"),
    "dg_total_mGal": dg_total.ravel().astype("float32"),
    "radius_m": np.full(pot_full.size, r0, dtype="float32")
})

if len(df) > N_SAMPLES:
    weights = np.abs(np.cos(np.radians(df["lat"].values)))
    df = df.sample(n=N_SAMPLES, weights=weights, random_state=42).reset_index(drop=True)

df.to_parquet(OUTPUT_FILE, index=False)
print("Saved:", OUTPUT_FILE)
