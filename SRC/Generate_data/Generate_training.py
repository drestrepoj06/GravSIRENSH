import pyshtools as pysh
import pandas as pd
import numpy as np
import os

LMAX_FULL = 2190     
LMAX_BASE = 2         
N_SAMPLES = 5_000_000 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, f"Samples_{LMAX_FULL}_5M.parquet")

clm_full = pysh.datasets.Earth.EGM2008(lmax=LMAX_FULL)
clm_low  = pysh.datasets.Earth.EGM2008(lmax=LMAX_BASE)

a = pysh.constants.Earth.wgs84.a.value
f = pysh.constants.Earth.wgs84.f.value

res_full = clm_full.expand(a=a, f=f, lmax=LMAX_FULL)
res_low  = clm_low.expand(a=a, f=f, lmax=LMAX_FULL, lmax_calc=LMAX_BASE)

V_full = res_full.pot.data
V_low  = res_low.pot.data
dV = V_full - V_low

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
    "dV_m2_s2": dV.flatten().astype("float32"),
    "dg_r_mGal": dg_r.flatten().astype("float32"),
    "dg_theta_mGal": dg_theta.flatten().astype("float32"),
    "dg_phi_mGal": dg_phi.flatten().astype("float32"),
    "dg_total_mGal": dg_total.flatten().astype("float32"),
})

if len(df) > N_SAMPLES:
    df = df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

df.to_parquet(OUTPUT_FILE, index=False)
