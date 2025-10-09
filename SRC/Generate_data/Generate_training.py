import pyshtools as pysh
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# --- PARAMETERS ---
BATCH_SIZE = 10000
N_TOTAL = 100000

# --- DEFINE PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")

# ✅ Ensure directories exist (create if missing)
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "Training.parquet")

print(f"Data directory:      {DATA_DIR}")
print(f"Final output file:   {OUTPUT_FILE}\n")

# --- LOAD MODELS ---
clm = pysh.datasets.Earth.EGM2008(lmax=2190)
clm2 = pysh.datasets.Earth.EGM2008(lmax=2)

a = pysh.constants.Earth.wgs84.a.value
f = pysh.constants.Earth.wgs84.f.value

# --- FUNCTION TO COMPUTE ONE BATCH ---
def compute_batch(batch_idx, batch_size=BATCH_SIZE):
    lons = np.random.uniform(0, 360, batch_size)
    lats = np.degrees(np.arcsin(np.random.uniform(-1, 1, batch_size)))
    # r = np.full_like(lats, clm.r0)

    results_2190 = clm.expand(lat=lats, lon=lons, a=a, f=f)
    results_2 = clm2.expand(lat=lats, lon=lons, a=a, f=f)

    g_r     = (results_2190[:, 0] - results_2[:, 0]) * 1e5
    g_theta = (results_2190[:, 1] - results_2[:, 1]) * 1e5
    g_phi   = (results_2190[:, 2] - results_2[:, 2]) * 1e5

    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'g_r_mGal': g_r.astype('float32'),
        'g_theta_mGal': g_theta.astype('float32'),
        'g_phi_mGal': g_phi.astype('float32'),
    })

    batch_file = os.path.join(DATA_DIR, f"gravity_batch_{batch_idx:03d}.parquet")
    df.to_parquet(batch_file, index=False)
    return batch_file


# --- PARALLEL BATCH EXECUTION ---
n_batches = N_TOTAL // BATCH_SIZE
print(f"Generating {N_TOTAL:,} samples in {n_batches} batches of {BATCH_SIZE:,} each...\n")

with tqdm_joblib(tqdm(desc="Processing batches", total=n_batches)):
    batch_files = Parallel(n_jobs=os.cpu_count())(
        delayed(compute_batch)(i) for i in range(n_batches)
    )

# --- MERGE ALL BATCHES ---
print("\nMerging batches into single dataset...")
df_list = [pd.read_parquet(f) for f in batch_files]
df = pd.concat(df_list, ignore_index=True)

df['g_total_mGal'] = np.sqrt(
    df['g_r_mGal']**2 + df['g_theta_mGal']**2 + df['g_phi_mGal']**2
)

df.to_parquet(OUTPUT_FILE, index=False)
print(f"✅ Saved full dataset to: {OUTPUT_FILE}")

# --- CLEAN-UP ---
for f in batch_files:
    os.remove(f)
print("🧹 Temporary batch files removed.")
