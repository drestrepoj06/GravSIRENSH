import os
import numpy as np
import matplotlib.pyplot as plt
import re

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
outputs_dir = os.path.join(base_dir, 'Outputs', 'Models')
figures_dir = os.path.join(base_dir, 'Outputs', 'Figures')
os.makedirs(figures_dir, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Find the most recent pair of loss files
# ---------------------------------------------------------------------
files = os.listdir(outputs_dir)

train_files = sorted(
    [f for f in files if f.startswith("sh_siren_pyshtools") and "train_losses" in f],
    key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)),
    reverse=True
)
val_files = sorted(
    [f for f in files if f.startswith("sh_siren_pyshtools") and "val_losses" in f],
    key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)),
    reverse=True
)

if not train_files or not val_files:
    raise FileNotFoundError("No recent loss files found in Outputs/Models")

latest_train = train_files[0]
latest_val = val_files[0]

print(f"📂 Latest train file: {latest_train}")
print(f"📂 Latest val file:   {latest_val}")

# ---------------------------------------------------------------------
# 3. Extract model characteristics (lmax and timestamp) from filename
# ---------------------------------------------------------------------
match = re.search(r"lmax(\d+)_(\d{8}_\d{6})", latest_train)
if match:
    lmax = match.group(1)
    timestamp = match.group(2)
else:
    lmax = "unknown"
    timestamp = "unknown"

# ---------------------------------------------------------------------
# 4. Load data and plot
# ---------------------------------------------------------------------
train_losses = np.load(os.path.join(outputs_dir, latest_train))
val_losses = np.load(os.path.join(outputs_dir, latest_val))

use_log_scale = True  # toggle

plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Training_test Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2, linestyle='--')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title(f"Training_test vs Validation Loss\nSIREN (lmax={lmax}, {timestamp})", fontsize=14)
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()

if use_log_scale and np.all(train_losses > 0) and np.all(val_losses > 0):
    plt.yscale('log')
    plt.ylabel("MSE Loss (log scale)", fontsize=12)

# ---------------------------------------------------------------------
# 5. Save figure with model info
# ---------------------------------------------------------------------
fig_path = os.path.join(figures_dir, f"loss_curve_lmax{lmax}_{timestamp}.png")
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"✅ Saved loss curve to: {fig_path}")
