import os
import numpy as np
import matplotlib.pyplot as plt


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_dir = os.path.join(base_dir, 'SRC', 'Models')
outputs_dir = os.path.join(base_dir, 'Outputs')
os.makedirs(outputs_dir, exist_ok=True)

train_losses = np.load(os.path.join(model_dir, "train_losses.npy"))
val_losses = np.load(os.path.join(model_dir, "val_losses.npy"))

plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Training Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2, linestyle='--')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("Training vs Validation Loss", fontsize=14)
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()

# ---------------------------------------------------------------------
# 4. Save and show
# ---------------------------------------------------------------------
fig_path = os.path.join(outputs_dir, "loss_curve.png")
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"✅ Saved loss curve to: {fig_path}")