"""Training_test of the SIRENSH network
jhonr"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'Data', 'Samples_10-2_1k_r0_train.parquet')

    df = pd.read_parquet(data_path)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    scaler = SHSirenScaler(r_scale=6378136.3)
    scaler.fit_potential(train_df["dV_m2_s2"].values)

    for subdf in [train_df, val_df]:
        subdf["a_theta_mps2"] = subdf["dg_theta_mGal"] * 1e-5
        subdf["a_phi_mps2"] = subdf["dg_phi_mGal"] * 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def df_to_tensors(df):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32)
        r = torch.tensor(df["radius_m"].values, dtype=torch.float32)
        y = torch.tensor(df["dV_m2_s2"].values, dtype=torch.float32).unsqueeze(1)
        a_true = torch.tensor(
            df[["a_theta_mps2", "a_phi_mps2"]].values,
            dtype=torch.float32
        )
        return lon, lat, r, y, a_true

    lon_train, lat_train, r_train, y_train, a_train = df_to_tensors(train_df)
    lon_val, lat_val, r_val, y_val, a_val = df_to_tensors(val_df)

    train_dataset = TensorDataset(lon_train, lat_train, r_train, y_train, a_train)
    val_dataset = TensorDataset(lon_val, lat_val, r_val, y_val, a_val)

    # num_workers = max(1, os.cpu_count() // 2) uncomment in GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=10240,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        # persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=10240,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        # persistent_workers=True
    )

    lmax = 10
    hidden_features = 8
    hidden_layers = 2
    out_features = 1
    first_omega_0 = 20
    hidden_omega_0 = 1.0

    model = SH_SIREN(
        lmax=lmax,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        device=device,
        scaler=scaler
    )

    criterion_val = nn.MSELoss()  # for potential value loss
    alpha = 1.0
    beta = 50.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 500
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for lon_b, lat_b, r_b, y_b, a_true_b in train_loader:
            lon_b = lon_b.to(device)
            lat_b = lat_b.to(device)
            r_b = r_b.to(device)
            y_b = y_b.to(device)
            a_true_b = a_true_b.to(device)

            # === Diagnostic printout (only for first few batches) ===
            if epoch == 0 and train_loss == 0.0:  # only print once per epoch or batch
                r_bar = r_b / scaler.r_scale if scaler.r_scale else r_b
                print("──── DIAGNOSTIC ────")
                print(f"lon range: {lon_b.min():.3f} → {lon_b.max():.3f}")
                print(f"lat range: {lat_b.min():.3f} → {lat_b.max():.3f}")
                print(f"r range:   {r_b.min():.3e} → {r_b.max():.3e}")
                print(f"r_bar range: {r_bar.min():.6f} → {r_bar.max():.6f}")
                print(f"mean(r): {r_b.mean():.3e}  std(r): {r_b.std():.3e}")
                print(f"mean(y): {y_b.mean():.3e}  std(y): {y_b.std():.3e}")
                print("────────────────────")

            optimizer.zero_grad()

            # === Forward pass with gradients ===
            y_pred, grads_scaled = model(lon_b, lat_b, r_b, return_gradients=True, physical_units=False)

            # === Gradient diagnostics ===
            if epoch == 0 and train_loss == 0.0:
                with torch.no_grad():
                    grad_norms = [g.abs().mean().item() for g in grads_scaled]
                    print(f"grad mean norms (scaled): dU/dlon={grad_norms[0]:.3e}, "
                          f"dU/dlat={grad_norms[1]:.3e}")

            # Unscale gradients
            g_lon, g_lat, _ = scaler.unscale_acceleration(grads_scaled)  # g_r unused
            g_AD = torch.stack([g_lon, g_lat], dim=1)

            loss_val = criterion_val(y_pred, scaler.scale_potential(y_b))
            loss_grad = torch.mean((a_true_b + g_AD) ** 2)
            loss = alpha * loss_val + beta * loss_grad

            # Check for NaNs before backward
            if torch.isnan(loss):
                print("⚠️ NaN detected in loss. "
                      f"Loss_val={loss_val.item():.3e}, "
                      f"Loss_grad={loss_grad.item():.3e}")
                break

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0

        for lon_b, lat_b, r_b, y_b, a_true_b in val_loader:
            lon_b, lat_b, r_b = lon_b.to(device), lat_b.to(device), r_b.to(device)
            y_b, a_true_b = y_b.to(device), a_true_b.to(device)

            lon_b.requires_grad_(True)
            lat_b.requires_grad_(True)
            r_b.requires_grad_(True)

            # === Forward pass ===
            y_pred, grads_scaled = model(lon_b, lat_b, r_b, return_gradients=True, physical_units=False)

            # === Convert to physical gradients ===
            g_lon, g_lat, _ = scaler.unscale_acceleration(grads_scaled)
            g_AD = torch.stack([g_lon, g_lat], dim=1)

            # === Loss computation ===
            loss_val = criterion_val(y_pred, scaler.scale_potential(y_b))
            loss_grad = torch.mean((a_true_b + g_AD) ** 2)

            loss = alpha * loss_val + beta * loss_grad
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Time: {epoch_time:.1f}s")



    outputs_dir = os.path.join(base_dir, "Outputs")
    save_dir = os.path.join(outputs_dir, "Models")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}.pth")
    np.save(os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_val_losses.npy"), np.array(val_losses))

    torch.save({
        "state_dict": model.state_dict(),
        "scaler": {
            "r_scale": scaler.r_scale,
            "U_min": scaler.U_min,
            "U_max": scaler.U_max
        },
    }, save_path)

    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "lmax": lmax,
        "hidden_features": hidden_features,
        "hidden_layers": hidden_layers,
        "out_features": out_features,
        "first_omega_0": first_omega_0,
        "hidden_omega_0": hidden_omega_0,
        "scaler": {
            "r_scale": float(scaler.r_scale),
            "U_min": float(scaler.U_min),
            "U_max": float(scaler.U_max)
        },
        "training": {
            "epochs": epochs,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
        },
        "paths": {
            "model_file": save_path,
            "train_losses": os.path.join(save_dir, "train_losses.npy"),
            "val_losses": os.path.join(save_dir, "val_losses.npy"),
        }
    }

    config_path = os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    print(f"✅ Model configuration saved to: {config_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()